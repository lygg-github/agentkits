from __future__ import annotations

import os
import time
import warnings
from collections import OrderedDict
from typing import Any, AsyncGenerator, Type, TypeVar

from pydantic import BaseModel

from ..exceptions import ModelBehaviorError, ModelError
from ..message import ChatMessageBase, ChatUsage, ToolCall
from ..types import ToolChoice
from ..utils._id import make_resp_id
from ..utils._json import safe_json_loads, try_model_dump
from ..utils._retry import RetryPolicy, retry_async
from ._base import ChatModelBase, _finalize_structured
from ._response import ChatResponse, ChatStreamChunk


T = TypeVar("T", bound=BaseModel)


class OpenAIChatModel(ChatModelBase):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        stream_tool_parsing: bool = True,
        organization: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name, stream=stream, retry_policy=retry_policy,
        )
        import openai

        self._openai = openai
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **(client_kwargs or {}),
        )
        self.stream_tool_parsing = stream_tool_parsing
        self.generate_kwargs = generate_kwargs or {}
        self._structured_tool_fallback = False

    @classmethod
    def from_ali_env(
        cls,
        *,
        model_name: str = "qwen3-max",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> "OpenAIChatModel":
        key = api_key or os.environ.get("ALI_API_KEY") or os.environ.get("ali_api_key")
        if not key:
            raise ValueError("Set ALI_API_KEY or ali_api_key first.")
        return cls(model_name=model_name, api_key=key, base_url=base_url, **kwargs)

    async def chat(
        self,
        msg: list[ChatMessageBase],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        structured_model: Type[T] | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatStreamChunk, None]:
        if structured_model is not None:
            return await self._structured(msg, structured_model, **kwargs)
        return await self._raw_chat(msg, tools=tools, tool_choice=tool_choice, **kwargs)

    async def _raw_chat(
        self,
        msg: list[ChatMessageBase],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatStreamChunk, None]:
        api_kwargs = self._build_api_kwargs(msg, tools, tool_choice, **kwargs)
        start = time.time()

        async def _op() -> Any:
            return await self.client.chat.completions.create(**api_kwargs)

        try:
            response = await retry_async(
                _op,
                policy=self.retry_policy,
                is_retryable=self._is_retryable,
                op_name=f"openai.chat[{self.model_name}]",
            )
        except Exception as e:
            raise self._wrap_error(e) from e

        if self.stream:
            return self._iter_stream(response, start)
        return self._parse_non_stream(response, start)

    async def _structured(
        self,
        msg: list[ChatMessageBase],
        structured_model: Type[T],
        **kwargs: Any,
    ) -> ChatResponse:
        if getattr(self, "_structured_tool_fallback", False):
            return await self._structured_via_tool(msg, structured_model, **kwargs)

        start = time.time()
        try:
            response = await self._structured_via_parse(
                msg, structured_model, start=start, **kwargs,
            )
            return _finalize_structured(response, structured_model)
        except ModelError as e:
            if e.status_code != 400:
                raise
            self._structured_tool_fallback = True
            return await self._structured_via_tool(msg, structured_model, **kwargs)

    async def _structured_via_parse(
        self,
        msg: list[ChatMessageBase],
        structured_model: Type[T],
        *,
        start: float,
        **kwargs: Any,
    ) -> ChatResponse:
        openai_messages: list[dict] = []
        for m in msg:
            openai_messages.extend(m.to_openai())

        api_kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
            "response_format": structured_model,
            **self.generate_kwargs,
            **kwargs,
        }

        async def _op() -> Any:
            return await self.client.beta.chat.completions.parse(**api_kwargs)

        try:
            completion = await retry_async(
                _op,
                policy=self.retry_policy,
                is_retryable=self._is_retryable,
                op_name=f"openai.parse[{self.model_name}]",
            )
        except Exception as e:
            raise self._wrap_error(e) from e

        response = self._parse_non_stream(completion, start)
        parsed = None
        try:
            parsed = completion.choices[0].message.parsed
        except Exception:
            parsed = None
        if parsed is None:
            raw_text = response.message.text
            parsed = safe_json_loads(raw_text, default=None)
        response.parsed = parsed
        return response

    async def _structured_via_tool(
        self,
        msg: list[ChatMessageBase],
        structured_model: Type[T],
        **kwargs: Any,
    ) -> ChatResponse:
        tool_name = "generate_structured_output"
        schema = structured_model.model_json_schema()
        schema.pop("title", None)
        fake_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    "Generate the required structured output by calling this "
                    f"function with a valid {structured_model.__name__} payload."
                ),
                "parameters": schema,
            },
        }

        prev_stream = self.stream
        self.stream = False
        try:
            try:
                response = await self._raw_chat(
                    msg,
                    tools=[fake_tool],
                    tool_choice=tool_name,
                    **kwargs,
                )
            except ModelError as e:
                if e.status_code != 400:
                    raise
                return await ChatModelBase._structured(
                    self, msg, structured_model, **kwargs,
                )
        finally:
            self.stream = prev_stream

        assert isinstance(response, ChatResponse)
        if response.message.tool_calls:
            response.parsed = response.message.tool_calls[0].input or {}
            response.message.tool_calls = []
        return _finalize_structured(response, structured_model)

    def _build_api_kwargs(
        self,
        msg: list[ChatMessageBase],
        tools: list[dict] | None,
        tool_choice: ToolChoice | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        openai_messages: list[dict] = []
        for m in msg:
            openai_messages.extend(m.to_openai())

        api_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = tools
        if tool_choice is not None:
            if tool_choice == "any":
                warnings.warn(
                    '"any" is deprecated; use "required".',
                    DeprecationWarning,
                    stacklevel=2,
                )
                tool_choice = "required"
            self._validate_tool_choice(tool_choice, tools)
            api_kwargs["tool_choice"] = _tool_choice_to_openai(tool_choice)

        if self.stream:
            api_kwargs["stream_options"] = {"include_usage": True}
        return api_kwargs

    def _parse_non_stream(self, response: Any, start: float) -> ChatResponse:
        if not getattr(response, "choices", None):
            raise ModelBehaviorError(
                "OpenAI response contained no choices",
                provider="openai",
            )

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        response_id = getattr(response, "id", None)
        message = choice.message

        content = getattr(message, "content", None) or ""
        reasoning = _extract_reasoning(message) or ""

        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            args_str = tc.function.arguments or ""
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=safe_json_loads(args_str, default={}),
                    raw_input=args_str,
                ),
            )

        usage = None
        if getattr(response, "usage", None):
            usage = ChatUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                runtime=time.time() - start,
                metadata=try_model_dump(response.usage),
            )

        assistant = ChatMessageBase.assistant(
            content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )
        if response_id:
            assistant.id = response_id

        return ChatResponse(
            message=assistant,
            id=response_id or assistant.id,
            created=int(getattr(response, "created", time.time())),
            usage=usage,
            finish_reason=finish_reason,
        )

    async def _iter_stream(
        self,
        stream: Any,
        start: float,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        content = ""
        reasoning = ""
        tool_slots: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        usage: ChatUsage | None = None
        response_id: str | None = None
        finish_reason: str | None = None
        created_ts = 0

        try:
            async for chunk in stream:
                if response_id is None:
                    response_id = getattr(chunk, "id", None)
                if not created_ts:
                    created_ts = int(getattr(chunk, "created", 0) or 0)

                if getattr(chunk, "usage", None):
                    usage = ChatUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                        runtime=time.time() - start,
                        metadata=try_model_dump(chunk.usage),
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason or finish_reason

                delta_text = getattr(delta, "content", None) or ""
                delta_reasoning = _extract_reasoning(delta) or ""
                content += delta_text
                reasoning += delta_reasoning

                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = tool_slots.setdefault(
                        tc.index,
                        {"id": None, "name": None, "input": ""},
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["input"] += tc.function.arguments

                if delta_text or delta_reasoning:
                    yield ChatStreamChunk(
                        message=ChatMessageBase.assistant(
                            content, reasoning_content=reasoning,
                        ),
                        delta_text=delta_text,
                        delta_reasoning=delta_reasoning,
                        is_last=False,
                        id=response_id or make_resp_id(),
                        created=created_ts,
                        finish_reason=finish_reason,
                    )
        except Exception as e:
            raise self._wrap_error(e) from e

        final = _finalize_assistant(content, reasoning, tool_slots, response_id)
        yield ChatStreamChunk(
            message=final,
            delta_text="",
            delta_reasoning="",
            is_last=True,
            id=response_id or final.id,
            created=created_ts,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _is_retryable(self, exc: BaseException) -> bool:
        openai = self._openai
        retryable_types: tuple[type, ...] = tuple(
            t
            for t in (
                getattr(openai, "APIConnectionError", None),
                getattr(openai, "APITimeoutError", None),
                getattr(openai, "InternalServerError", None),
                getattr(openai, "RateLimitError", None),
            )
            if t is not None
        )
        if retryable_types and isinstance(exc, retryable_types):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status >= 500

    def _wrap_error(self, exc: BaseException) -> ModelError:
        if isinstance(exc, ModelError):
            return exc
        status = getattr(exc, "status_code", None)
        return ModelError(
            f"OpenAI request failed: {exc}",
            provider="openai",
            status_code=status if isinstance(status, int) else None,
            cause=exc,
        )


def _tool_choice_to_openai(tool_choice: str) -> Any:
    if tool_choice in {"auto", "none", "required"}:
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _extract_reasoning(obj: Any) -> str | None:
    val = getattr(obj, "reasoning_content", None)
    if isinstance(val, str):
        return val
    val = getattr(obj, "reasoning", None)
    if isinstance(val, str):
        return val
    return None


def _finalize_assistant(
    content: str,
    reasoning: str,
    tool_slots: "OrderedDict[int, dict[str, Any]]",
    response_id: str | None,
) -> ChatMessageBase:
    tool_calls: list[ToolCall] = []
    for slot in tool_slots.values():
        if not slot["id"] and not slot["name"]:
            continue
        raw = slot["input"]
        tool_calls.append(
            ToolCall(
                id=slot["id"] or "",
                name=slot["name"] or "",
                input=safe_json_loads(raw, default={}),
                raw_input=raw,
            ),
        )
    assistant = ChatMessageBase.assistant(
        content, reasoning_content=reasoning, tool_calls=tool_calls,
    )
    if response_id:
        assistant.id = response_id
    return assistant
