from __future__ import annotations

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


class ClaudeChatModel(ChatModelBase):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        stream: bool = False,
        thinking: dict | None = None,
        stream_tool_parsing: bool = True,
        client_kwargs: dict[str, Any] | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name, stream=stream, retry_policy=retry_policy,
        )
        import anthropic

        self._anthropic = anthropic
        ck = dict(client_kwargs or {})
        if base_url is not None:
            ck.setdefault("base_url", base_url)
        self.client = anthropic.AsyncAnthropic(api_key=api_key, **ck)
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.stream_tool_parsing = stream_tool_parsing
        self.generate_kwargs = generate_kwargs or {}

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
            return await self.client.messages.create(**api_kwargs)

        try:
            response = await retry_async(
                _op,
                policy=self.retry_policy,
                is_retryable=self._is_retryable,
                op_name=f"anthropic.messages[{self.model_name}]",
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
        schema = structured_model.model_json_schema()
        schema.pop("title", None)

        tool_name = "_structured_output"
        fake_tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": (
                    f"Return the answer as a structured "
                    f"{structured_model.__name__}."
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
                    tools=[fake_tool_schema],
                    tool_choice=tool_name,
                    **kwargs,
                )
            except ModelError as e:
                if e.status_code != 400:
                    raise
                return await super()._structured(msg, structured_model, **kwargs)
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
        system_text: str | None = None
        body_msgs: list[ChatMessageBase] = []
        for m in msg:
            if m.role == "system":
                system_text = (
                    (system_text + "\n" if system_text else "") + m.text
                )
            else:
                body_msgs.append(m)

        claude_messages: list[dict] = []
        for m in body_msgs:
            claude_messages.extend(m.to_claude())

        api_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": claude_messages,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
        }
        if system_text:
            api_kwargs["system"] = system_text
        if self.thinking and "thinking" not in api_kwargs:
            api_kwargs["thinking"] = self.thinking

        if tools:
            api_kwargs["tools"] = [_openai_tool_to_claude(t) for t in tools]
        if tool_choice is not None:
            if tool_choice == "any":
                warnings.warn(
                    '"any" is deprecated; use "required".',
                    DeprecationWarning,
                    stacklevel=2,
                )
                tool_choice = "required"
            self._validate_tool_choice(tool_choice, tools)
            api_kwargs["tool_choice"] = _tool_choice_to_claude(tool_choice)

        return api_kwargs

    def _parse_non_stream(self, response: Any, start: float) -> ChatResponse:
        if not hasattr(response, "content"):
            raise ModelBehaviorError(
                "Anthropic response missing 'content'",
                provider="anthropic",
            )

        content = ""
        reasoning = ""
        tool_calls: list[ToolCall] = []

        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                reasoning += getattr(block, "thinking", "")
            elif btype == "text":
                content += getattr(block, "text", "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input or {}),
                    ),
                )

        usage = None
        if getattr(response, "usage", None):
            usage = ChatUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=getattr(response.usage, "output_tokens", 0),
                runtime=time.time() - start,
                metadata=try_model_dump(response.usage),
            )

        response_id = getattr(response, "id", None)
        assistant = ChatMessageBase.assistant(
            content, reasoning_content=reasoning, tool_calls=tool_calls,
        )
        if response_id:
            assistant.id = response_id

        return ChatResponse(
            message=assistant,
            id=response_id or assistant.id,
            usage=usage,
            finish_reason=getattr(response, "stop_reason", None),
        )

    async def _iter_stream(
        self,
        response: Any,
        start: float,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        content = ""
        reasoning = ""
        tool_slots: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        usage: ChatUsage | None = None
        response_id: str | None = None
        stop_reason: str | None = None

        try:
            async for event in response:
                ev_type = getattr(event, "type", None)

                if ev_type == "message_start":
                    message = event.message
                    response_id = getattr(message, "id", None) or response_id
                    if getattr(message, "usage", None):
                        usage = ChatUsage(
                            input_tokens=message.usage.input_tokens,
                            output_tokens=getattr(
                                message.usage, "output_tokens", 0,
                            ),
                            runtime=time.time() - start,
                        )

                elif ev_type == "content_block_start":
                    cb = event.content_block
                    if getattr(cb, "type", None) == "tool_use":
                        tool_slots[event.index] = {
                            "id": cb.id,
                            "name": cb.name,
                            "input": "",
                        }

                elif ev_type == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    delta_text = ""
                    delta_reasoning = ""
                    if dtype == "text_delta":
                        delta_text = delta.text
                        content += delta_text
                    elif dtype == "thinking_delta":
                        delta_reasoning = delta.thinking
                        reasoning += delta_reasoning
                    elif dtype == "input_json_delta" and event.index in tool_slots:
                        tool_slots[event.index]["input"] += delta.partial_json or ""

                    if delta_text or delta_reasoning:
                        yield ChatStreamChunk(
                            message=ChatMessageBase.assistant(
                                content, reasoning_content=reasoning,
                            ),
                            delta_text=delta_text,
                            delta_reasoning=delta_reasoning,
                            is_last=False,
                            id=response_id or make_resp_id(),
                        )

                elif ev_type == "message_delta":
                    if getattr(event, "usage", None) and usage:
                        usage.output_tokens = event.usage.output_tokens
                        usage.runtime = time.time() - start
                    stop_reason = getattr(
                        getattr(event, "delta", None), "stop_reason", stop_reason,
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
            usage=usage,
            finish_reason=stop_reason,
        )

    def _is_retryable(self, exc: BaseException) -> bool:
        ant = self._anthropic
        retryable_types: tuple[type, ...] = tuple(
            t
            for t in (
                getattr(ant, "APIConnectionError", None),
                getattr(ant, "APITimeoutError", None),
                getattr(ant, "InternalServerError", None),
                getattr(ant, "RateLimitError", None),
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
            f"Anthropic request failed: {exc}",
            provider="anthropic",
            status_code=status if isinstance(status, int) else None,
            cause=exc,
        )


def _openai_tool_to_claude(schema: dict) -> dict:
    fn = schema.get("function") or {}
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _tool_choice_to_claude(tool_choice: str) -> dict:
    mapping = {
        "auto": {"type": "auto"},
        "none": {"type": "none"},
        "required": {"type": "any"},
    }
    if tool_choice in mapping:
        return mapping[tool_choice]
    return {"type": "tool", "name": tool_choice}


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
