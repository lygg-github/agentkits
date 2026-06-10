from __future__ import annotations

import time
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


class OpenAIResponsesModel(ChatModelBase):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        stream: bool = False,
        stream_tool_parsing: bool = True,
        previous_response_id: str | None = None,
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
        self.previous_response_id = previous_response_id
        self._structured_tool_fallback = False

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
        input_items, system_text = _messages_to_responses_input(msg)

        api_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
            "stream": self.stream,
            **self.generate_kwargs,
            **kwargs,
        }
        if system_text and "instructions" not in api_kwargs:
            api_kwargs["instructions"] = system_text
        if self.previous_response_id and "previous_response_id" not in api_kwargs:
            api_kwargs["previous_response_id"] = self.previous_response_id

        if tools:
            api_kwargs["tools"] = [_openai_fn_tool_to_responses(t) for t in tools]
        if tool_choice is not None:
            self._validate_tool_choice(tool_choice, tools)
            api_kwargs["tool_choice"] = _tool_choice_to_responses(tool_choice)

        start = time.time()

        async def _op() -> Any:
            return await self.client.responses.create(**api_kwargs)

        try:
            response = await retry_async(
                _op,
                policy=self.retry_policy,
                is_retryable=self._is_retryable,
                op_name=f"openai.responses[{self.model_name}]",
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
        if self._structured_tool_fallback:
            return await self._structured_via_tool(msg, structured_model, **kwargs)

        schema = structured_model.model_json_schema()
        schema.pop("title", None)

        text_cfg = {
            "format": {
                "type": "json_schema",
                "name": structured_model.__name__,
                "schema": schema,
                "strict": False,
            },
        }

        prev_stream = self.stream
        self.stream = False
        try:
            try:
                response = await self._raw_chat(msg, text=text_cfg, **kwargs)
            except ModelError as e:
                if e.status_code != 400:
                    raise
                self._structured_tool_fallback = True
                return await self._structured_via_tool(
                    msg, structured_model, **kwargs,
                )
        finally:
            self.stream = prev_stream

        assert isinstance(response, ChatResponse)
        return _finalize_structured(response, structured_model)

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
            response = await self._raw_chat(
                msg,
                tools=[fake_tool],
                tool_choice=tool_name,
                **kwargs,
            )
        finally:
            self.stream = prev_stream

        assert isinstance(response, ChatResponse)
        if response.message.tool_calls:
            response.parsed = response.message.tool_calls[0].input or {}
            response.message.tool_calls = []
        return _finalize_structured(response, structured_model)

    def _parse_non_stream(self, response: Any, start: float) -> ChatResponse:
        text = ""
        reasoning = ""
        tool_calls: list[ToolCall] = []

        for item in getattr(response, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "message":
                for part in getattr(item, "content", None) or []:
                    ptype = getattr(part, "type", None)
                    if ptype in ("output_text", "text"):
                        text += getattr(part, "text", "") or ""
            elif itype == "reasoning":
                for s in getattr(item, "summary", None) or []:
                    reasoning += getattr(s, "text", "") or ""
                for c in getattr(item, "content", None) or []:
                    if getattr(c, "type", None) == "reasoning_text":
                        reasoning += getattr(c, "text", "") or ""
            elif itype == "function_call":
                args = getattr(item, "arguments", "") or ""
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                        name=getattr(item, "name", ""),
                        input=safe_json_loads(args, default={}),
                        raw_input=args,
                    ),
                )

        usage = None
        resp_usage = getattr(response, "usage", None)
        if resp_usage is not None:
            input_tokens = getattr(resp_usage, "input_tokens", None)
            output_tokens = getattr(resp_usage, "output_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(resp_usage, "prompt_tokens", 0)
            if output_tokens is None:
                output_tokens = getattr(resp_usage, "completion_tokens", 0)
            usage = ChatUsage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                runtime=time.time() - start,
                metadata=try_model_dump(resp_usage),
            )

        response_id = getattr(response, "id", None)
        if response_id:
            self.previous_response_id = response_id

        assistant = ChatMessageBase.assistant(
            text, reasoning_content=reasoning, tool_calls=tool_calls,
        )
        if response_id:
            assistant.id = response_id

        return ChatResponse(
            message=assistant,
            id=response_id or assistant.id,
            created=int(getattr(response, "created_at", time.time())),
            usage=usage,
            finish_reason=getattr(response, "status", None),
        )

    async def _iter_stream(
        self,
        stream: Any,
        start: float,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        text = ""
        reasoning = ""
        tool_slots: dict[int, dict[str, Any]] = {}
        usage: ChatUsage | None = None
        response_id: str | None = None

        try:
            async for event in stream:
                etype = getattr(event, "type", "")

                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    text += delta
                    if delta:
                        yield ChatStreamChunk(
                            message=ChatMessageBase.assistant(
                                text, reasoning_content=reasoning,
                            ),
                            delta_text=delta,
                            is_last=False,
                            id=response_id or make_resp_id(),
                        )

                elif etype in (
                    "response.reasoning.delta",
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                ):
                    delta_r = getattr(event, "delta", "") or ""
                    reasoning += delta_r
                    if delta_r:
                        yield ChatStreamChunk(
                            message=ChatMessageBase.assistant(
                                text, reasoning_content=reasoning,
                            ),
                            delta_text="",
                            delta_reasoning=delta_r,
                            is_last=False,
                            id=response_id or make_resp_id(),
                        )

                elif etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", None) == "function_call":
                        idx = getattr(event, "output_index", 0)
                        tool_slots[idx] = {
                            "id": getattr(item, "call_id", None) or getattr(item, "id", None),
                            "name": getattr(item, "name", None),
                            "input": "",
                        }
                elif etype == "response.function_call_arguments.delta":
                    idx = getattr(event, "output_index", 0)
                    if idx in tool_slots:
                        tool_slots[idx]["input"] += getattr(event, "delta", "") or ""

                elif etype in ("response.completed", "response.done"):
                    resp = getattr(event, "response", None)
                    if resp is not None:
                        response_id = getattr(resp, "id", None) or response_id
                        resp_usage = getattr(resp, "usage", None)
                        if resp_usage is not None:
                            usage = ChatUsage(
                                input_tokens=getattr(resp_usage, "input_tokens", 0) or 0,
                                output_tokens=getattr(resp_usage, "output_tokens", 0) or 0,
                                runtime=time.time() - start,
                                metadata=try_model_dump(resp_usage),
                            )
                elif etype == "response.created":
                    resp = getattr(event, "response", None)
                    if resp is not None:
                        response_id = getattr(resp, "id", None) or response_id
        except Exception as e:
            raise self._wrap_error(e) from e

        if response_id:
            self.previous_response_id = response_id

        final = _finalize_assistant(text, reasoning, tool_slots, response_id)
        yield ChatStreamChunk(
            message=final,
            delta_text="",
            delta_reasoning="",
            is_last=True,
            id=response_id or final.id,
            usage=usage,
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
            f"OpenAI Responses request failed: {exc}",
            provider="openai-responses",
            status_code=status if isinstance(status, int) else None,
            cause=exc,
        )


def _messages_to_responses_input(
    msgs: list[ChatMessageBase],
) -> tuple[list[dict], str]:
    from ..message._content import BinaryContent

    input_items: list[dict] = []
    system_parts: list[str] = []

    for m in msgs:
        if m.role == "system":
            if m.text:
                system_parts.append(m.text)
            continue

        if m.role == "tool":
            for tr in m.tool_results:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tr.id,
                        "output": tr.text,
                    },
                )
            continue

        parts: list[dict] = []
        for item in m.content:
            if isinstance(item, str):
                if item:
                    kind = "output_text" if m.role == "assistant" else "input_text"
                    parts.append({"type": kind, "text": item})
            elif isinstance(item, BinaryContent):
                parts.append(_binary_to_responses_part(item, role=m.role))

        if parts:
            input_items.append({"type": "message", "role": m.role, "content": parts})

        for tc in m.tool_calls:
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": tc.raw_input or _dumps(tc.input or {}),
                },
            )

    return input_items, "\n".join(system_parts)


def _binary_to_responses_part(b: "BinaryContent", *, role: str) -> dict:
    if b.kind == "image":
        if b.url is not None:
            return {"type": "input_image", "image_url": b.url}
        if b.data is not None:
            return {
                "type": "input_image",
                "image_url": f"data:{b.media_type};base64,{b.data}",
            }
        return {"type": "input_text", "text": f"[image file:{b.file_id}]"}

    if b.kind == "file" and b.file_id:
        return {"type": "input_file", "file_id": b.file_id}

    ref = b.url or b.file_id or "<binary>"
    return {"type": "input_text", "text": f"[{b.kind}: {ref}]"}


def _openai_fn_tool_to_responses(schema: dict) -> dict:
    fn = schema.get("function") or {}
    return {
        "type": "function",
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _tool_choice_to_responses(tool_choice: str) -> Any:
    if tool_choice in {"auto", "none", "required"}:
        return tool_choice
    return {"type": "function", "name": tool_choice}


def _finalize_assistant(
    content: str,
    reasoning: str,
    tool_slots: dict[int, dict[str, Any]],
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


def _dumps(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"
