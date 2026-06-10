from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Type, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..exceptions import ModelBehaviorError
from ..message import ChatMessageBase
from ..types import ToolChoice
from ..utils._async import maybe_await
from ..utils._json import safe_json_loads
from ..utils._retry import RetryPolicy
from ._response import ChatResponse, ChatStreamChunk


_TOOL_CHOICE_MODES = {"auto", "none", "required"}

OnMessageCb = Callable[[ChatMessageBase], Awaitable[None] | None]
OnChunkCb = Callable[[ChatStreamChunk], Awaitable[None] | None]

T = TypeVar("T", bound=BaseModel)


class ChatModelBase(ABC):
    model_name: str
    stream: bool
    retry_policy: RetryPolicy

    def __init__(
        self,
        model_name: str,
        stream: bool,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.model_name = model_name
        self.stream = stream
        self.retry_policy = retry_policy or RetryPolicy()

    async def close(self) -> None:
        client = getattr(self, "client", None)
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is not None:
            res = close()
            if hasattr(res, "__await__"):
                await res

    async def __aenter__(self) -> "ChatModelBase":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @abstractmethod
    async def chat(
        self,
        msg: list[ChatMessageBase],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatStreamChunk, None]:
        raise NotImplementedError

    async def chat_cb(
        self,
        msg: list[ChatMessageBase],
        on_message: OnMessageCb,
        on_chunk: OnChunkCb | None = None,
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatMessageBase:
        result = await self.chat(
            msg, tools=tools, tool_choice=tool_choice, **kwargs,
        )

        if isinstance(result, ChatResponse):
            _attach_usage_metadata(result.message, result)
            if result.parsed is not None:
                md = result.message.metadata or {}
                md.setdefault("parsed", result.parsed)
                result.message.metadata = md
            await maybe_await(on_message, result.message)
            return result.message

        final: ChatMessageBase | None = None
        last_chunk: ChatStreamChunk | None = None
        async for chunk in result:
            if on_chunk is not None:
                await maybe_await(on_chunk, chunk)
            final = chunk.message
            last_chunk = chunk

        if final is None:
            final = ChatMessageBase.assistant("")
        if last_chunk is not None:
            _attach_usage_metadata(final, last_chunk)

        await maybe_await(on_message, final)
        return final

    async def _structured(
        self,
        msg: list[ChatMessageBase],
        structured_model: Type[T],
        **kwargs: Any,
    ) -> ChatResponse:
        schema = structured_model.model_json_schema()
        schema.pop("title", None)
        instruction = ChatMessageBase.system(
            "Respond with ONLY a single JSON object matching this schema, "
            "no prose, no markdown fences:\n\n"
            + json.dumps(schema, ensure_ascii=False, indent=2),
        )
        augmented = list(msg) + [instruction]

        prev_stream = self.stream
        self.stream = False
        try:
            response = await self.chat(augmented, **kwargs)
        finally:
            self.stream = prev_stream

        assert isinstance(response, ChatResponse)
        return _finalize_structured(response, structured_model)

    def _validate_tool_choice(
        self,
        tool_choice: ToolChoice,
        tools: list[dict] | None,
    ) -> None:
        if not isinstance(tool_choice, str):
            raise TypeError(
                f"tool_choice must be a string, got {type(tool_choice).__name__}",
            )
        if tool_choice in _TOOL_CHOICE_MODES:
            return
        names = [t.get("function", {}).get("name", "") for t in (tools or [])]
        if tool_choice not in names:
            options = sorted(_TOOL_CHOICE_MODES | set(names))
            raise ValueError(
                f"Invalid tool_choice '{tool_choice}'. "
                f"Available options: {', '.join(options)}",
            )


def _attach_usage_metadata(msg: ChatMessageBase, carrier: Any) -> None:
    usage = getattr(carrier, "usage", None)
    if usage is None:
        return
    md = msg.metadata or {}
    md["usage"] = usage.to_dict()
    msg.metadata = md


def _finalize_structured(
    response: ChatResponse,
    structured_model: Type[T],
) -> ChatResponse:
    if response.parsed is not None and isinstance(response.parsed, structured_model):
        return response

    if response.parsed is not None:
        try:
            response.parsed = TypeAdapter(structured_model).validate_python(
                response.parsed,
            )
        except ValidationError as e:
            raise ModelBehaviorError(
                f"Structured output failed schema validation: {e}",
            ) from e
        return response

    text = response.message.text
    raw = safe_json_loads(text, default=None)
    if raw is None:
        raise ModelBehaviorError(
            "Structured output requested but model produced no parseable JSON.",
        )
    try:
        response.parsed = TypeAdapter(structured_model).validate_python(raw)
    except ValidationError as e:
        raise ModelBehaviorError(
            f"Structured output failed schema validation: {e}",
        ) from e
    return response
