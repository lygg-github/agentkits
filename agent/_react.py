from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._logging import logger
from ..exceptions import MaxIterationsReached
from ..message import ChatMessageBase, ToolCall, ToolResult
from ..model import ChatModelBase
from ..model._base import OnChunkCb, OnMessageCb
from ..session import SessionBase
from ..tool import Toolkit, ToolResponse
from ..types import ToolChoice
from ..utils._async import maybe_await
from ._base import AgentBase, AgentResult, _accumulate_usage, finalize_structured_output


@dataclass
class ReActResult(AgentResult):
    pass


class ReActAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "agent",
        description: str = "",
        model: ChatModelBase,
        toolkit: Toolkit | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        tool_choice: ToolChoice | None = None,
        stream: bool | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"A ReAct agent named {name}."
        self.model = model
        self.toolkit = toolkit or Toolkit()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.tool_choice = tool_choice
        self.default_stream = stream

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase] | None = None,
        *,
        stream: bool | None = None,
        on_chunk: OnChunkCb | None = None,
        on_message: OnMessageCb | None = None,
        max_iterations: int | None = None,
        session: "SessionBase | None" = None,
        session_id: str | None = None,
        output_type: type | None = None,
    ) -> ReActResult:
        if (session is None) != (session_id is None):
            raise ValueError(
                "`session` and `session_id` must be provided together.",
            )

        limit = max_iterations if max_iterations is not None else self.max_iterations
        history = await self._initial_history(user_input, session, session_id)
        baseline = len(history)
        if user_input is not None and isinstance(user_input, (str, ChatMessageBase)):
            session_cursor = baseline - 1
        else:
            session_cursor = baseline
        tool_calls_made = 0

        from ..message import ChatUsage
        usage_total: ChatUsage | None = None

        def _observe(msg: ChatMessageBase) -> None:
            nonlocal usage_total
            usage_total = _accumulate_usage(usage_total, msg)

        def _compose_on_message(cb: OnMessageCb | None) -> OnMessageCb:
            async def _inner(msg: ChatMessageBase) -> None:
                _observe(msg)
                if cb is not None:
                    await maybe_await(cb, msg)
            return _inner

        wrapped_on_message = _compose_on_message(on_message)

        use_stream = self._resolve_stream(stream)
        prev_stream = self.model.stream
        self.model.stream = use_stream
        try:
            for iteration in range(1, limit + 1):
                tools_schema = self.toolkit.get_json_schemas()

                assistant_msg = await self.model.chat_cb(
                    history,
                    on_chunk=on_chunk if use_stream else None,
                    on_message=wrapped_on_message,
                    tools=tools_schema or None,
                    tool_choice=self.tool_choice,
                )
                history.append(assistant_msg)

                if not assistant_msg.tool_calls:
                    logger.debug(
                        "ReAct[%s] converged after %d iteration(s)",
                        self.name,
                        iteration,
                    )
                    parsed_obj = None
                    if output_type is not None:
                        parsed_obj, extra_usage = await finalize_structured_output(
                            self.model, history, output_type,
                        )
                        if extra_usage is not None:
                            usage_total = (
                                extra_usage
                                if usage_total is None
                                else usage_total + extra_usage
                            )
                    result = ReActResult(
                        messages=history,
                        final_message=assistant_msg,
                        iterations=iteration,
                        tool_calls=tool_calls_made,
                        usage=usage_total,
                        parsed=parsed_obj,
                    )
                    await self._persist(session, session_id, history, session_cursor)
                    return result

                results_message = await self._run_tools(assistant_msg.tool_calls)
                history.append(results_message)
                tool_calls_made += len(assistant_msg.tool_calls)
                await maybe_await(wrapped_on_message, results_message)

                target, filtered_history = await self._detect_handoff(
                    assistant_msg.tool_calls, history,
                )
                if target is not None:
                    await self._persist(session, session_id, history, session_cursor)
                    return await self._delegate_handoff(
                        target,
                        filtered_history,
                        stream=stream,
                        on_chunk=on_chunk,
                        on_message=on_message,
                        max_iterations=max_iterations,
                        session=session,
                        session_id=session_id,
                        iterations_so_far=iteration,
                        tool_calls_so_far=tool_calls_made,
                    )
        finally:
            self.model.stream = prev_stream

        await self._persist(session, session_id, history, session_cursor)
        raise MaxIterationsReached(
            limit,
            last_state=ReActResult(
                messages=history,
                final_message=history[-1] if history else None,
                iterations=limit,
                tool_calls=tool_calls_made,
                usage=usage_total,
            ),
        )

    def _resolve_stream(self, stream: bool | None) -> bool:
        if stream is not None:
            return stream
        if self.default_stream is not None:
            return self.default_stream
        return self.model.stream

    async def _initial_history(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase] | None,
        session: "SessionBase | None",
        session_id: str | None,
    ) -> list[ChatMessageBase]:
        history: list[ChatMessageBase] = []
        if self.system_prompt:
            history.append(ChatMessageBase.system(self.system_prompt))

        if session is not None and session_id is not None:
            prior = await session.load(session_id)
            history.extend(m for m in prior if m.role != "system")

        if user_input is None:
            return history
        if isinstance(user_input, str):
            history.append(ChatMessageBase.user(user_input))
        elif isinstance(user_input, ChatMessageBase):
            history.append(user_input)
        else:
            history.extend(user_input)
        return history

    async def _persist(
        self,
        session: "SessionBase | None",
        session_id: str | None,
        history: list[ChatMessageBase],
        cursor: int,
    ) -> None:
        if session is None or session_id is None:
            return
        new_tail = [m for m in history[cursor:] if m.role != "system"]
        await session.append(session_id, new_tail)

    async def _run_tools(
        self,
        tool_calls: list[ToolCall],
    ) -> ChatMessageBase:
        results: list[ToolResult] = []
        for call in tool_calls:
            collected: list[ToolResponse] = []
            async for chunk in self.toolkit.call_tool(call):
                collected.append(chunk)
            final_resp = _merge_tool_responses(collected)

            results.append(
                ToolResult(
                    id=call.id,
                    name=call.name,
                    content=list(final_resp.content),
                    is_error=final_resp.is_error or final_resp.is_interrupted,
                    metadata=final_resp.metadata,
                ),
            )

        return ChatMessageBase.tool(results)

    async def _detect_handoff(
        self,
        tool_calls: list[ToolCall],
        history: list[ChatMessageBase],
    ) -> tuple[AgentBase | None, list[ChatMessageBase]]:
        from ._handoff import get_handoff_marker

        for call in tool_calls:
            entry = self.toolkit.tools.get(call.name)
            if entry is None:
                continue
            marker = get_handoff_marker(entry.original_func)
            if marker is None:
                continue

            filtered = history
            if marker.input_filter is not None:
                out = marker.input_filter(history)
                if hasattr(out, "__await__"):
                    out = await out
                filtered = list(out)
            return marker.target, filtered

        return None, history

    async def _delegate_handoff(
        self,
        target: AgentBase,
        history: list[ChatMessageBase],
        *,
        stream: bool | None,
        on_chunk: OnChunkCb | None,
        on_message: OnMessageCb | None,
        max_iterations: int | None,
        session: "SessionBase | None",
        session_id: str | None,
        iterations_so_far: int,
        tool_calls_so_far: int,
    ) -> ReActResult:
        sub_kwargs: dict[str, object] = {
            "stream": stream,
            "on_chunk": on_chunk,
            "on_message": on_message,
            "max_iterations": max_iterations,
        }
        if session is not None:
            sub_kwargs["session"] = session
            sub_kwargs["session_id"] = session_id

        nested = await target.run(history, **sub_kwargs)

        total_usage = nested.usage
        return ReActResult(
            messages=nested.messages,
            final_message=nested.final_message,
            iterations=iterations_so_far + nested.iterations,
            tool_calls=tool_calls_so_far + nested.tool_calls,
            usage=total_usage,
            metadata={"handoff_to": target.name},
        )


def _merge_tool_responses(chunks: list[ToolResponse]) -> ToolResponse:
    if not chunks:
        return ToolResponse()
    if len(chunks) == 1:
        return chunks[0]

    merged: list = []
    for c in chunks:
        for item in c.content:
            if (
                isinstance(item, str)
                and merged
                and isinstance(merged[-1], str)
            ):
                merged[-1] = merged[-1] + item
            else:
                merged.append(item)

    return ToolResponse(
        content=merged,
        metadata=chunks[-1].metadata,
        is_interrupted=any(c.is_interrupted for c in chunks),
        is_error=any(c.is_error for c in chunks),
    )
