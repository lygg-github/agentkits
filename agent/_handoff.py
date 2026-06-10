from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, TypeAlias

from ..message import ChatMessageBase
from ..tool import ToolResponse
from ._base import AgentBase


HandoffInputFilter: TypeAlias = Callable[
    [list[ChatMessageBase]],
    list[ChatMessageBase] | Awaitable[list[ChatMessageBase]],
]


@dataclass
class _HandoffMarker:
    target: AgentBase
    input_filter: HandoffInputFilter | None = None


def handoff(
    agent: AgentBase,
    *,
    name: str | None = None,
    description: str | None = None,
    input_filter: HandoffInputFilter | None = None,
) -> Callable[..., "Awaitable[ToolResponse]"]:
    tool_name = name or f"transfer_to_{agent.name}"
    tool_description = description or (
        f"Hand the conversation off to the '{agent.name}' agent. "
        f"{agent.description or ''}"
    ).strip()

    async def _transfer(reason: str = "") -> ToolResponse:
        ack = (
            f"Handing off to {agent.name}."
            if not reason
            else f"Handing off to {agent.name}: {reason}"
        )
        return ToolResponse.from_value(ack)

    _transfer.__name__ = tool_name
    _transfer.__doc__ = tool_description + (
        "\n\nArgs:\n    reason: Short explanation of why the handoff is needed."
    )
    _transfer._handoff = _HandoffMarker(
        target=agent, input_filter=input_filter,
    )
    return _transfer


def get_handoff_marker(fn: object) -> _HandoffMarker | None:
    return getattr(fn, "_handoff", None)
