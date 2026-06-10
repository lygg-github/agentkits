from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Type, TypeVar

from pydantic import BaseModel

from ..message import ChatMessageBase, ChatUsage
from ..model import ChatModelBase


T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentResult:
    messages: List[ChatMessageBase] = field(default_factory=list)
    final_message: ChatMessageBase | None = None
    iterations: int = 0
    tool_calls: int = 0
    usage: ChatUsage | None = None
    parsed: Any = None
    metadata: dict[str, Any] | None = None

    def text(self) -> str:
        return self.final_message.text if self.final_message else ""


class AgentBase(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        **kwargs: Any,
    ) -> AgentResult: ...


def _accumulate_usage(
    total: ChatUsage | None,
    msg: ChatMessageBase,
) -> ChatUsage | None:
    usage_dict = (msg.metadata or {}).get("usage") if msg.metadata else None
    if not isinstance(usage_dict, dict):
        return total
    per = ChatUsage(
        input_tokens=int(usage_dict.get("input_tokens", 0)),
        output_tokens=int(usage_dict.get("output_tokens", 0)),
        runtime=float(usage_dict.get("runtime", 0.0)),
    )
    return per if total is None else total + per


async def finalize_structured_output(
    model: ChatModelBase,
    history: list[ChatMessageBase],
    output_type: Type[T],
) -> tuple[T, ChatUsage | None]:
    tail = list(history) + [
        ChatMessageBase.user(
            f"Now produce the final answer strictly as a structured "
            f"{output_type.__name__} object.",
        ),
    ]
    from ..model._response import ChatResponse

    prev_stream = model.stream
    model.stream = False
    try:
        response = await model.chat(tail, structured_model=output_type)
    finally:
        model.stream = prev_stream
    assert isinstance(response, ChatResponse)
    return response.parsed, response.usage
