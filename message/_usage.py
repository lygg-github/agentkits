from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChatUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    runtime: float = 0.0
    type: Literal["chat"] = field(default="chat")
    metadata: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "ChatUsage | None") -> "ChatUsage":
        if other is None:
            return ChatUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                runtime=self.runtime,
            )
        return ChatUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            runtime=self.runtime + other.runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "runtime": self.runtime,
        }
