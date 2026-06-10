from dataclasses import dataclass, field
from typing import Any, List

from ..message import (
    BinaryContent,
    ContentItem,
    ContentLike,
    coerce_content,
    content_binary,
    content_text,
)
from ..utils._id import make_tool_id


@dataclass
class ToolResponse:
    content: List[ContentItem] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    stream: bool = False
    is_last: bool = True
    is_interrupted: bool = False
    is_error: bool = False
    id: str = field(default_factory=make_tool_id)

    def __post_init__(self) -> None:
        if not isinstance(self.content, list) or any(
            not isinstance(i, (str, BinaryContent)) for i in self.content
        ):
            self.content = coerce_content(self.content)

    @property
    def text(self) -> str:
        return content_text(self.content)

    @property
    def binary(self) -> list[BinaryContent]:
        return content_binary(self.content)

    @classmethod
    def from_value(cls, value: ContentLike, **kw: Any) -> "ToolResponse":
        return cls(content=coerce_content(value), **kw)

    @classmethod
    def error(cls, message: str, **kw: Any) -> "ToolResponse":
        kw.setdefault("is_error", True)
        return cls(content=[message] if message else [], **kw)
