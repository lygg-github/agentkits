from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Union

BinaryKind = Literal["image", "audio", "video", "file"]


@dataclass
class BinaryContent:
    kind: BinaryKind
    media_type: str
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        present = sum(v is not None for v in (self.data, self.url, self.file_id))
        if present != 1:
            raise ValueError(
                "BinaryContent requires exactly one of data / url / file_id",
            )

    @classmethod
    def image(
        cls,
        *,
        url: str | None = None,
        data: str | None = None,
        media_type: str = "image/png",
        file_id: str | None = None,
    ) -> "BinaryContent":
        return cls(
            kind="image",
            media_type=media_type,
            url=url,
            data=data,
            file_id=file_id,
        )

    @classmethod
    def audio(
        cls,
        *,
        url: str | None = None,
        data: str | None = None,
        media_type: str = "audio/wav",
        file_id: str | None = None,
    ) -> "BinaryContent":
        return cls(
            kind="audio",
            media_type=media_type,
            url=url,
            data=data,
            file_id=file_id,
        )

    @classmethod
    def file(
        cls,
        *,
        file_id: str | None = None,
        url: str | None = None,
        data: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> "BinaryContent":
        return cls(
            kind="file",
            media_type=media_type,
            url=url,
            data=data,
            file_id=file_id,
        )


ContentItem = Union[str, BinaryContent]
ContentLike = Union[str, BinaryContent, Iterable[Any], None]


def coerce_content(value: ContentLike) -> list[ContentItem]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, BinaryContent):
        return [value]
    if isinstance(value, Iterable):
        out: list[ContentItem] = []
        for item in value:
            if isinstance(item, str):
                if item:
                    out.append(item)
            elif isinstance(item, BinaryContent):
                out.append(item)
            else:
                raise TypeError(
                    f"content item must be str or BinaryContent, "
                    f"got {type(item).__name__}",
                )
        return out
    raise TypeError(f"cannot coerce {type(value).__name__} to content list")


def content_text(items: list[ContentItem]) -> str:
    return "".join(i for i in items if isinstance(i, str))


def content_binary(items: list[ContentItem]) -> list[BinaryContent]:
    return [i for i in items if isinstance(i, BinaryContent)]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any] | None = None
    raw_input: str = ""

    def __post_init__(self) -> None:
        if self.input is None:
            self.input = {}


@dataclass
class ToolResult:
    id: str
    name: str
    content: list[ContentItem] | None = None
    is_error: bool = False
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.content is None:
            self.content = []
        else:
            self.content = coerce_content(self.content)

    @property
    def text(self) -> str:
        return content_text(self.content or [])

    @property
    def binary(self) -> list[BinaryContent]:
        return content_binary(self.content or [])
