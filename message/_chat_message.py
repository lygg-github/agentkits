from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from ..types import Role
from ..utils._id import make_msg_id
from ..utils._json import safe_json_loads
from ._content import (
    BinaryContent,
    ContentItem,
    ContentLike,
    ToolCall,
    ToolResult,
    coerce_content,
    content_binary,
    content_text,
)


@dataclass
class ChatMessageBase:
    role: Role
    content: List[ContentItem] = field(default_factory=list)
    reasoning_content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    name: str | None = None
    id: str = field(default_factory=make_msg_id)
    created: int = field(default_factory=lambda: int(time.time()))
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, list) or any(
            not isinstance(i, (str, BinaryContent)) for i in self.content
        ):
            self.content = coerce_content(self.content)

    @classmethod
    def system(cls, content: ContentLike, **kw: Any) -> "ChatMessageBase":
        return cls(role="system", content=coerce_content(content), **kw)

    @classmethod
    def user(cls, content: ContentLike = "", **kw: Any) -> "ChatMessageBase":
        return cls(role="user", content=coerce_content(content), **kw)

    @classmethod
    def assistant(
        cls,
        content: ContentLike = "",
        *,
        reasoning_content: str = "",
        tool_calls: Iterable[ToolCall] | None = None,
        **kw: Any,
    ) -> "ChatMessageBase":
        return cls(
            role="assistant",
            content=coerce_content(content),
            reasoning_content=reasoning_content,
            tool_calls=list(tool_calls or []),
            **kw,
        )

    @classmethod
    def tool(
        cls,
        results: Iterable[ToolResult],
        **kw: Any,
    ) -> "ChatMessageBase":
        return cls(role="tool", tool_results=list(results), **kw)

    @property
    def text(self) -> str:
        return content_text(self.content)

    @property
    def binary(self) -> list[BinaryContent]:
        return content_binary(self.content)

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def has_binary(self) -> bool:
        return any(isinstance(i, BinaryContent) for i in self.content)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": [
                i if isinstance(i, str) else i.__dict__ for i in self.content
            ],
            "reasoning_content": self.reasoning_content,
            "tool_calls": [tc.__dict__ for tc in self.tool_calls],
            "tool_results": [
                {
                    **{k: v for k, v in tr.__dict__.items() if k != "content"},
                    "content": [
                        i if isinstance(i, str) else i.__dict__
                        for i in (tr.content or [])
                    ],
                }
                for tr in self.tool_results
            ],
            "name": self.name,
            "created": self.created,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessageBase":
        content: list[ContentItem] = []
        for item in data.get("content", []) or []:
            if isinstance(item, str):
                content.append(item)
            elif isinstance(item, dict):
                content.append(BinaryContent(**item))
        tool_calls = [ToolCall(**tc) for tc in data.get("tool_calls", []) or []]
        tool_results: list[ToolResult] = []
        for tr in data.get("tool_results", []) or []:
            tr_content: list[ContentItem] = []
            for item in tr.get("content", []) or []:
                if isinstance(item, str):
                    tr_content.append(item)
                elif isinstance(item, dict):
                    tr_content.append(BinaryContent(**item))
            tool_results.append(
                ToolResult(
                    id=tr.get("id", ""),
                    name=tr.get("name", ""),
                    content=tr_content,
                    is_error=bool(tr.get("is_error", False)),
                    metadata=tr.get("metadata"),
                ),
            )

        msg = cls(
            role=data.get("role", "user"),
            content=content,
            reasoning_content=data.get("reasoning_content", "") or "",
            tool_calls=tool_calls,
            tool_results=tool_results,
            name=data.get("name"),
            metadata=data.get("metadata"),
        )
        if data.get("id"):
            msg.id = data["id"]
        if data.get("created"):
            msg.created = int(data["created"])
        return msg

    def to_openai(self) -> List[dict]:
        if self.role == "tool":
            return [_tool_result_to_openai(tr) for tr in self.tool_results]

        msg: dict[str, Any] = {"role": self.role}
        if self.name:
            msg["name"] = self.name

        msg["content"] = _content_to_openai(
            self.content, has_tool_calls=bool(self.tool_calls),
        )

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_input or _dumps(tc.input or {}),
                    },
                }
                for tc in self.tool_calls
            ]

        if self.reasoning_content and self.role == "assistant":
            msg["reasoning_content"] = self.reasoning_content

        return [msg]

    @classmethod
    def from_openai(cls, raw: dict) -> "ChatMessageBase":
        role = raw.get("role", "assistant")

        if role == "tool":
            tr = ToolResult(
                id=raw.get("tool_call_id", ""),
                name=raw.get("name", ""),
                content=[str(raw.get("content", "") or "")],
            )
            return cls(role="tool", tool_results=[tr])

        content: list[ContentItem] = []
        raw_content = raw.get("content")
        if isinstance(raw_content, str):
            if raw_content:
                content.append(raw_content)
        elif isinstance(raw_content, list):
            text_buf = ""
            for part in raw_content:
                ptype = part.get("type")
                if ptype == "text":
                    text_buf += part.get("text", "")
                elif ptype == "image_url":
                    if text_buf:
                        content.append(text_buf)
                        text_buf = ""
                    url = part.get("image_url", {}).get("url", "")
                    content.append(_openai_image_to_binary(url))
                elif ptype == "input_audio":
                    if text_buf:
                        content.append(text_buf)
                        text_buf = ""
                    ia = part.get("input_audio", {})
                    content.append(
                        BinaryContent(
                            kind="audio",
                            media_type=f"audio/{ia.get('format', 'wav')}",
                            data=ia.get("data", ""),
                        ),
                    )
            if text_buf:
                content.append(text_buf)

        reasoning = raw.get("reasoning_content") or raw.get("reasoning") or ""

        tool_calls: list[ToolCall] = []
        for tc in raw.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            args_str = fn.get("arguments", "") or ""
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=safe_json_loads(args_str, default={}),
                    raw_input=args_str,
                ),
            )

        return cls(
            role=role,
            content=content,
            reasoning_content=reasoning if isinstance(reasoning, str) else "",
            tool_calls=tool_calls,
            name=raw.get("name"),
        )

    def to_claude(self) -> List[dict]:
        if self.role == "system":
            return [{"role": "system", "content": self.text}]

        if self.role == "tool":
            blocks = [_tool_result_to_claude(tr) for tr in self.tool_results]
            return [{"role": "user", "content": blocks}]

        blocks: list[dict] = []

        if self.reasoning_content:
            blocks.append(
                {"type": "thinking", "thinking": self.reasoning_content},
            )

        for item in self.content:
            if isinstance(item, str):
                if item:
                    blocks.append({"type": "text", "text": item})
            else:
                claude_block = _binary_to_claude_block(item)
                if claude_block:
                    blocks.append(claude_block)

        for tc in self.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input or {},
                },
            )

        return [{"role": self.role, "content": blocks or ""}]

    @classmethod
    def from_claude(cls, raw: Any) -> "ChatMessageBase":
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()

        role = raw.get("role", "assistant")
        content: list[ContentItem] = []
        reasoning = ""
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolResult] = []

        for block in raw.get("content") or []:
            is_dict = isinstance(block, dict)
            btype = block.get("type") if is_dict else getattr(block, "type", None)
            get = (
                block.get
                if is_dict
                else (lambda k, d=None, _b=block: getattr(_b, k, d))
            )

            if btype == "text":
                t = get("text", "")
                if t:
                    content.append(t)
            elif btype == "thinking":
                reasoning += get("thinking", "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=get("id", ""),
                        name=get("name", ""),
                        input=dict(get("input", {}) or {}),
                    ),
                )
            elif btype == "tool_result":
                out = get("content", "")
                if isinstance(out, list):
                    flat = "".join(
                        (x.get("text", "") if isinstance(x, dict) else "")
                        for x in out
                        if (x.get("type") if isinstance(x, dict) else None) == "text"
                    )
                else:
                    flat = str(out or "")
                tool_results.append(
                    ToolResult(
                        id=get("tool_use_id", ""),
                        name="",
                        content=[flat] if flat else [],
                        is_error=bool(get("is_error", False)),
                    ),
                )
            elif btype == "image":
                src = get("source", {}) or {}
                content.append(_claude_source_to_binary("image", src))

        msg_id = raw.get("id") if isinstance(raw, dict) else getattr(raw, "id", None)
        kwargs: dict[str, Any] = {
            "role": role,
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }
        if msg_id:
            kwargs["id"] = msg_id
        return cls(**kwargs)


def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _content_to_openai(
    content: list[ContentItem],
    *,
    has_tool_calls: bool,
) -> Any:
    if not content:
        return None if has_tool_calls else ""

    has_bin = any(isinstance(i, BinaryContent) for i in content)
    if not has_bin:
        return "".join(i for i in content if isinstance(i, str))

    parts: list[dict] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append({"type": "text", "text": item})
        else:
            parts.append(_binary_to_openai_part(item))
    return parts


def _binary_to_openai_part(b: BinaryContent) -> dict:
    if b.kind == "image":
        if b.url is not None:
            return {"type": "image_url", "image_url": {"url": b.url}}
        if b.data is not None:
            data_url = f"data:{b.media_type};base64,{b.data}"
            return {"type": "image_url", "image_url": {"url": data_url}}
        return {"type": "text", "text": f"[image file:{b.file_id}]"}

    if b.kind == "audio":
        fmt = b.media_type.split("/")[-1] or "wav"
        if b.data is not None:
            return {
                "type": "input_audio",
                "input_audio": {"data": b.data, "format": fmt},
            }
        return {"type": "text", "text": f"[audio: {b.url or b.file_id}]"}

    ref = b.url or b.file_id or "<binary>"
    return {"type": "text", "text": f"[{b.kind}: {ref}]"}


def _openai_image_to_binary(url: str) -> BinaryContent:
    if url.startswith("data:"):
        try:
            header, payload = url.split(",", 1)
            media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
        except ValueError:
            media_type, payload = "image/png", ""
        return BinaryContent(kind="image", media_type=media_type, data=payload)
    return BinaryContent(kind="image", media_type="image/png", url=url)


def _binary_to_claude_block(b: BinaryContent) -> dict | None:
    if b.kind == "image":
        if b.data is not None:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": b.media_type,
                    "data": b.data,
                },
            }
        if b.url is not None:
            return {"type": "image", "source": {"type": "url", "url": b.url}}
    return None


def _claude_source_to_binary(kind: str, src: dict) -> BinaryContent:
    if src.get("type") == "base64":
        return BinaryContent(
            kind=kind,
            media_type=src.get("media_type", "image/png"),
            data=src.get("data", ""),
        )
    return BinaryContent(
        kind=kind,
        media_type="image/png",
        url=src.get("url", ""),
    )


def _tool_result_to_openai(tr: ToolResult) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tr.id,
        "content": tr.text,
    }


def _tool_result_to_claude(tr: ToolResult) -> dict:
    out: dict[str, Any] = {"type": "tool_result", "tool_use_id": tr.id}
    binary = tr.binary
    if binary:
        inner: list[dict] = []
        if tr.text:
            inner.append({"type": "text", "text": tr.text})
        for b in binary:
            claude_block = _binary_to_claude_block(b)
            if claude_block:
                inner.append(claude_block)
        out["content"] = inner
    else:
        out["content"] = tr.text
    if tr.is_error:
        out["is_error"] = True
    return out
