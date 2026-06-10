from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..._logging import logger
from ...message import BinaryContent, ContentItem
from .._response import ToolResponse


class MCPToolFunction:
    def __init__(
        self,
        mcp_name: str,
        tool: Any,
        client: "MCPClientBase",
        wrap_tool_result: bool = True,
    ) -> None:
        self.mcp_name = mcp_name
        self.name = tool.name
        self.description = tool.description or ""
        self.json_schema = _mcp_tool_to_json_schema(tool)
        self._client = client
        self._wrap = wrap_tool_result

    async def __call__(self, **kwargs: Any) -> Any:
        result = await self._client.call_tool(self.name, kwargs)
        if not self._wrap:
            return result
        content = MCPClientBase._convert_mcp_content(
            getattr(result, "content", []) or [],
        )
        return ToolResponse(
            content=content,
            metadata=getattr(result, "meta", None),
            is_error=bool(getattr(result, "isError", False)),
        )


class MCPClientBase(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    async def list_tool_functions(self) -> list[MCPToolFunction]: ...

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict) -> Any: ...

    @staticmethod
    def _convert_mcp_content(content_blocks: list) -> list[ContentItem]:
        items: list[ContentItem] = []
        text_buf = ""
        for c in content_blocks:
            ctype = getattr(c, "type", None)
            if ctype == "text":
                text_buf += getattr(c, "text", "")
            elif ctype == "image":
                if text_buf:
                    items.append(text_buf)
                    text_buf = ""
                items.append(
                    BinaryContent(
                        kind="image",
                        media_type=getattr(c, "mimeType", "image/png"),
                        data=getattr(c, "data", ""),
                    ),
                )
            elif ctype == "audio":
                if text_buf:
                    items.append(text_buf)
                    text_buf = ""
                items.append(
                    BinaryContent(
                        kind="audio",
                        media_type=getattr(c, "mimeType", "audio/wav"),
                        data=getattr(c, "data", ""),
                    ),
                )
            else:
                logger.debug("Skipping unsupported MCP content type: %s", ctype)
        if text_buf:
            items.append(text_buf)
        return items


def _mcp_tool_to_json_schema(tool: Any) -> dict:
    input_schema = getattr(tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": input_schema,
        },
    }
