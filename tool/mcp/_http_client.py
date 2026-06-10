from __future__ import annotations

from datetime import timedelta
from typing import Any

from ._base import MCPClientBase, MCPToolFunction


class HttpStatelessClient(MCPClientBase):
    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(name)
        self.url = url
        self.headers = headers or {}
        self.timeout = timedelta(seconds=timeout) if timeout else None

    async def list_tool_functions(self) -> list[MCPToolFunction]:
        async with self._open() as session:
            result = await session.list_tools()
            return [
                MCPToolFunction(mcp_name=self.name, tool=t, client=self)
                for t in result.tools
            ]

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        async with self._open() as session:
            return await session.call_tool(
                tool_name,
                arguments=arguments,
                read_timeout_seconds=self.timeout,
            )

    def _open(self):
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            raise ImportError(
                "MCP support requires the `mcp` package. "
                "Install with `pip install agentkits[mcp]`.",
            ) from e

        class _Ctx:
            def __init__(self_inner):
                self_inner._transport_cm = streamablehttp_client(
                    self.url, headers=self.headers,
                )
                self_inner._session_cm = None
                self_inner._session = None

            async def __aenter__(self_inner):
                read, write, _ = await self_inner._transport_cm.__aenter__()
                self_inner._session_cm = ClientSession(read, write)
                self_inner._session = await self_inner._session_cm.__aenter__()
                await self_inner._session.initialize()
                return self_inner._session

            async def __aexit__(self_inner, exc_type, exc, tb):
                try:
                    if self_inner._session_cm is not None:
                        await self_inner._session_cm.__aexit__(exc_type, exc, tb)
                finally:
                    await self_inner._transport_cm.__aexit__(exc_type, exc, tb)

        return _Ctx()
