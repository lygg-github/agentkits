from __future__ import annotations

from datetime import timedelta
from typing import Any

from ._base import MCPClientBase, MCPToolFunction


class StdIOStatefulClient(MCPClientBase):
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(name)
        self.command = command
        self.args = args or []
        self.env = env
        self.timeout = timedelta(seconds=timeout) if timeout else None
        self._session = None
        self._cm = None
        self._client_cm = None

    async def __aenter__(self) -> "StdIOStatefulClient":
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise ImportError(
                "MCP support requires the `mcp` package. "
                "Install with `pip install agentkits[mcp]`.",
            ) from e

        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        self._cm = stdio_client(params)
        read, write = await self._cm.__aenter__()
        self._client_cm = ClientSession(read, write)
        self._session = await self._client_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._client_cm is not None:
                await self._client_cm.__aexit__(exc_type, exc, tb)
        finally:
            self._session = None
            self._client_cm = None
            if self._cm is not None:
                await self._cm.__aexit__(exc_type, exc, tb)
                self._cm = None

    async def list_tool_functions(self) -> list[MCPToolFunction]:
        if self._session is None:
            raise RuntimeError(
                "StdIOStatefulClient must be used as an async context manager.",
            )
        result = await self._session.list_tools()
        return [
            MCPToolFunction(mcp_name=self.name, tool=t, client=self)
            for t in result.tools
        ]

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        if self._session is None:
            raise RuntimeError(
                "StdIOStatefulClient must be used as an async context manager.",
            )
        return await self._session.call_tool(
            tool_name,
            arguments=arguments,
            read_timeout_seconds=self.timeout,
        )
