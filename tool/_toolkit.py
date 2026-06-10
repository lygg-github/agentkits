from __future__ import annotations

import inspect
from copy import deepcopy
from functools import partial
from typing import Any, AsyncGenerator, Callable, Generator, Literal

import shortuuid

from .._logging import logger
from ..exceptions import ToolExecutionError, ToolNotFoundError
from ..message import ToolCall
from ..types import ToolFunction
from ..utils._schema import parse_tool_function
from ._response import ToolResponse
from ._types import AgentSkill, RegisteredToolFunction, ToolGroup


class Toolkit:
    def __init__(self) -> None:
        self.tools: dict[str, RegisteredToolFunction] = {}
        self.groups: dict[str, ToolGroup] = {
            "basic": ToolGroup(name="basic", description="Always-on tools.", active=True),
        }
        self.skills: dict[str, AgentSkill] = {}

    def create_tool_group(
        self,
        name: str,
        description: str,
        active: bool = True,
        notes: str | None = None,
    ) -> None:
        if name in self.groups:
            raise ValueError(f"Tool group '{name}' already exists.")
        self.groups[name] = ToolGroup(
            name=name, description=description, active=active, notes=notes,
        )

    def set_group_active(self, name: str, active: bool) -> None:
        if name == "basic":
            logger.warning("The 'basic' group is always active.")
            return
        if name in self.groups:
            self.groups[name].active = active

    def remove_tool_group(self, name: str) -> None:
        if name == "basic":
            raise ValueError("Cannot remove the 'basic' group.")
        self.groups.pop(name, None)
        for tool_name in list(self.tools):
            if self.tools[tool_name].group == name:
                self.tools.pop(tool_name)

    def register_tool_function(
        self,
        tool_func: ToolFunction,
        group_name: str = "basic",
        preset_kwargs: dict[str, Any] | None = None,
        func_name: str | None = None,
        func_description: str | None = None,
        json_schema: dict | None = None,
        namesake_strategy: Literal["raise", "override", "skip", "rename"] = "raise",
        postprocess_func: Callable[..., Any] | None = None,
    ) -> str:
        if group_name not in self.groups:
            raise ValueError(f"Tool group '{group_name}' not found.")

        from .mcp import MCPToolFunction

        mcp_name: str | None = None

        if isinstance(tool_func, MCPToolFunction):
            input_name = tool_func.name
            original_func = tool_func.__call__
            json_schema = json_schema or tool_func.json_schema
            mcp_name = tool_func.mcp_name

        elif isinstance(tool_func, partial):
            merged_preset: dict[str, Any] = dict(tool_func.keywords or {})
            if tool_func.args:
                params = list(inspect.signature(tool_func.func).parameters.keys())
                for i, arg in enumerate(tool_func.args):
                    if i < len(params):
                        merged_preset[params[i]] = arg
            merged_preset.update(preset_kwargs or {})
            preset_kwargs = merged_preset

            input_name = tool_func.func.__name__
            original_func = tool_func.func
            json_schema = json_schema or parse_tool_function(tool_func.func)

        else:
            input_name = getattr(tool_func, "__name__", "tool")
            original_func = tool_func
            json_schema = json_schema or parse_tool_function(tool_func)

        name = func_name or input_name
        json_schema = deepcopy(json_schema)
        json_schema["function"]["name"] = name
        if func_description:
            json_schema["function"]["description"] = func_description

        props = json_schema["function"]["parameters"].get("properties", {})
        required = json_schema["function"]["parameters"].get("required", [])
        for key in preset_kwargs or {}:
            props.pop(key, None)
            if key in required:
                required.remove(key)
        if not required:
            json_schema["function"]["parameters"].pop("required", None)

        entry = RegisteredToolFunction(
            name=name,
            group=group_name,
            source="mcp_server" if mcp_name else "function",
            original_func=original_func,
            json_schema=json_schema,
            preset_kwargs=preset_kwargs or {},
            original_name=input_name if func_name else None,
            mcp_name=mcp_name,
            postprocess_func=postprocess_func,
        )

        return self._insert(entry, namesake_strategy)

    def _insert(
        self,
        entry: RegisteredToolFunction,
        strategy: Literal["raise", "override", "skip", "rename"],
    ) -> str:
        name = entry.name
        if name not in self.tools:
            self.tools[name] = entry
            return name

        if strategy == "raise":
            raise ValueError(f"Tool function '{name}' is already registered.")
        if strategy == "skip":
            logger.warning("Tool function '%s' already registered, skipping.", name)
            return name
        if strategy == "override":
            self.tools[name] = entry
            return name
        if strategy == "rename":
            for _ in range(100):
                candidate = f"{name}_{shortuuid.uuid()[:5]}"
                if candidate not in self.tools:
                    entry.original_name = entry.original_name or name
                    entry.name = candidate
                    entry.json_schema["function"]["name"] = candidate
                    self.tools[candidate] = entry
                    return candidate
            raise RuntimeError(f"Failed to produce a unique name for '{name}'.")
        raise ValueError(f"Invalid namesake_strategy: {strategy}")

    def remove_tool_function(self, name: str, allow_not_exist: bool = True) -> None:
        if name not in self.tools and not allow_not_exist:
            raise ValueError(f"Tool function '{name}' not found.")
        self.tools.pop(name, None)

    async def register_mcp_client(
        self,
        client: Any,
        group_name: str = "basic",
        namesake_strategy: Literal["raise", "override", "skip", "rename"] = "raise",
    ) -> list[str]:
        from .mcp import MCPClientBase

        if not isinstance(client, MCPClientBase):
            raise TypeError(
                f"Expected MCPClientBase, got {type(client).__name__}",
            )
        tool_functions = await client.list_tool_functions()
        registered: list[str] = []
        for tool_fn in tool_functions:
            registered.append(
                self.register_tool_function(
                    tool_fn,
                    group_name=group_name,
                    namesake_strategy=namesake_strategy,
                ),
            )
        return registered

    async def remove_mcp_client(self, mcp_name: str) -> list[str]:
        removed: list[str] = []
        for name in list(self.tools):
            if self.tools[name].mcp_name == mcp_name:
                self.tools.pop(name)
                removed.append(name)
        return removed

    def register_skill(self, skill: AgentSkill) -> None:
        self.skills[skill["name"]] = skill

    def tool(
        self,
        group_name: str = "basic",
        *,
        name: str | None = None,
        description: str | None = None,
        preset_kwargs: dict[str, Any] | None = None,
        namesake_strategy: Literal["raise", "override", "skip", "rename"] = "raise",
    ) -> Callable[[ToolFunction], ToolFunction]:
        def decorator(fn: ToolFunction) -> ToolFunction:
            self.register_tool_function(
                fn,
                group_name=group_name,
                func_name=name,
                func_description=description,
                preset_kwargs=preset_kwargs,
                namesake_strategy=namesake_strategy,
            )
            return fn

        return decorator

    def get_json_schemas(self) -> list[dict]:
        return [
            deepcopy(entry.json_schema)
            for entry in self.tools.values()
            if entry.group == "basic" or self.groups[entry.group].active
        ]

    async def call_tool(
        self,
        tool_call: ToolCall,
        *,
        raise_on_missing: bool = False,
        raise_on_error: bool = False,
    ) -> AsyncGenerator[ToolResponse, None]:
        name = tool_call.name
        entry = self.tools.get(name)
        if entry is None:
            if raise_on_missing:
                raise ToolNotFoundError(name)
            yield ToolResponse.error(f"Tool '{name}' not found.")
            return

        kwargs = dict(tool_call.input or {})
        kwargs.update(entry.preset_kwargs)

        try:
            if inspect.iscoroutinefunction(entry.original_func):
                res = await entry.original_func(**kwargs)
            else:
                res = entry.original_func(**kwargs)
        except Exception as e:
            logger.exception("Tool '%s' raised", name)
            if raise_on_error:
                raise ToolExecutionError(name, e) from e
            yield ToolResponse.error(f"Error executing tool '{name}': {e}")
            return

        async for chunk in _iter_tool_result(res):
            if entry.postprocess_func is not None:
                cb = entry.postprocess_func(tool_call, chunk)
                if inspect.isawaitable(cb):
                    cb = await cb
                if isinstance(cb, ToolResponse):
                    chunk = cb
            yield chunk


async def _iter_tool_result(res: Any) -> AsyncGenerator[ToolResponse, None]:
    if isinstance(res, ToolResponse):
        yield res
        return

    if isinstance(res, AsyncGenerator):
        async for chunk in res:
            yield _coerce_chunk(chunk)
        return

    if isinstance(res, Generator):
        for chunk in res:
            yield _coerce_chunk(chunk)
        return

    yield ToolResponse.from_value("" if res is None else str(res))


def _coerce_chunk(value: Any) -> ToolResponse:
    if isinstance(value, ToolResponse):
        return value
    return ToolResponse.from_value(str(value))
