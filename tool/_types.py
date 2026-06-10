from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, TypedDict

from ..types import ToolFunction


@dataclass
class RegisteredToolFunction:
    name: str
    group: str
    source: Literal["function", "mcp_server", "skill"]
    original_func: ToolFunction
    json_schema: dict
    preset_kwargs: dict[str, Any] = field(default_factory=dict)
    original_name: str | None = None
    mcp_name: str | None = None
    postprocess_func: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None


@dataclass
class ToolGroup:
    name: str
    description: str
    active: bool = True
    notes: str | None = None


class AgentSkill(TypedDict, total=False):
    name: str
    description: str
    dir: str
