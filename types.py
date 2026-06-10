from typing import Any, Awaitable, Callable, Literal, Union

JSONValue = Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]]
Role = Literal["system", "user", "assistant", "tool"]
ToolChoice = Literal["auto", "none", "required"] | str
MaybeAwaitable = Union[Any, Awaitable[Any]]
ToolFunction = Callable[..., Any]
