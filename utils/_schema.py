from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from pydantic import TypeAdapter

from ._docstring import parse_docstring


def parse_tool_function(
    func: Callable[..., Any],
    include_long_description: bool = True,
) -> dict:
    if not callable(func):
        raise TypeError(f"Expected a callable, got {type(func).__name__}")

    name = getattr(func, "__name__", "tool")
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    description, param_docs = parse_docstring(
        func.__doc__, include_long=include_long_description,
    )

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, param.annotation)
        schema: dict[str, Any]
        if annotation is inspect.Parameter.empty:
            schema = {}
        else:
            try:
                schema = TypeAdapter(annotation).json_schema()
            except Exception:
                schema = {}

        schema.pop("title", None)

        doc = param_docs.get(param_name)
        if doc:
            schema["description"] = doc

        properties[param_name] = schema

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    params_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        params_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": params_schema,
        },
    }
