from __future__ import annotations

import json
from typing import Any

from json_repair import repair_json


def safe_json_loads(text: str, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            return default


def try_model_dump(obj: Any) -> dict | None:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return None
    return None


def repair_partial_json(
    text: str,
    previous: dict | None = None,
) -> dict:
    if not text:
        return previous if previous is not None else {}
    try:
        repaired = repair_json(text, return_objects=True)
    except Exception:
        return previous if previous is not None else {}
    if isinstance(repaired, dict):
        return repaired
    return previous if previous is not None else {}
