from __future__ import annotations

import shortuuid


def make_id(prefix: str, size: int = 16) -> str:
    return f"{prefix}_{shortuuid.uuid()[:size]}"


def make_msg_id() -> str:
    return make_id("msg")


def make_resp_id() -> str:
    return make_id("resp")


def make_tool_id() -> str:
    return make_id("tool")
