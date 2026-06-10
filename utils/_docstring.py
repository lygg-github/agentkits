from __future__ import annotations

from typing import Tuple

from docstring_parser import parse as _parse


def parse_docstring(
    doc: str | None,
    include_long: bool = True,
) -> Tuple[str, dict[str, str]]:
    if not doc:
        return "", {}

    parsed = _parse(doc)

    short = (parsed.short_description or "").strip()
    long = (parsed.long_description or "").strip() if include_long else ""
    if short and long:
        description = f"{short}\n\n{long}"
    elif short:
        description = short
    else:
        description = long

    param_docs: dict[str, str] = {}
    for p in parsed.params:
        if p.arg_name and p.description:
            param_docs[p.arg_name] = p.description.strip()

    return description, param_docs
