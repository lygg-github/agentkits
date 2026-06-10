from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PatternSchema:
    pattern: re.Pattern
    fields: tuple[str, ...] | None = None
    _defaults: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        pattern: str,
        *,
        verbose: bool = False,
        dotall: bool = False,
        ignorecase: bool = False,
        multiline: bool = False,
        fields: Iterable[str] | None = None,
        defaults: dict[str, str] | None = None,
    ) -> "PatternSchema":
        flags = 0
        if verbose:
            flags |= re.VERBOSE
        if dotall:
            flags |= re.DOTALL
        if ignorecase:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE
        compiled = re.compile(pattern, flags)
        return cls(
            pattern=compiled,
            fields=tuple(fields) if fields else None,
            _defaults=dict(defaults or {}),
        )

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self.pattern.groupindex.keys())

    def match_one(self, text: str) -> dict[str, str] | None:
        m = self.pattern.search(text or "")
        if m is None:
            return None
        return self._to_dict(m)

    def match_all(self, text: str) -> list[dict[str, str]]:
        if not text:
            return []
        return [self._to_dict(m) for m in self.pattern.finditer(text)]

    def _to_dict(self, m: re.Match) -> dict[str, str]:
        gd = m.groupdict()
        raw: dict[str, str] = {}
        keys = self.fields or tuple(self.pattern.groupindex.keys())
        for k in keys:
            v = gd.get(k)
            if v is None:
                v = self._defaults.get(k, "")
            raw[k] = v.strip() if isinstance(v, str) else v
        return raw
