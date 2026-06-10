from __future__ import annotations

import os
import re
from typing import List

from .._types import AgentSkill


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def load_skill(skill_dir: str) -> AgentSkill:
    skill_dir = os.path.abspath(skill_dir)
    manifest_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Expected SKILL.md inside {skill_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        text = f.read()

    meta: dict[str, str] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
        body = text[m.end():]

    name = meta.get("name") or os.path.basename(skill_dir.rstrip(os.sep))
    description = meta.get("description") or _first_non_empty_line(body) or name

    return AgentSkill(name=name, description=description, dir=skill_dir)


def load_skills_dir(root_dir: str) -> List[AgentSkill]:
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(root_dir)

    skills: list[AgentSkill] = []
    for entry in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, entry)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "SKILL.md")):
            skills.append(load_skill(sub))
    return skills


def _first_non_empty_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return None
