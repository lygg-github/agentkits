import json
import os
from typing import AsyncIterator

from ..message import ChatMessageBase
from ._base import SessionBase


class JSONSession(SessionBase):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe = session_id.replace(os.sep, "_")
        return os.path.join(self.root, f"{safe}.json")

    async def load(self, session_id: str) -> list[ChatMessageBase]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [ChatMessageBase.from_dict(r) for r in raw]

    async def append(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        if not messages:
            return
        existing = await self.load(session_id)
        existing.extend(messages)
        await self.save(session_id, existing)

    async def save(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        path = self._path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [m.to_dict() for m in messages],
                f,
                ensure_ascii=False,
                indent=2,
            )

    async def clear(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)

    async def list_sessions(self) -> AsyncIterator[str]:
        for name in sorted(os.listdir(self.root)):
            if name.endswith(".json"):
                yield name[: -len(".json")]
