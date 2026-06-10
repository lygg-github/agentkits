from copy import deepcopy
from typing import AsyncIterator

from ..message import ChatMessageBase
from ._base import SessionBase


class MemorySession(SessionBase):
    def __init__(self) -> None:
        self._data: dict[str, list[ChatMessageBase]] = {}

    async def load(self, session_id: str) -> list[ChatMessageBase]:
        return list(self._data.get(session_id, []))

    async def append(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        if not messages:
            return
        bucket = self._data.setdefault(session_id, [])
        bucket.extend(deepcopy(messages))

    async def save(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        self._data[session_id] = deepcopy(list(messages))

    async def clear(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    async def list_sessions(self) -> AsyncIterator[str]:
        for sid in list(self._data):
            yield sid
