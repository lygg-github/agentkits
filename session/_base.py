from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..message import ChatMessageBase


class SessionBase(ABC):
    @abstractmethod
    async def load(self, session_id: str) -> list[ChatMessageBase]: ...

    @abstractmethod
    async def append(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None: ...

    @abstractmethod
    async def save(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...

    @abstractmethod
    async def list_sessions(self) -> AsyncIterator[str]:
        if False:
            yield ""
