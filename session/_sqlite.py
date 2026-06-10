import asyncio
import json
import sqlite3
import threading
from typing import AsyncIterator

from ..message import ChatMessageBase
from ._base import SessionBase


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agentkits_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS agentkits_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES agentkits_sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_agentkits_messages_session
ON agentkits_messages (session_id, id);
"""


class SQLiteSession(SessionBase):
    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _ensure_session(self, session_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO agentkits_sessions (session_id) VALUES (?)",
            (session_id,),
        )

    def _insert_many(self, session_id: str, messages: list[ChatMessageBase]) -> None:
        self._ensure_session(session_id)
        self._conn.executemany(
            "INSERT INTO agentkits_messages (session_id, payload) VALUES (?, ?)",
            [(session_id, json.dumps(m.to_dict(), ensure_ascii=False)) for m in messages],
        )
        self._conn.execute(
            "UPDATE agentkits_sessions SET updated_at = CURRENT_TIMESTAMP "
            "WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    def _fetch_all(self, session_id: str) -> list[ChatMessageBase]:
        cur = self._conn.execute(
            "SELECT payload FROM agentkits_messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        out: list[ChatMessageBase] = []
        for (payload,) in cur.fetchall():
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            out.append(ChatMessageBase.from_dict(data))
        return out

    def _delete_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM agentkits_messages WHERE session_id = ?",
            (session_id,),
        )
        self._conn.execute(
            "DELETE FROM agentkits_sessions WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()

    async def load(self, session_id: str) -> list[ChatMessageBase]:
        def _run() -> list[ChatMessageBase]:
            with self._lock:
                return self._fetch_all(session_id)

        return await asyncio.to_thread(_run)

    async def append(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        if not messages:
            return

        def _run() -> None:
            with self._lock:
                self._insert_many(session_id, messages)

        await asyncio.to_thread(_run)

    async def save(
        self,
        session_id: str,
        messages: list[ChatMessageBase],
    ) -> None:
        def _run() -> None:
            with self._lock:
                self._delete_session(session_id)
                if messages:
                    self._insert_many(session_id, messages)

        await asyncio.to_thread(_run)

    async def clear(self, session_id: str) -> None:
        def _run() -> None:
            with self._lock:
                self._delete_session(session_id)

        await asyncio.to_thread(_run)

    async def list_sessions(self) -> AsyncIterator[str]:
        def _run() -> list[str]:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT session_id FROM agentkits_sessions ORDER BY updated_at DESC",
                )
                return [row[0] for row in cur.fetchall()]

        for sid in await asyncio.to_thread(_run):
            yield sid

    def close(self) -> None:
        with self._lock:
            self._conn.close()
