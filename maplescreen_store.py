"""Persist captured chat messages and their screen regions in SQLite."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS capture_sessions (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES capture_sessions(id),
    observed_at TEXT NOT NULL,
    medal TEXT,
    player_name TEXT,
    channel TEXT,
    text TEXT NOT NULL,
    message_type TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ChatStore:
    """Use a short-lived connection per operation so callers can share the store."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL").fetchone()
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection

    def start_session(self, region: tuple[int, int, int, int]) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """INSERT INTO capture_sessions (started_at, x, y, width, height)
                   VALUES (?, ?, ?, ?, ?) RETURNING id""",
                (_utc_now(), *region),
            ).fetchone()
            return row["id"]

    def finish_session(self, session_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE capture_sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (_utc_now(), session_id),
            )

    def append(self, session_id: int, observed_at: str, messages: list[dict]) -> list[dict]:
        if not messages:
            return []
        committed = []
        with self._connect() as connection:
            for message in messages:
                row = connection.execute(
                    """INSERT INTO chat_messages
                           (session_id, observed_at, medal, player_name, channel, text, message_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *""",
                    (session_id, observed_at, message["medal"], message["player_name"],
                     message["channel"], message["text"], message["message_type"]),
                ).fetchone()
                committed.append(dict(row))
        return committed

    def recent(self, limit: int = 500) -> list[dict]:
        """Return up to limit committed messages across sessions, oldest first."""
        if limit < 0:
            raise ValueError("The message limit must be nonnegative.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        # Older captures stored a newline at each screen wrap.
        return [dict(row) | {"text": row["text"].replace("\n", "")} for row in reversed(rows)]
