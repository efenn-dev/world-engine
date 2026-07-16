"""Append-only SQLite event log — the spine of the engine.

Every state change, Director decision, rejection, sample and verdict is an
event. Replaying the log rebuilds any past world state (see
`worldengine.world.fold`); copying a prefix into a new log is a branch.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    t      REAL NOT NULL,
    kind   TEXT NOT NULL,
    entity INTEGER,
    data   TEXT NOT NULL
);
"""


class EventLog:
    """Append-only event store backed by a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def append(self, event: Event) -> Event:
        """Append one event; return a copy carrying its assigned sequence number."""
        cur = self._conn.execute(
            "INSERT INTO events (t, kind, entity, data) VALUES (?, ?, ?, ?)",
            (event.t, event.kind, event.entity, json.dumps(event.data)),
        )
        self._conn.commit()
        return event.model_copy(update={"seq": cur.lastrowid})

    def append_many(self, events: Iterable[Event]) -> list[Event]:
        """Append events in order inside one transaction; return stored copies."""
        stored: list[Event] = []
        with self._conn:
            for event in events:
                cur = self._conn.execute(
                    "INSERT INTO events (t, kind, entity, data) VALUES (?, ?, ?, ?)",
                    (event.t, event.kind, event.entity, json.dumps(event.data)),
                )
                stored.append(event.model_copy(update={"seq": cur.lastrowid}))
        return stored

    def read(self, since: int = 0, kinds: tuple[str, ...] | None = None) -> list[Event]:
        """Read events with ``seq > since`` in order, optionally filtered by kind."""
        query = "SELECT seq, t, kind, entity, data FROM events WHERE seq > ?"
        params: list[object] = [since]
        if kinds:
            query += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        query += " ORDER BY seq"
        rows = self._conn.execute(query, params).fetchall()
        return [
            Event(seq=seq, t=t, kind=kind, entity=entity, data=json.loads(data))
            for seq, t, kind, entity, data in rows
        ]

    def last_seq(self) -> int:
        """Sequence number of the newest event, or 0 for an empty log."""
        row = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
        return int(row[0])

    def branch(self, dest: str | Path, upto_seq: int) -> "EventLog":
        """Copy events with ``seq <= upto_seq`` into a fresh log at ``dest``.

        The branch gets its own sequence numbering starting from 1; diverge it
        by appending new events. The source log is untouched.
        """
        child = EventLog(dest)
        prefix = [e.model_copy(update={"seq": None}) for e in self.read() if e.seq is not None and e.seq <= upto_seq]
        child.append_many(prefix)
        return child

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
