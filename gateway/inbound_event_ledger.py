"""Durable, lease-based inbox for externally delivered gateway events."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hermes_constants import get_hermes_home

_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MIN_REPLAY_RETENTION_SECONDS = 10 * 60
_MAX_ROWS = 10_000
_SCHEMA_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[Path] = set()


class InboundEventConflictError(ValueError):
    """The same source/event ID was reused with a different payload."""


class InboundEventCapacityError(RuntimeError):
    """The durable inbox is full of entries that are not safe to evict."""


@dataclass(frozen=True)
class InboundEventAcceptance:
    status: Literal["accepted", "pending", "completed"]


@dataclass(frozen=True)
class InboundEventClaim:
    payload: bytes
    attempt: int


def _db_path() -> Path:
    path = get_hermes_home() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _initialize(path: Path) -> None:
    if path in _INITIALIZED_PATHS:
        return
    with _SCHEMA_LOCK:
        if path in _INITIALIZED_PATHS:
            return
        conn = _connect(path)
        try:
            from hermes_state import apply_wal_with_fallback

            apply_wal_with_fallback(conn, db_label="state.db (inbound_event_ledger)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS inbound_event_receipts (
                    source TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload BLOB,
                    state TEXT NOT NULL DEFAULT 'pending',
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    received_at REAL NOT NULL,
                    updated_at REAL NOT NULL DEFAULT 0,
                    completed_at REAL,
                    PRIMARY KEY (source, event_id)
                )"""
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(inbound_event_receipts)")
            }
            additions = {
                "payload": "BLOB",
                "state": "TEXT NOT NULL DEFAULT 'pending'",
                "lease_owner": "TEXT",
                "lease_expires_at": "REAL",
                "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "next_attempt_at": "REAL NOT NULL DEFAULT 0",
                "last_error": "TEXT",
                "updated_at": "REAL NOT NULL DEFAULT 0",
                "completed_at": "REAL",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    conn.execute(
                        f"ALTER TABLE inbound_event_receipts ADD COLUMN {name} {declaration}"
                    )
            # Rows written by the pre-inbox implementation contain no recoverable
            # payload. Preserve their replay protection as completed receipts.
            conn.execute(
                """UPDATE inbound_event_receipts
                   SET state = 'completed',
                       updated_at = CASE WHEN updated_at = 0 THEN received_at ELSE updated_at END,
                       completed_at = COALESCE(completed_at, received_at)
                   WHERE payload IS NULL"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS inbound_event_receipts_due_idx
                   ON inbound_event_receipts (source, state, next_attempt_at, lease_expires_at)"""
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS inbound_event_receipts_completed_idx
                   ON inbound_event_receipts (state, completed_at)"""
            )
            conn.commit()
        finally:
            conn.close()
        _INITIALIZED_PATHS.add(path)


def _begin(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def _prune_for_acceptance(conn: sqlite3.Connection, now: float) -> None:
    conn.execute(
        """DELETE FROM inbound_event_receipts
           WHERE state = 'completed'
             AND completed_at IS NOT NULL
             AND completed_at < ?""",
        (now - _RETENTION_SECONDS,),
    )
    count = int(
        conn.execute("SELECT COUNT(*) FROM inbound_event_receipts").fetchone()[0]
    )
    if count < _MAX_ROWS:
        return

    needed = count - _MAX_ROWS + 1
    conn.execute(
        """DELETE FROM inbound_event_receipts
           WHERE rowid IN (
               SELECT rowid
               FROM inbound_event_receipts
               WHERE state = 'completed'
                 AND completed_at IS NOT NULL
                 AND completed_at < ?
               ORDER BY completed_at ASC
               LIMIT ?
           )""",
        (now - _MIN_REPLAY_RETENTION_SECONDS, needed),
    )
    count = int(
        conn.execute("SELECT COUNT(*) FROM inbound_event_receipts").fetchone()[0]
    )
    if count >= _MAX_ROWS:
        raise InboundEventCapacityError(
            "durable inbound event inbox is full of live replay-protection entries"
        )


def accept_inbound_event(
    *,
    source: str,
    event_id: str,
    payload: bytes,
    now: float | None = None,
) -> InboundEventAcceptance:
    """Durably accept raw payload bytes before the transport is acknowledged."""

    path = _db_path()
    _initialize(path)
    received_at = float(now if now is not None else time.time())
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    conn = _connect(path)
    try:
        _begin(conn)
        existing = conn.execute(
            """SELECT payload_sha256, state
               FROM inbound_event_receipts
               WHERE source = ? AND event_id = ?""",
            (source, event_id),
        ).fetchone()
        if existing is not None:
            if existing["payload_sha256"] != payload_sha256:
                raise InboundEventConflictError(
                    f"event ID {event_id!r} was reused with a different payload"
                )
            state = str(existing["state"])
            if state == "completed":
                conn.commit()
                return InboundEventAcceptance(status="completed")
            if state == "failed":
                conn.execute(
                    """UPDATE inbound_event_receipts
                       SET state = 'pending', next_attempt_at = ?, updated_at = ?
                       WHERE source = ? AND event_id = ?""",
                    (received_at, received_at, source, event_id),
                )
            conn.commit()
            return InboundEventAcceptance(status="pending")

        _prune_for_acceptance(conn, received_at)
        conn.execute(
            """INSERT INTO inbound_event_receipts (
                   source, event_id, payload_sha256, payload, state,
                   attempt_count, next_attempt_at, received_at, updated_at
               ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?, ?)""",
            (
                source,
                event_id,
                payload_sha256,
                sqlite3.Binary(payload),
                received_at,
                received_at,
                received_at,
            ),
        )
        conn.commit()
        return InboundEventAcceptance(status="accepted")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_recoverable_inbound_events(
    *,
    source: str,
    limit: int = 100,
    now: float | None = None,
    max_attempts: int = 5,
) -> list[str]:
    path = _db_path()
    _initialize(path)
    current = float(now if now is not None else time.time())
    conn = _connect(path)
    try:
        rows = conn.execute(
            """SELECT event_id
               FROM inbound_event_receipts
               WHERE source = ?
                 AND payload IS NOT NULL
                 AND attempt_count < ?
                 AND (
                     (state IN ('pending', 'failed') AND next_attempt_at <= ?)
                     OR (state = 'processing' AND lease_expires_at <= ?)
                 )
               ORDER BY received_at ASC
               LIMIT ?""",
            (source, max_attempts, current, current, max(1, min(limit, 1000))),
        ).fetchall()
        return [str(row["event_id"]) for row in rows]
    finally:
        conn.close()


def claim_inbound_event(
    *,
    source: str,
    event_id: str,
    lease_owner: str,
    lease_seconds: float,
    now: float | None = None,
    max_attempts: int = 5,
) -> InboundEventClaim | None:
    path = _db_path()
    _initialize(path)
    current = float(now if now is not None else time.time())
    conn = _connect(path)
    try:
        _begin(conn)
        row = conn.execute(
            """SELECT payload, state, lease_expires_at, next_attempt_at, attempt_count
               FROM inbound_event_receipts
               WHERE source = ? AND event_id = ?""",
            (source, event_id),
        ).fetchone()
        if row is None or row["payload"] is None:
            conn.commit()
            return None
        state = str(row["state"])
        attempt_count = int(row["attempt_count"])
        due = (
            state in {"pending", "failed"}
            and float(row["next_attempt_at"] or 0) <= current
        ) or (state == "processing" and float(row["lease_expires_at"] or 0) <= current)
        if not due or attempt_count >= max_attempts:
            conn.commit()
            return None
        attempt = attempt_count + 1
        conn.execute(
            """UPDATE inbound_event_receipts
               SET state = 'processing', lease_owner = ?, lease_expires_at = ?,
                   attempt_count = ?, updated_at = ?, last_error = NULL
               WHERE source = ? AND event_id = ?""",
            (
                lease_owner,
                current + lease_seconds,
                attempt,
                current,
                source,
                event_id,
            ),
        )
        conn.commit()
        return InboundEventClaim(payload=bytes(row["payload"]), attempt=attempt)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def renew_inbound_event_lease(
    *,
    source: str,
    event_id: str,
    lease_owner: str,
    lease_seconds: float,
    now: float | None = None,
) -> bool:
    path = _db_path()
    _initialize(path)
    current = float(now if now is not None else time.time())
    conn = _connect(path)
    try:
        cursor = conn.execute(
            """UPDATE inbound_event_receipts
               SET lease_expires_at = ?, updated_at = ?
               WHERE source = ? AND event_id = ?
                 AND state = 'processing' AND lease_owner = ?""",
            (
                current + lease_seconds,
                current,
                source,
                event_id,
                lease_owner,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def complete_inbound_event(
    *,
    source: str,
    event_id: str,
    lease_owner: str,
    now: float | None = None,
) -> bool:
    path = _db_path()
    _initialize(path)
    current = float(now if now is not None else time.time())
    conn = _connect(path)
    try:
        cursor = conn.execute(
            """UPDATE inbound_event_receipts
               SET state = 'completed', payload = NULL, lease_owner = NULL,
                   lease_expires_at = NULL, updated_at = ?, completed_at = ?
               WHERE source = ? AND event_id = ?
                 AND state = 'processing' AND lease_owner = ?""",
            (current, current, source, event_id, lease_owner),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def fail_inbound_event(
    *,
    source: str,
    event_id: str,
    lease_owner: str,
    error: str,
    retry_delay_seconds: float,
    now: float | None = None,
) -> bool:
    path = _db_path()
    _initialize(path)
    current = float(now if now is not None else time.time())
    conn = _connect(path)
    try:
        cursor = conn.execute(
            """UPDATE inbound_event_receipts
               SET state = 'failed', lease_owner = NULL, lease_expires_at = NULL,
                   next_attempt_at = ?, last_error = ?, updated_at = ?
               WHERE source = ? AND event_id = ?
                 AND state = 'processing' AND lease_owner = ?""",
            (
                current + max(0.0, retry_delay_seconds),
                error[:2000],
                current,
                source,
                event_id,
                lease_owner,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()
