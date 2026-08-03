"""Durable replay protection for authenticated inbound gateway events.

Webhook retries are at-least-once at the transport boundary. A valid signature
proves origin and freshness, but it does not make a repeated delivery safe. This
ledger atomically reserves the upstream event identifier in Hermes' shared
``state.db`` before an adapter schedules agent work. The primary key protects
multiple gateway processes that share one Hermes home, and the durable receipt
survives process restarts.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Literal

from hermes_constants import get_hermes_home

ReceiptStatus = Literal["reserved", "duplicate", "conflict"]

_DB_LOCK = threading.Lock()
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 10_000


def _connect() -> sqlite3.Connection:
    path = get_hermes_home() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    try:
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="state.db (inbound_event_ledger)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS inbound_event_receipts (
                source TEXT NOT NULL,
                event_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                received_at REAL NOT NULL,
                PRIMARY KEY (source, event_id)
            )"""
        )
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def reserve_inbound_event(
    *,
    source: str,
    event_id: str,
    payload: str,
    now: float | None = None,
) -> ReceiptStatus:
    """Atomically reserve one signed event before scheduling its side effects.

    ``duplicate`` means the same identifier and exact payload were already
    accepted. ``conflict`` means an identifier was reused for different bytes;
    callers must reject that request rather than silently treating it as a retry.
    Storage errors intentionally propagate so callers can fail closed and let the
    trusted sender retry later.
    """

    received_at = time.time() if now is None else now
    payload_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    with _DB_LOCK, _transaction() as conn:
        inserted = conn.execute(
            """INSERT OR IGNORE INTO inbound_event_receipts
               (source, event_id, payload_sha256, received_at)
               VALUES (?, ?, ?, ?)""",
            (source, event_id, payload_sha256, received_at),
        )
        if inserted.rowcount:
            _prune(conn, received_at)
            return "reserved"

        existing = conn.execute(
            """SELECT payload_sha256 FROM inbound_event_receipts
               WHERE source = ? AND event_id = ?""",
            (source, event_id),
        ).fetchone()
        if existing and existing[0] == payload_sha256:
            return "duplicate"
        return "conflict"


def _prune(conn: sqlite3.Connection, now: float) -> None:
    conn.execute(
        "DELETE FROM inbound_event_receipts WHERE received_at < ?",
        (now - _RETENTION_SECONDS,),
    )
    total = conn.execute("SELECT COUNT(*) FROM inbound_event_receipts").fetchone()[0]
    excess = max(0, total - _MAX_ROWS)
    if excess:
        conn.execute(
            """DELETE FROM inbound_event_receipts WHERE rowid IN (
                 SELECT rowid FROM inbound_event_receipts
                 ORDER BY received_at ASC
                 LIMIT ?
               )""",
            (excess,),
        )
