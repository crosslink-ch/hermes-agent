"""Regression coverage for FTS-trigger fail-open on generic constraints.

A legacy inline FTS table can retain a rowid that no longer exists in the
canonical ``messages`` table.  The next canonical append then fails from the
synchronous FTS trigger with only ``sqlite3.IntegrityError('constraint failed')``.
That generic text must be classified with a rollback-only trigger probe before
Hermes detaches derived FTS state and retries canonical persistence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    _FTS_TRIGGERS,
)


def _install_legacy_inline_fts(db: SessionDB) -> None:
    """Replace a fresh v23 FTS layout with the deployed pre-v23 layout."""
    db._drop_all_fts_triggers(db._conn)
    db._conn.executescript(
        """
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;
        """
    )
    db._conn.executescript(LEGACY_FTS_SQL)
    db._conn.executescript(LEGACY_FTS_TRIGRAM_SQL)
    db._conn.commit()
    db._fts_enabled = True
    db._trigram_available = True


def _trigger_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'trigger' AND name LIKE 'messages_fts_%'"
        ).fetchall()
    }


def _message_contents(path: Path) -> list[str | None]:
    with sqlite3.connect(path) as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT content FROM messages ORDER BY id"
            ).fetchall()
        ]


def test_healthy_probe_is_rollback_only(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this SQLite build")
        db.create_session("s1", source="test")
        triggers_before = _trigger_names(db._conn)
        sequence_before = db._conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'messages'"
        ).fetchone()

        assert db._probe_fts_trigger_failure() is False

        assert _message_contents(path) == []
        assert _trigger_names(db._conn) == triggers_before
        assert db._conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'messages'"
        ).fetchone() == sequence_before
    finally:
        db.close()


def test_generic_constraint_from_legacy_fts_ghost_fails_open_and_recovers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if not db._fts_enabled or not db._trigram_available:
            pytest.skip("FTS5 trigram support unavailable in this SQLite build")

        _install_legacy_inline_fts(db)
        db.create_session("s1", source="test")
        seed_id = db.append_message("s1", "user", "canonical seed")
        ghost_id = seed_id + 1

        # Galli's immediate blocker: derived trigram state owns the rowid that
        # the next AUTOINCREMENT canonical message will receive.
        db._conn.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) VALUES (?, ?)",
            (ghost_id, "derived ghost"),
        )
        db._conn.commit()

        msg_id = db.append_message("s1", "user", "must survive")

        assert msg_id == ghost_id
        assert _message_contents(path) == ["canonical seed", "must survive"]
        assert db._fts_stale is True
        assert db._fts_enabled is False
        assert _trigger_names(db._conn) == set()
        assert db._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_stale'"
        ).fetchone() is not None

        # Search remains available from canonical rows while the index is stale.
        hits = db.search_messages("survive")
        assert any(
            "survive" in ((hit.get("content") or "") + (hit.get("snippet") or ""))
            for hit in hits
        )
    finally:
        db.close()

    # A later open rebuilds from canonical messages before restoring triggers.
    recovered = SessionDB(db_path=path)
    try:
        assert recovered._fts_stale is False
        assert recovered._fts_enabled is True
        assert set(_FTS_TRIGGERS).issubset(_trigger_names(recovered._conn))
        assert recovered._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_stale'"
        ).fetchone() is None
        assert recovered._conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        hits = recovered.search_messages("survive")
        assert any(
            "survive" in ((hit.get("content") or "") + (hit.get("snippet") or ""))
            for hit in hits
        )
    finally:
        recovered.close()


def test_batch_collision_after_first_row_is_confirmed_from_legacy_orphan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if not db._fts_enabled or not db._trigram_available:
            pytest.skip("FTS5 trigram support unavailable in this SQLite build")

        _install_legacy_inline_fts(db)
        db.create_session("s1", source="test")
        seed_id = db.append_message("s1", "user", "canonical seed")
        # The first row in the retried batch will use seed+1. The collision is
        # deliberately on its second row, so probing only the next id would
        # miss the damaged FTS phase.
        db._conn.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) VALUES (?, ?)",
            (seed_id + 2, "later derived ghost"),
        )
        db._conn.commit()

        inserted = db.append_messages_batch(
            "s1",
            [
                {"role": "user", "content": "batch first"},
                {"role": "assistant", "content": "batch second"},
            ],
        )

        assert inserted == 2
        assert _message_contents(path) == [
            "canonical seed",
            "batch first",
            "batch second",
        ]
        assert db._fts_stale is True
        assert _trigger_names(db._conn) == set()
    finally:
        db.close()


def test_unconfirmed_generic_constraint_is_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.IntegrityError("constraint failed")
        )
        monkeypatch.setattr(db, "_probe_fts_trigger_failure", lambda: False)

        def _raise_constraint(_conn: sqlite3.Connection) -> None:
            raise sqlite3.IntegrityError("constraint failed")

        with pytest.raises(sqlite3.IntegrityError, match="^constraint failed$"):
            db._execute_write(_raise_constraint)

        assert db._fts_enabled is True
    finally:
        db.close()
