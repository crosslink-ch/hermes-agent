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


def test_probe_fails_closed_if_trigger_rolls_back_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this SQLite build")
        db.create_session("s1", source="test")
        db._conn.executescript(
            """
            DROP TRIGGER messages_fts_insert;
            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                SELECT RAISE(ROLLBACK, 'constraint failed');
            END;
            """
        )
        triggers_before = _trigger_names(db._conn)
        sequence_before = db._conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'messages'"
        ).fetchone()

        assert db._probe_fts_trigger_failure() is False

        assert db._conn.in_transaction is False
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


def test_writer_adopts_peer_fail_open_between_failure_and_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    peer = None
    try:
        if not writer._fts_enabled or not writer._trigram_available:
            pytest.skip("FTS5 trigram support unavailable in this SQLite build")

        _install_legacy_inline_fts(writer)
        writer.create_session("s1", source="test")
        seed_id = writer.append_message("s1", "user", "seed")
        writer._conn.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) VALUES (?, ?)",
            (seed_id + 1, "derived ghost"),
        )
        writer._conn.commit()
        peer = SessionDB(db_path=path)
        real_probe = writer._probe_fts_trigger_failure

        def _peer_detaches_before_probe() -> bool:
            assert peer is not None
            assert peer._enter_fts_fail_open(
                sqlite3.IntegrityError("constraint failed"), confirmed=True
            )
            # With peer triggers already gone, this probe intentionally sees
            # a valid canonical insert and returns False.
            return real_probe()

        monkeypatch.setattr(
            writer, "_probe_fts_trigger_failure", _peer_detaches_before_probe
        )

        msg_id = writer.append_message("s1", "user", "survives peer race")

        assert msg_id == seed_id + 1
        assert _message_contents(path) == ["seed", "survives peer race"]
        assert writer._fts_stale is True
        assert writer._fts_enabled is False
        assert _trigger_names(writer._conn) == set()
    finally:
        writer.close()
        if peer is not None:
            peer.close()


def test_stale_recovery_rechecks_marker_before_touching_peer_triggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.db"
    stale_peer = SessionDB(db_path=path)
    recovered_peer = None
    try:
        if not stale_peer._fts_enabled:
            pytest.skip("FTS5 unavailable in this SQLite build")
        stale_peer.create_session("s1", source="test")
        stale_peer.append_message("s1", "user", "canonical")
        assert stale_peer._enter_fts_fail_open(
            sqlite3.DatabaseError("database disk image is malformed"),
            confirmed=True,
        )

        # A second constructor owns recovery and commits restored triggers plus
        # marker removal while this connection still holds stale local state.
        recovered_peer = SessionDB(db_path=path)
        triggers_after_peer = _trigger_names(recovered_peer._conn)
        assert set(_FTS_TRIGGERS).issubset(triggers_after_peer)
        assert recovered_peer._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_stale'"
        ).fetchone() is None

        def _unexpected_drop(_cursor: sqlite3.Cursor) -> None:
            raise AssertionError("stale local state must not drop peer triggers")

        monkeypatch.setattr(
            stale_peer, "_drop_all_fts_triggers", _unexpected_drop
        )
        assert stale_peer._recover_stale_fts(
            stale_peer._conn.cursor(), legacy=False
        )
        assert _trigger_names(stale_peer._conn) == triggers_after_peer
    finally:
        stale_peer.close()
        if recovered_peer is not None:
            recovered_peer.close()


def test_failed_startup_rebuild_preserves_marker_trigger_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        if not db._fts_enabled or not db._trigram_available:
            pytest.skip("FTS5 trigram support unavailable in this SQLite build")
        _install_legacy_inline_fts(db)
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "canonical")
        assert db._enter_fts_fail_open(
            sqlite3.DatabaseError("database disk image is malformed"),
            confirmed=True,
        )

        def _fail_rebuild(*_args: object, **_kwargs: object) -> None:
            raise sqlite3.DatabaseError("synthetic rebuild failure")

        monkeypatch.setattr(db, "_rebuild_legacy_fts_indexes", _fail_rebuild)
        assert db._recover_stale_fts(db._conn.cursor(), legacy=True) is False

        assert db._conn.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_stale'"
        ).fetchone() is not None
        assert _trigger_names(db._conn) == set()
        assert _message_contents(path) == ["canonical"]
    finally:
        db.close()


def test_stale_like_fallback_caps_adversarial_term_count(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "go")
        _predicate, params, _snippet = db._compile_like_boolean_query(
            " ".join(["needle"] * 999)
        )
        assert len(params) == 256 * 3

        db._fts_stale = True
        db._fts_enabled = False
        hits = db.search_messages(" ".join(["go"] * 700))
        assert any(
            "go" in ((hit.get("content") or "") + (hit.get("snippet") or ""))
            for hit in hits
        )
    finally:
        db.close()


def test_peer_marker_retry_is_bounded_for_unrelated_constraint(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this SQLite build")
        assert db._enter_fts_fail_open(
            sqlite3.DatabaseError("database disk image is malformed"),
            confirmed=True,
        )
        attempts = 0

        def _raise_constraint(_conn: sqlite3.Connection) -> None:
            nonlocal attempts
            attempts += 1
            raise sqlite3.IntegrityError("constraint failed")

        with pytest.raises(sqlite3.IntegrityError, match="^constraint failed$"):
            db._execute_write(_raise_constraint)
        assert attempts == 2
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
