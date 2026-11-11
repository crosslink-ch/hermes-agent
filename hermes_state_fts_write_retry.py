"""One fail-open retry for a canonical write that failed in the derived FTS indexes.

``SessionDB._execute_write`` asks here when a write raised ``sqlite3.DatabaseError``: the FTS
shadow tables are derived state, so a failure that provably originates there detaches the
indexes durably and lets the canonical write run once more. Nothing is rebuilt on the live
write path.
"""

from __future__ import annotations

import sqlite3
from typing import Any


def fts_fail_open_for_write(
    db: Any, exc: sqlite3.DatabaseError, *, retry_attempted: bool, deadline: float, patience_s: float,
) -> bool:
    """True when the caller should retry the write once with FTS sync detached."""
    # A bare constraint is ambiguous: only a rollback-only trigger-on/off
    # probe can attribute it to legacy inline FTS. Canonical constraints stay errors.
    fts_failure = db._is_fts_write_corruption_error(exc)
    if not fts_failure and str(exc).strip().lower() == "constraint failed":
        fts_failure = db._probe_fts_trigger_failure()
        if not fts_failure and not retry_attempted and db._adopt_peer_fts_fail_open():
            return True
    # Never rebuild on the live write path; one retry after durable detach.
    return bool(
        fts_failure and not retry_attempted
        and db._enter_fts_fail_open(exc, confirmed=True, deadline=deadline, patience_s=patience_s)
    )
