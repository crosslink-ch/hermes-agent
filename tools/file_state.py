"""Cross-agent file state coordination.

Prevents mangled edits when concurrent subagents (same process, same
filesystem) touch the same file: B writes a file A already read, so A's next
write would clobber B's changes. Complements the single-agent path-overlap
check in ``agent.tool_dispatch_helpers._should_parallelize_tool_batch``. A process-wide
``FileStateRegistry`` tracks per-agent read stamps, the global last writer and
a per-path lock; every method is a no-op under ``HERMES_DISABLE_FILE_STATE_GUARD=1``.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

# (mtime, read_ts, partial). partial=True when read_file returned a windowed
# view (offset > 1 or limit < total_lines) — a later write should still warn
# so the model re-reads in full.
ReadStamp = Tuple[Optional[float], float, bool]
TaskKey = Hashable


def _raw_task_id(task_id: TaskKey) -> TaskKey:
    return task_id[0] if isinstance(task_id, tuple) and len(task_id) == 2 else task_id

# Bounded so long sessions don't accumulate unbounded state.
_MAX_PATHS_PER_AGENT = 4096
_MAX_GLOBAL_WRITERS = 4096


def _disabled() -> bool:
    # Re-read each call so tests can toggle via monkeypatch.setenv.
    return os.environ.get("HERMES_DISABLE_FILE_STATE_GUARD", "").strip() == "1"


def _mtime_or_none(resolved: str) -> Optional[float]:
    try:
        return os.path.getmtime(resolved)
    except OSError:
        return None


def _fmt_ts(ts: float) -> str:
    # Short wall-clock for warnings; avoids datetime formatting on the hot path.
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _evict_oldest(container, cap: int) -> None:
    """Pop entries until *container* is within *cap* (sets: arbitrary; dicts: oldest
    by insertion order). An eviction only costs one redundant re-send or staleness check."""
    for _ in range(len(container) - cap):
        try:
            if isinstance(container, set):
                container.pop()
            else:
                container.pop(next(iter(container)))
        except (StopIteration, KeyError):
            break


class FileStateRegistry:
    """Process-wide coordinator for cross-agent file edits."""

    def __init__(self) -> None:
        self._reads: Dict[TaskKey, Dict[str, ReadStamp]] = defaultdict(dict)
        self._last_writer: Dict[str, Tuple[TaskKey, float]] = {}
        self._path_locks: Dict[str, threading.Lock] = {}
        self._path_lock_users: Dict[str, int] = {}
        self._meta_lock = threading.Lock()  # guards _path_locks
        self._state_lock = threading.Lock()  # guards _reads + _last_writer

    @staticmethod
    def _state_path(resolved: str, namespace: Optional[str] = None) -> str:
        return f"{namespace}\0{resolved}" if namespace else resolved

    @staticmethod
    def _display_path(state_path: str) -> str:
        return state_path.split("\0", 1)[-1]

    @contextmanager
    def lock_path(self, resolved: str, namespace: Optional[str] = None):
        """Per-path lock: threads on the same path serialize, different paths proceed.
        The lock entry is dropped once the last holder/waiter exits."""
        key = self._state_path(resolved, namespace)
        with self._meta_lock:
            lock = self._path_locks.setdefault(key, threading.Lock())
            self._path_lock_users[key] = self._path_lock_users.get(key, 0) + 1
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._meta_lock:
                users = self._path_lock_users[key] - 1
                if users:
                    self._path_lock_users[key] = users
                else:
                    self._path_lock_users.pop(key, None)
                    self._path_locks.pop(key, None)

    def _stamp(self, task_id: TaskKey, resolved: str, mtime: Optional[float], now: float, partial: bool) -> None:
        """Caller holds ``_state_lock``."""
        agent_reads = self._reads[task_id]
        agent_reads[resolved] = (float(mtime) if mtime is not None else None, now, bool(partial))
        _evict_oldest(agent_reads, _MAX_PATHS_PER_AGENT)

    def record_read(self, task_id: TaskKey, resolved: str, *, partial: bool = False,
                    mtime: Optional[float] = None, namespace: Optional[str] = None,
                    stat_path: bool = True) -> None:
        if _disabled():
            return
        mtime = _mtime_or_none(resolved) if mtime is None and stat_path else mtime
        if mtime is None and stat_path:
            return
        with self._state_lock:
            self._stamp(task_id, self._state_path(resolved, namespace), mtime, time.time(), partial)

    def note_write(self, task_id: TaskKey, resolved: str, *, mtime: Optional[float] = None,
                   namespace: Optional[str] = None, stat_path: bool = True) -> None:
        """Record a successful write: global last-writer AND this agent's own
        read stamp (a write is an implicit read of the current content)."""
        if _disabled():
            return
        mtime = _mtime_or_none(resolved) if mtime is None and stat_path else mtime
        if mtime is None and stat_path:
            return
        now = time.time()
        state_path = self._state_path(resolved, namespace)
        with self._state_lock:
            self._last_writer[state_path] = (task_id, now)
            _evict_oldest(self._last_writer, _MAX_GLOBAL_WRITERS)
            self._stamp(task_id, state_path, mtime, now, False)

    def check_stale(self, task_id: TaskKey, resolved: str,
                    namespace: Optional[str] = None) -> Optional[str]:
        """Model-facing warning if this write would be stale, else ``None``. Severity
        order: sibling wrote after our read > mtime drift / partial read > never read."""
        if _disabled():
            return None
        state_path = self._state_path(resolved, namespace)
        with self._state_lock:
            stamp = self._reads.get(task_id, {}).get(state_path)
            last_writer = self._last_writer.get(state_path)

        if stamp is None and last_writer is None:  # net-new file / first touch
            return None

        if last_writer is not None:
            writer_tid, writer_ts = last_writer
            if writer_tid != task_id:
                if stamp is None:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} but this agent never read it. "
                        "Read the file before writing to avoid overwriting "
                        "the sibling's changes.")
                read_ts = stamp[1]
                if writer_ts > read_ts:
                    return (
                        f"{resolved} was modified by sibling subagent "
                        f"{writer_tid!r} at {_fmt_ts(writer_ts)} — after "
                        f"this agent's last read at {_fmt_ts(read_ts)}. "
                        "Re-read the file before writing.")

        if stamp is not None:
            read_mtime, _read_ts, partial = stamp
            if read_mtime is not None:
                current_mtime = _mtime_or_none(resolved)
                if current_mtime is None:
                    return None
                if current_mtime != read_mtime:
                    return (
                        f"{resolved} was modified since you last read it "
                        "on disk (external edit or unrecorded writer). "
                        "Re-read the file before writing.")
            if partial:
                return (
                    f"{resolved} was last read with offset/limit pagination "
                    "(partial view). Re-read the whole file before "
                    "overwriting it.")
            return None

        return (
            f"{resolved} was not read by this agent. "
            "Read the file first so you can write an informed edit.")

    def writes_since(self, exclude_task_id: str, since_ts: float,
                     paths: Iterable[str]) -> Dict[str, List[str]]:
        """``{writer_task_id: [paths]}`` for writes after ``since_ts`` by agents
        other than ``exclude_task_id`` (delegate_task's "subagent modified files
        you previously read" reminder)."""
        if _disabled():
            return {}
        paths_set = set(paths)
        exclude_raw = _raw_task_id(exclude_task_id)
        out: Dict[str, List[str]] = defaultdict(list)
        with self._state_lock:
            for p, (writer_tid, ts) in self._last_writer.items():
                if _raw_task_id(writer_tid) != exclude_raw and ts >= since_ts and p in paths_set:
                    out[str(_raw_task_id(writer_tid))].append(self._display_path(p))
        return dict(out)

    def known_reads(self, task_id: str) -> List[str]:
        """Resolved paths this agent has read."""
        if _disabled():
            return []
        with self._state_lock:
            reads: list[str] = []
            seen: set[str] = set()
            for key, paths in self._reads.items():
                if key != task_id and _raw_task_id(key) != task_id:
                    continue
                for path in paths:
                    # Preserve the internal target namespace for delegate
                    # coordination; writes_since strips it before user-facing
                    # output but uses it to avoid cross-host false conflicts.
                    read_path = path if "\0" in path else self._display_path(path)
                    if read_path not in seen:
                        reads.append(read_path)
                        seen.add(read_path)
            return reads

    def forget_task(self, task_id: str) -> None:
        """Release read stamps owned by a task after its lifecycle ends."""
        with self._state_lock:
            for key in list(self._reads):
                if key == task_id or _raw_task_id(key) == task_id:
                    self._reads.pop(key, None)

    def clear(self) -> None:
        """Reset all state. Intended for tests only."""
        with self._state_lock:
            self._reads.clear()
            self._last_writer.clear()
        with self._meta_lock:
            self._path_locks.clear()
            self._path_lock_users.clear()


_registry = FileStateRegistry()


def get_registry() -> FileStateRegistry:
    return _registry


# Convenience wrappers (short names used at call sites).
def record_read(task_id: TaskKey, resolved_or_path: str | Path, *, partial: bool = False,
                namespace: Optional[str] = None, stat_path: bool = True) -> None:
    _registry.record_read(task_id, str(resolved_or_path), partial=partial,
                          namespace=namespace, stat_path=stat_path)


def note_write(task_id: TaskKey, resolved_or_path: str | Path, *,
               namespace: Optional[str] = None,
               stat_path: bool = True) -> None:
    _registry.note_write(
        task_id, str(resolved_or_path), namespace=namespace,
        stat_path=stat_path,
    )


def check_stale(task_id: TaskKey, resolved_or_path: str | Path, *,
                namespace: Optional[str] = None) -> Optional[str]:
    return _registry.check_stale(
        task_id, str(resolved_or_path), namespace=namespace,
    )


def lock_path(resolved_or_path: str | Path, *, namespace: Optional[str] = None):
    return _registry.lock_path(str(resolved_or_path), namespace=namespace)


def writes_since(exclude_task_id: str, since_ts: float, paths: Iterable[str | Path]) -> Dict[str, List[str]]:
    return _registry.writes_since(exclude_task_id, since_ts, [str(p) for p in paths])


def known_reads(task_id: TaskKey) -> List[str]:
    reads = _registry.known_reads(task_id)
    try:
        from tools.execution_targets import resolve_execution_target

        scoped = resolve_execution_target().scope_task_key(task_id)
    except Exception:
        scoped = task_id
    if scoped != task_id:
        for path in _registry.known_reads(scoped):
            if path not in reads:
                reads.append(path)
    return reads


__all__ = [
    "FileStateRegistry",
    "get_registry",
    "record_read",
    "note_write",
    "check_stale",
    "lock_path",
    "writes_since",
    "known_reads"]
