"""Background process record and synchronized output buffer."""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

MAX_OUTPUT_CHARS = 200_000

@dataclass
class ProcessSession:
    """A tracked background process with output buffering."""
    id: str                                     # "proc_xxxxxxxxxxxx"
    command: str
    task_id: str = ""                           # Task/sandbox isolation key (CONTAINER key,
                                                # may be collapsed by _resolve_container_task_id)
    owner_task_id: str = ""                     # RAW spawning task id ("sa-..."); ownership
                                                # checks must use this, not task_id
    session_key: str = ""                       # Gateway session key (reset protection)
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None  # Popen handle (local only)
    env_ref: Any = None                         # Environment object (sandbox spawns)
    cwd: Optional[str] = None
    started_at: float = 0.0                     # time.time() of spawn
    host_start_time: Optional[int] = None       # kernel start ticks (/proc/<pid>/stat f22) — PID-reuse guard
    exited: bool = False
    exited_at: float = 0.0                      # time.time() of the FIRST move to finished (0 = unknown)
    exit_code: Optional[int] = None             # None while running
    completion_reason: str = "exited"           # exited|killed|lost|failed_start|already_exited
    termination_source: str = ""                # process.kill|kill_all|backend_lost|failed_start
    output_buffer: str = ""                     # Rolling tail (last max_output_chars)
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False                      # Recovered from checkpoint (no pipe)
    pid_scope: str = "host"                     # "host" for local/PTY PIDs, "sandbox" for env-local PIDs
    systemd_unit: str = ""                      # transient scope unit name when spawned under systemd-run
    wsl_chain: bool = False                     # spawned via wsl[.exe]: the host PID is the short-lived
                                                # launcher; Linux-side workers outlive it (#120546)
    handoff_note: str = ""                      # why a subagent handed this process to its parent (rides the notice)
    persist_on_release: bool = False           # opt out of agent-lifecycle cleanup (release()/turn-abandon kill
                                                # sweeps), per terminal(background=true, persist_on_release=true) (#41225)
    output_log: str = ""                        # file the process writes to (worker spawns; no pipe to lose)
    pty_fallback: str = ""                      # why a pty=true local spawn runs in pipe mode instead
    # Watcher/notification routing (persisted for crash recovery)
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""                # Triggering message id — reply anchor for topic routing
    watcher_interval: int = 0                   # 0 = no watcher configured
    # Session-db id of the spawning conversation; lets the gateway drop completions whose
    # session was closed at a user boundary (/new) instead of injecting into the NEW one.
    parent_session_id: str = ""
    notify_on_complete: bool = False            # Queue agent notification on exit
    completion_output_chars: int = 0            # Output chars the completion carries; 0 = COMPLETION_OUTPUT_CHARS
    watch_patterns: list[str] = field(default_factory=list)
    heartbeat_seconds: int = 0                  # 0 = off; else a "heartbeat" event every N s while running
    total_output_chars: int = 0                 # Chars ever ingested (the buffer is a rolling tail)
    _heartbeat_last: float = field(default=0.0, repr=False)          # time of the last heartbeat (or spawn)
    _heartbeat_total_at_last: int = field(default=0, repr=False)     # total_output_chars at that moment
    _heartbeat_seq: int = field(default=0, repr=False)
    _watch_hits: int = field(default=0, repr=False)          # total matches delivered
    _watch_suppressed: int = field(default=0, repr=False)    # matches dropped by rate limit
    _watch_disabled: bool = field(default=False, repr=False) # permanently killed after strike limit
    # Rate-limit window state (see WATCH_*). A strike is a WINDOW with drops, not a drop.
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _completion_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # Set by the _move_to_finished call that moved the session out of _running; only that call
    # publishes the completion, so a duplicate finisher must not release waiters early.
    _finish_claimed: bool = field(default=False, repr=False)
    # The last stop of ``systemd_unit`` failed, so its descendants may still be alive; bulk
    # cleanup retries the stop even after the session finished.
    _scope_stop_pending: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _reader_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _reader_finish_requested: threading.Event = field(default_factory=threading.Event, repr=False)
    _reader_selectable: bool = field(default=False, repr=False)
    _pty: Any = field(default=None, repr=False)  # ptyprocess handle (use_pty=True)
    _owned_pgid: Optional[int] = field(default=None, repr=False)  # see ProcessOwnershipMixin
    # Trailing fields keep existing positional constructors compatible.
    target: str = ""
    backend: str = ""
    timeout_seconds: int = 0
    environment_task_key: str = ""
    runtime_scope: str = ""

    def __post_init__(self):
        # A session built without an explicit owner is owned by its own task, so ownership checks compare
        # ``owner_task_id`` alone instead of repeating an ``or task_id`` fallback at every call site.
        if not self.owner_task_id:
            self.owner_task_id = self.task_id

    def append_output(self, text: str) -> None:
        """Append to the rolling output buffer under the session lock, keeping the tail."""
        with self._lock:
            self._append_locked(text)

    def append_output_if_running(self, text: str) -> bool:
        """Append unless the session has exited. Decided under the lock a kill holds while it
        snapshots the output and sets ``exited``, so a chunk is either in the kill's receipt or
        dropped, never added after it."""
        with self._lock:
            if self.exited:
                return False
            self._append_locked(text)
        return True

    def _append_locked(self, text: str) -> None:
        self.output_buffer += text
        self.total_output_chars += len(text)
        if len(self.output_buffer) > self.max_output_chars:
            self.output_buffer = self.output_buffer[-self.max_output_chars:]

    def mark_exited(self, exit_code, reason: str = "exited", source: str = "") -> None:
        """Record an exit. A kill that raced the observer already recorded its own
        exit_code/reason; never overwrite it."""
        self.exited = True
        if self.completion_reason != "killed":
            self.exit_code = exit_code
            self.completion_reason = reason
            if source:
                self.termination_source = source

