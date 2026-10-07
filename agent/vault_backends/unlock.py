"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by backend, and expires after an idle TTL or an
explicit lock. The master password itself is consumed by the CLI call and
dropped; nothing is written to disk or env.

The surface owns the prompt: ``set_unlock_prompt_callback`` is installed by
the CLI panel / TUI gateway bridge for the current thread, exactly like the
sudo-password callback. Headless contexts (cron, webhook, api_server,
single-query) install none and the vault stays locked — the same posture
approvals take where nobody can answer.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import threading
import time
from typing import Callable, Dict, Optional

_IDLE_TTL_S = 30 * 60

_lock = threading.Lock()
_sessions: Dict[tuple[str, str], tuple[str, float]] = {}   # (profile home, backend) → (token, last_used)
_callback_tls = threading.local()

UnlockPrompt = Callable[[str, str], str]  # (backend_name, display_name) -> master password ("" = cancelled)
# (origin, site label) -> {"identifier": str, "password": str} or None when the user declines. The
# surface owns the masked fields; the tool stores the answer in the local vault and fills at once.
SaveLoginPrompt = Callable[[str, str], Optional[Dict[str, str]]]


def set_unlock_prompt_callback(cb: Optional[UnlockPrompt]) -> None:
    """Register the current surface's masked master-password prompt (per-thread slot)."""
    _callback_tls.prompt = cb


def get_unlock_prompt_callback() -> Optional[UnlockPrompt]:
    return getattr(_callback_tls, "prompt", None)


# (site, hint) -> the one-time code the user reads off their phone/email/app, "" when declined.
CodePrompt = Callable[[str, str], str]


def set_code_prompt_callback(cb: Optional[CodePrompt]) -> None:
    """Register the surface's "enter the code {site} sent you" prompt, per thread."""
    _callback_tls.code = cb


def get_code_prompt_callback() -> Optional[CodePrompt]:
    return getattr(_callback_tls, "code", None)


def set_save_login_prompt_callback(cb: Optional[SaveLoginPrompt]) -> None:
    """Register the surface's "save this login" prompt (identifier + masked password), per thread."""
    _callback_tls.save_login = cb


def get_save_login_prompt_callback() -> Optional[SaveLoginPrompt]:
    return getattr(_callback_tls, "save_login", None)


def _key(backend: str) -> tuple[str, str]:
    # Tokens are profile-scoped: a Desktop gateway hosts several profiles in one process and
    # profile B must never reuse (or lock) profile A's manager session.
    from hermes_constants import get_hermes_home
    return (str(get_hermes_home()), backend)


# Lock generation per key: ``lock()`` bumps it, and an unlock that started before the bump must
# not commit its token afterwards (a slow `bw unlock` child would otherwise silently undo an
# acknowledged Lock).
_generation: Dict[tuple[str, str], int] = {}
# Session-owned CLI/Desktop tokens; successful native shared Bitwarden tokens have no owner.
_owner_session: Dict[tuple[str, str], Optional[str]] = {}
_pending_attempts: Dict[threading.Event, tuple[tuple[str, str], Optional[str]]] = {}
_current_session = contextvars.ContextVar("vault_owner_session", default=None)
_external_access = contextvars.ContextVar("vault_external_access", default=None)
_external_live = contextvars.ContextVar("vault_external_live", default=None)
_bitwarden_access = contextvars.ContextVar("vault_bitwarden_access", default=None)


@contextmanager
def external_access_scope(allowed: bool | Callable[[], bool], *, live=None, bitwarden_allowed: Optional[bool] = None):
    """Server-attested authority; an explicit native Bitwarden grant is profile-shared.

    CLI/Desktop omit the grant and keep their session-owned token lifetime.
    This context is inherited by tool workers, never supplied by model arguments.
    """
    token = _external_access.set(allowed)
    live_token = _external_live.set(live)
    bitwarden_token = _bitwarden_access.set(bitwarden_allowed)
    try:
        yield
    finally:
        _bitwarden_access.reset(bitwarden_token)
        _external_live.reset(live_token)
        _external_access.reset(token)


def external_access_allowed(backend: str = "") -> bool:
    allowed = _external_access.get()
    if backend == "bitwarden" and _bitwarden_access.get() is not None:
        allowed = _bitwarden_access.get()
    if allowed is None:
        from tools.approval_context import _get_session_platform
        return _get_session_platform() != "thechat"
    live = _external_live.get()
    if callable(allowed):
        allowed = allowed()
    return bool(allowed and (live is None or live()))


def get_current_session_id() -> Optional[str]:
    return _current_session.get()


def set_current_session_id(session_id: Optional[str]) -> None:
    """Gateway surfaces bind the session so propagated tool workers retain token ownership."""
    _current_session.set(session_id)


def _live(backend: str, *, touch: bool) -> Optional[str]:
    if not external_access_allowed(backend):
        return None
    key = _key(backend)
    with _lock:
        entry = _sessions.get(key)
        if entry is None:
            return None
        token, last = entry
        if time.monotonic() - last > _IDLE_TTL_S:
            _forget(key)
            return None
        if touch:
            _sessions[key] = (token, time.monotonic())
        return token


def get_session_token(backend: str) -> Optional[str]:
    """Token for a real manager call; refreshes the idle timer."""
    return _live(backend, touch=True)


_attempt_generation = contextvars.ContextVar("vault_unlock_attempt", default=None)


@contextmanager
def unlock_attempt(backend: str):
    """A lock during the surface prompt must fence the later CLI token commit too."""
    key = _key(backend)
    cancelled = threading.Event()
    with _lock:
        generation = _generation.setdefault(key, 0)
        _pending_attempts[cancelled] = (key, get_current_session_id())
    token = _attempt_generation.set((key, generation, cancelled))
    try:
        yield
    finally:
        _attempt_generation.reset(token)
        with _lock:
            _pending_attempts.pop(cancelled, None)


def unlock_attempt_is_current(backend: str) -> bool:
    """Let a surface retire a prompt immediately when a native Lock/session release wins."""
    attempt = _attempt_generation.get()
    key = _key(backend)
    with _lock:
        return attempt is None or (attempt[0] == key and attempt[1] == _generation.get(key, 0)
                                   and not attempt[2].is_set())


def begin_unlock(backend: str) -> int:
    """Reuse a pre-prompt fence, or begin a direct native CLI unlock."""
    key = _key(backend)
    attempt = _attempt_generation.get()
    if attempt is not None and attempt[0] == key:
        return attempt[1]
    with _lock:
        return _generation.setdefault(key, 0)


def store_session_token(backend: str, token: str, generation: Optional[int] = None) -> bool:
    """Commit an unlock. Returns False (and drops the token) when a Lock happened since ``begin_unlock``."""
    if not external_access_allowed(backend):
        return False
    key = _key(backend)
    with _lock:
        attempt = _attempt_generation.get()
        if attempt is not None and attempt[0] == key and attempt[2].is_set():
            return False
        if generation is not None and generation != _generation.get(key, 0):
            return False
        _sessions[key] = (token, time.monotonic())
        # Native TheChat Bitwarden is shared by the profile after success; CLI/Desktop and
        # other managers retain their existing initiating-session lifetime.
        shared = backend == "bitwarden" and _bitwarden_access.get() is True
        _owner_session[key] = None if shared else get_current_session_id()
        return True


def lock(backend: Optional[str] = None) -> None:
    """Forget the current profile's session for one backend (or all of them when None)."""
    home = _key("")[0]
    with _lock:
        # Bump the generation for every key the lock names (not only the ones holding a token):
        # an unlock that is still running for this backend must see the lock when it returns.
        keys = {k for k in list(_sessions) + list(_generation) if k[0] == home and (backend is None or k[1] == backend)}
        if backend is not None:
            keys.add((home, backend))
        for key in keys:
            _forget(key)


def release_session(session_id: str) -> None:
    """Fence this session's pending work; retain successful profile-shared tokens."""
    home = _key("")[0]
    with _lock:
        for cancelled, (key, sid) in _pending_attempts.items():
            if key[0] == home and sid == session_id:
                cancelled.set()
        for key in [k for k, sid in _owner_session.items() if k[0] == home and sid == session_id]:
            _forget(key)


def _forget(key: tuple[str, str]) -> None:
    _sessions.pop(key, None)
    _owner_session.pop(key, None)
    _generation[key] = _generation.get(key, 0) + 1


def lock_all_profiles() -> None:
    """Process shutdown: drop every token."""
    with _lock:
        for key in set(_sessions) | set(_generation):
            _forget(key)


def is_unlocked(backend: str) -> bool:
    """Status probe: does NOT extend the idle TTL (only real manager calls do)."""
    return _live(backend, touch=False) is not None


def can_prompt_here() -> bool:
    """False in contexts where no human can answer (cron, webhook, api_server, -q)."""
    from tools.approval_context import (
        _is_cron_approval_context,
        _is_single_query_approval_context,
        _is_unattended_platform_approval_context,
    )
    if _is_cron_approval_context() or _is_unattended_platform_approval_context() or _is_single_query_approval_context():
        return False
    return get_unlock_prompt_callback() is not None
