"""Profile/target-bound session cwd, task overrides and child container aliases."""
from __future__ import annotations

from typing import Any, Hashable, Optional

def record_session_cwd(session_key: Optional[str], cwd: Optional[str], target: Optional[str] = None, *, _resolution=None) -> None:
    """Record *cwd* as *session_key*'s working directory (after a completed
    command, or on workspace-override registration). None/empty keys collapse
    to ``"default"``; non-string / empty cwds are ignored. Keys are routed-profile
    qualified (:func:`_qualify_task_key`); the read/clear side qualifies identically."""
    from tools.terminal_tool import _session_cwd, _session_cwd_lock, _session_cwd_specs, _target_resolution
    if not isinstance(cwd, str) or not cwd.strip():
        return
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd
        if resolution.named:
            _session_cwd_specs[key] = resolution.spec_fingerprint
        else:
            _session_cwd_specs.pop(key, None)


def get_session_cwd(session_key: Optional[str], target: Optional[str] = None, *, _resolution=None) -> Optional[str]:
    """Recorded cwd for *session_key*, or None. No fallback chain on purpose:
    callers decide what an absent record means. None/empty keys read ``"default"``."""
    from tools.terminal_tool import _session_cwd, _session_cwd_lock, _session_cwd_specs, _target_resolution
    resolution = _resolution or _target_resolution(target)
    key = resolution.session_key(session_key)
    with _session_cwd_lock:
        spec = _session_cwd_specs.get(key)
        if resolution.named and spec is not None and spec != resolution.spec_fingerprint:
            return None
        return _session_cwd.get(key)


def inherit_session_cwds(parent_task_id: str, child_task_id: str) -> int:
    """Seed a child with every cwd scope currently owned by its parent."""
    from tools.terminal_tool import _profile_scoped_task_key, _session_cwd, _session_cwd_lock, _session_cwd_specs
    if not parent_task_id or not child_task_id:
        return 0
    parent_key = _profile_scoped_task_key(parent_task_id)
    child_key = _profile_scoped_task_key(child_task_id)
    inherited: dict[Hashable, str] = {}
    inherited_specs: dict[Hashable, str] = {}
    with _session_cwd_lock:
        for key, cwd in _session_cwd.items():
            if key == parent_key:
                inherited[child_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_key] = _session_cwd_specs[key]
            elif (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] == parent_key
            ):
                child_target_key = (child_key, key[1])
                inherited[child_target_key] = cwd
                if key in _session_cwd_specs:
                    inherited_specs[child_target_key] = _session_cwd_specs[key]
        _session_cwd.update(inherited)
        _session_cwd_specs.update(inherited_specs)
    return len(inherited)


def clear_session_cwd(session_key: str) -> None:
    """Drop all legacy and named-target cwd records for a raw session."""
    from tools.terminal_tool import _profile_scoped_task_key, _session_cwd, _session_cwd_lock, _session_cwd_specs
    raw = str(session_key or "default")
    scoped = _profile_scoped_task_key(raw)
    with _session_cwd_lock:
        _session_cwd.pop(raw, None)
        _session_cwd.pop(scoped, None)
        _session_cwd_specs.pop(raw, None)
        _session_cwd_specs.pop(scoped, None)
        for key in list(_session_cwd):
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and key[0] in {raw, scoped}
            ):
                _session_cwd.pop(key, None)
                _session_cwd_specs.pop(key, None)


def _sanitize_cwd_for_live_env(env: Any, new_cwd: str) -> Optional[str]:
    """Cwd to write into a LIVE cached env, or None to leave it untouched.

    On container backends a raw host path (a desktop/TUI session registering its
    workspace, e.g. ``C:\\Users\\me`` or ``/Users/me/workspace``) cannot be the
    in-sandbox workdir: every file-tools ``_exec`` wrapper does
    ``builtin cd -- <env.cwd> || exit 126``, so a host cwd poisons all later
    file operations with an unrelated ``cd:`` error. Prefix-shaped host paths
    are already rejected on the creation paths. This write classifies the
    directory mounted at ``/workspace`` as unusable before that prefix
    heuristic, then remaps the match (or a child of it) to its container mount
    instead of storing the host path. SSH maps the Hermes subprocess home onto
    the peer's home, as environment creation already does. Other backends apply
    the override verbatim (ACP project-root switching must keep working).
    """
    from tools.terminal_tool import _is_container_backend, _is_mounted_host_cwd, _is_unusable_container_cwd, coerce_ssh_remote_cwd, translate_mounted_host_path
    env_type = getattr(env, "env_type", None)
    new_cwd = coerce_ssh_remote_cwd(new_cwd, env_type)
    if not env_type or not _is_container_backend(env_type):
        return new_cwd
    host_mount = getattr(env, "host_cwd", None)
    mounted = host_mount if isinstance(host_mount, str) and host_mount else None
    # Mount equality before the prefix heuristic. /mnt and /srv are absolute,
    # so the heuristic alone would write the host path through as env.cwd and
    # every later file-tools exec would `cd` to it (exit 126).
    if not _is_unusable_container_cwd(new_cwd, mounted_host=mounted):
        return new_cwd
    if mounted:
        # The bind may sit at a fallback mount when /workspace is claimed.
        container_mount = getattr(env, "host_cwd_mount", None) or "/workspace"
        if _is_mounted_host_cwd(new_cwd, mounted):
            return container_mount
        translated = translate_mounted_host_path(new_cwd, mounted, container_mount)
        if translated:
            return translated
    return None


def register_task_env_overrides(task_id: str, overrides: dict[str, Any]):
    """Register per-task sandbox overrides (``docker_image``/``modal_image``/
    ``daytona_image`` or a plugin backend's ``<backend>_image``, ``env_type``,
    ``cwd``) before the agent loop runs.

    A ``cwd`` override takes effect immediately: it becomes the session's
    recorded cwd (until a ``cd`` changes it) and any live env's cwd is updated
    too, so env-side seeding stays consistent (ACP switching project root
    mid-session via ``session/load``). The session record keeps the RAW path
    (host workspaces are tracked there on purpose); only the live-env write is
    sanitized, since a host cwd can never be a container workdir.

    ``host_workspace`` is host-side only: a Desktop session's own folder when its ``cwd`` is left
    empty because an unmounted Docker sandbox cannot see it (project tools read it; the terminal
    never does).

    Keyed like ``_session_cwd`` — routed-profile qualified (:func:`_qualify_task_key`) — so two
    profiles registering the same task id never read each other's image/cwd (#123989).
    """
    from tools.terminal_tool import _active_environments, _env_lock, _profile_scoped_task_key, _resolve_container_task_id, _sanitize_cwd_for_live_env, _target_resolution, _task_env_overrides, record_session_cwd
    _task_env_overrides[_profile_scoped_task_key(task_id)] = overrides
    try:
        default_resolution = _target_resolution(None)
    except ValueError:
        default_resolution = None

    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        if default_resolution is not None and not (default_resolution.named and default_resolution.backend == "ssh"):
            record_session_cwd(task_id, new_cwd, _resolution=default_resolution)
        # Live env may be cached under the raw task_id (per-session surfaces)
        # or the collapsed container id (isolation-keyed rollouts); try both so
        # a CWD-only override (which collapses to "default") still finds it.
        container_id = _resolve_container_task_id(task_id, config=default_resolution.config if default_resolution else None)
        with _env_lock:
            if (
                default_resolution is not None
                and default_resolution.named
                and default_resolution.backend != "ssh"
            ):
                candidate_keys = {
                    default_resolution.environment_key(task_id),
                    default_resolution.environment_key(container_id),
                }
            else:
                candidate_keys = {task_id, container_id}
            envs = [
                env for key, env in _active_environments.items()
                if key in candidate_keys
            ]
        for env in envs:
            if getattr(env, "cwd", None) is not None:
                sanitized = _sanitize_cwd_for_live_env(env, new_cwd)
                if sanitized is not None:
                    env.cwd = sanitized


def clear_task_env_overrides(task_id: str):
    """Drop a task's overrides, cwd record and container alias (rollout cleanup)."""
    from tools.terminal_tool import _container_alias_key, _container_alias_lock, _container_aliases, _profile_scoped_task_key, _task_env_overrides, clear_session_cwd
    _task_env_overrides.pop(_profile_scoped_task_key(task_id), None)
    clear_session_cwd(task_id)
    alias_key = _container_alias_key(task_id)
    with _container_alias_lock:
        _container_aliases.pop(alias_key, None)


def register_container_alias(child_task_id: str, parent_task_id: Optional[str]) -> None:
    """Make *child_task_id* resolve to *parent_task_id*'s container (called at
    delegate_task spawn). A missing parent id aliases to ``"default"``."""
    from tools.terminal_tool import _container_alias_key, _container_alias_lock, _container_aliases
    if not child_task_id:
        return
    alias_key = _container_alias_key(child_task_id)
    with _container_alias_lock:
        _container_aliases[alias_key] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow the profile-scoped child→parent alias chain, cycle-safe."""
    from tools.terminal_tool import _container_alias_lock, _container_alias_profile_scope, _container_aliases
    profile_scope = _container_alias_profile_scope()
    seen: set[tuple[str, str]] = set()
    raw_task_id = str(task_id)
    with _container_alias_lock:
        key = (profile_scope, raw_task_id)
        while key in _container_aliases and key not in seen:
            seen.add(key)
            raw_task_id = _container_aliases[key]
            key = (profile_scope, raw_task_id)
    return raw_task_id

