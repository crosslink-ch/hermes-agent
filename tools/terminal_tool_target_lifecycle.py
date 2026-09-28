"""Named-target runtime lifecycle: turn leases, hot replacement, retired resources.

Only target-specific lifecycle mechanics live here; backend construction and
legacy terminal control flow remain in their dedicated modular components.
"""
from __future__ import annotations
import asyncio
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Hashable, Mapping, Optional
from tools.terminal_tool_config import _CONTAINER_BACKENDS
from tools.execution_targets import ExecutionTargetResolution

logger = logging.getLogger("tools.terminal_tool")
_active_turn_counts: dict[Hashable, int] = {}
_active_turn_counts_lock = threading.RLock()
_deferred_environment_cleanups: dict[Hashable, Hashable] = {}

_logical_environment_lease: ContextVar[Optional[_EnvironmentTurnLease]] = ContextVar(
    "logical_environment_lease", default=None,
)

_tool_environment_lease: ContextVar[Optional[_EnvironmentTurnLease]] = ContextVar(
    "tool_environment_lease", default=None,
)


def _turn_scope_key(task_id: Hashable) -> Hashable:
    from tools.terminal_tool import _profile_scoped_task_key, _resolve_container_task_id, _target_resolution
    try:
        resolution = _target_resolution(None)
        collapsed = _resolve_container_task_id(
            str(task_id),
            config=resolution.config,
        )
        return resolution.scope_task_key(collapsed)
    except Exception:
        collapsed = _resolve_container_task_id(str(task_id))
        return _profile_scoped_task_key(collapsed)


def _run_deferred_environment_cleanup(task_id: Hashable) -> None:
    from tools.terminal_tool_lifecycle import cleanup_vm
    try:
        cleanup_vm(
            task_id,
            preserve_persistent=True,
            include_collapsed=True,
        )
    except Exception:
        logger.warning(
            "Deferred environment cleanup failed for task %s",
            task_id,
            exc_info=True,
        )


def _turn_keys_overlap(left: Hashable, right: Hashable) -> bool:
    if left == right:
        return True
    if isinstance(left, tuple) and left and left[0] == right:
        return True
    if isinstance(right, tuple) and right and right[0] == left:
        return True
    return False


def _related_active_turns_unlocked(environment_key: Hashable) -> int:
    from tools.terminal_tool import _active_turn_counts
    return sum(
        count for key, count in _active_turn_counts.items()
        if _turn_keys_overlap(key, environment_key)
    )


def _register_environment_turn_key(key: Hashable) -> Hashable:
    from tools.terminal_tool import _active_turn_counts
    with _active_turn_counts_lock:
        _active_turn_counts[key] = _active_turn_counts.get(key, 0) + 1
    return key


def register_environment_turn(task_id: Hashable) -> Hashable:
    return _register_environment_turn_key(_turn_scope_key(task_id))


def _release_environment_turn_key(key: Hashable) -> int:
    from tools.terminal_tool import _active_turn_counts, _deferred_environment_cleanups
    deferred_task_ids = []
    with _active_turn_counts_lock:
        current = _active_turn_counts.get(key, 0)
        if current <= 1:
            _active_turn_counts.pop(key, None)
        else:
            _active_turn_counts[key] = current - 1
        remaining = _related_active_turns_unlocked(key)
        for deferred_key, deferred_task_id in list(
            _deferred_environment_cleanups.items()
        ):
            if _related_active_turns_unlocked(deferred_key) == 0:
                deferred_task_ids.append(deferred_task_id)
                _deferred_environment_cleanups.pop(deferred_key, None)
    for deferred_task_id in deferred_task_ids:
        _run_deferred_environment_cleanup(deferred_task_id)
    return remaining


def release_environment_turn(task_id: Hashable) -> int:
    return _release_environment_turn_key(_turn_scope_key(task_id))


def defer_environment_turn_cleanup(task_id: Hashable) -> None:
    from tools.terminal_tool import _deferred_environment_cleanups
    # Run collapsed cleanup when the final overlapping lease releases.
    key = _turn_scope_key(task_id)
    run_now = False
    with _active_turn_counts_lock:
        if _related_active_turns_unlocked(key) > 0:
            _deferred_environment_cleanups.setdefault(key, task_id)
        else:
            run_now = True
    if run_now:
        _run_deferred_environment_cleanup(task_id)


def active_environment_turns(task_id: Hashable) -> int:
    return _active_turns_for_environment_key(_turn_scope_key(task_id))


def _active_turns_for_environment_key(environment_key: Hashable) -> int:
    with _active_turn_counts_lock:
        return _related_active_turns_unlocked(environment_key)


class _EnvironmentTurnLease:
    def __init__(
        self,
        task_id: Hashable,
        *,
        environment_key: Hashable | None = None,
    ):
        self._key = (
            _register_environment_turn_key(environment_key)
            if environment_key is not None
            else register_environment_turn(task_id)
        )
        self._released = False
        self._lock = threading.Lock()

    @property
    def key(self) -> Hashable:
        return self._key

    @property
    def active(self) -> bool:
        with self._lock:
            return not self._released

    def release(self) -> int:
        with self._lock:
            if self._released:
                return _active_turns_for_environment_key(self._key)
            self._released = True
        return _release_environment_turn_key(self._key)


@contextmanager
def logical_environment_turn(task_id: Hashable):
    # Hold the shared environment for one complete logical conversation turn.
    lease = _EnvironmentTurnLease(task_id)
    token = _logical_environment_lease.set(lease)
    try:
        yield lease
    finally:
        lease.release()
        _logical_environment_lease.reset(token)


def release_logical_environment_turn(task_id: Hashable) -> int:
    # Release this context's logical lease before final cleanup.
    lease = _logical_environment_lease.get()
    key = _turn_scope_key(task_id)
    if lease is not None and lease.key == key:
        return lease.release()
    return active_environment_turns(task_id)


def release_logical_environment_turn_for_cleanup(task_id: Hashable) -> bool:
    # Preserve the established boolean cleanup-hook contract.
    lease = _logical_environment_lease.get()
    if lease is None or lease.key != _turn_scope_key(task_id):
        return False
    lease.release()
    return True


def execution_environment_turn_key(
    function_name: str,
    arguments: Mapping[str, Any],
    *,
    task_id: Hashable | None = None,
) -> Hashable | None:
    from tools.terminal_tool import _resolve_container_task_id
    if function_name not in {
        "terminal", "read_file", "write_file", "patch", "search_files",
        "execute_code", "process",
    }:
        return None
    task_id = arguments.get("task_id") or task_id
    if not task_id:
        return None
    if function_name == "process":
        # Follow-up calls select a persisted session_id rather than a target.
        # A parent-scope lease safely covers whichever named runtime owns it.
        return _turn_scope_key(task_id)
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = resolve_execution_target(arguments.get("execution_target"))
        base_task_id = _resolve_container_task_id(
            str(task_id),
            config=resolution.config,
        )
        return resolution.session_key(base_task_id)
    except Exception:
        # Invalid-target tools still execute to return their normal user-visible
        # validation error; the raw logical lease remains the safe fallback.
        return None


@contextmanager
def environment_turn_usage(
    task_id: Hashable,
    *,
    environment_key: Hashable | None = None,
):
    # Protect one terminal, file, or code invocation from idle cleanup.
    lease = _EnvironmentTurnLease(task_id, environment_key=environment_key)
    token = _tool_environment_lease.set(lease)
    try:
        yield
    finally:
        lease.release()
        _tool_environment_lease.reset(token)


def _current_owned_environment_turns(environment_key: Hashable) -> int:
    # Count this call's own logical/tool leases for replacement checks.
    owned = 0
    for lease in (
        _logical_environment_lease.get(),
        _tool_environment_lease.get(),
    ):
        if (
            lease is not None
            and lease.active
            and _turn_keys_overlap(lease.key, environment_key)
        ):
            owned += 1
    return owned


def _build_environment_constructor_configs(
    config: Dict[str, Any],
    resolution: 'ExecutionTargetResolution',
    base_task_id: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build backend constructor inputs from one canonical normalized config."""
    env_type = config["env_type"]
    container_config: Optional[Dict[str, Any]] = None
    if env_type in _CONTAINER_BACKENDS:
        container_config = {
            "container_cpu": config.get("container_cpu", 1),
            "container_memory": config.get("container_memory", 5120),
            "container_disk": config.get("container_disk", 51200),
            "container_persistent": config.get("container_persistent", True),
            "vercel_runtime": config.get("vercel_runtime", ""),
            "modal_mode": config.get("modal_mode", "auto"),
            "docker_volumes": config.get("docker_volumes", []),
            "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
            "docker_forward_env": config.get("docker_forward_env", []),
            "docker_env": config.get("docker_env", {}),
            "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
            "docker_extra_args": config.get("docker_extra_args", []),
            "docker_network": config.get("docker_network", True),
            "docker_shm_size": config.get("docker_shm_size", "1g"),
            "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
            "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
            "lifetime_seconds": config.get("lifetime_seconds", 300),
            "storage_task_id": resolution.storage_task_id(base_task_id),
            "legacy_storage_task_id": resolution.legacy_backend_task_id(base_task_id),
        }

    ssh_config: Optional[Dict[str, Any]] = None
    if env_type == "ssh":
        ssh_config = {
            "host": config.get("ssh_host", ""),
            "user": config.get("ssh_user", ""),
            "port": config.get("ssh_port", 22),
            "key": config.get("ssh_key", ""),
            "persistent": config.get("ssh_persistent", False),
            "runtime_scope": resolution.security_scope if resolution.named else "",
        }

    local_config: Optional[Dict[str, Any]] = None
    if env_type == "local":
        local_config = {"persistent": config.get("local_persistent", False)}
    return container_config, ssh_config, local_config


def _record_environment_lifetime(env: Any, config: Dict[str, Any]) -> None:
    """Attach the resolved target's idle lifetime to its environment."""
    try:
        env._hermes_lifetime_seconds = int(config["lifetime_seconds"])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


def _record_environment_target(env: Any, resolution: Any) -> None:
    """Bind a created environment to the exact resolved named-target spec."""
    try:
        setattr(env, "_hermes_target_name", resolution.target)
        setattr(
            env, "_hermes_target_fingerprint",
            resolution.spec_fingerprint if resolution.named else None,
        )
        setattr(env, "_hermes_target_backend", resolution.backend)
        setattr(
            env, "_hermes_target_scope",
            resolution.security_scope if resolution.named else None,
        )
        setattr(env, "_hermes_target_resolution", resolution)
        persistent = resolution.config.get("container_persistent", True)
        if isinstance(persistent, str):
            persistent = persistent.strip().lower() in {"1", "true", "yes", "on"}
        setattr(
            env,
            "_hermes_stable_storage",
            resolution.backend == "docker" and bool(persistent),
        )
    except (AttributeError, TypeError):
        pass


def _environment_matches_target(env: Any, resolution: Any) -> bool:
    """Reject cache reuse after a named target's effective config changes."""
    if env is None or not resolution.named:
        return env is not None
    fingerprint = getattr(env, "_hermes_target_fingerprint", None)
    # Third-party/test-provided environments predating named targets have no
    # binding metadata. Preserve their registration contract; every environment
    # created by core Hermes is stamped before entering the cache.
    if fingerprint is None:
        return True
    return (
        fingerprint == resolution.spec_fingerprint
        and getattr(env, "_hermes_target_name", resolution.target) == resolution.target
        and getattr(env, "_hermes_target_backend", resolution.backend) == resolution.backend
    )


def _environment_has_stable_storage(env: Any) -> bool:
    return bool(getattr(env, "_hermes_stable_storage", False))


def _has_active_environment_process(env: Any) -> bool:
    """Retain retired resources while their tracked background jobs own them."""
    from tools.process_registry import process_registry
    checker = getattr(process_registry, "has_active_environment", None)
    if checker is not None:
        return bool(checker(env))
    with process_registry._lock:
        return any(session.env_ref is env and not session.exited
                   for session in process_registry._running.values())


def _environment_replacement_is_busy(env: Any, environment_key: Hashable) -> bool:
    """Protect shared persistent storage while the old runtime is still active."""
    if not _environment_has_stable_storage(env):
        return False
    active = _active_turns_for_environment_key(environment_key)
    owned = _current_owned_environment_turns(environment_key)
    if active > owned:
        return True
    try:
        from tools.process_registry import process_registry

        return _has_active_environment_process(env)
    except Exception:
        logger.warning("Cannot determine whether old runtime has active processes", exc_info=True)
        return True


def _cleanup_environment_resource(
    env: Any,
    *,
    force_remove: bool = False,
    preserve_storage: bool = False,
) -> None:
    """Stop one environment, optionally preserving its persistent storage."""
    import inspect

    ownership_attrs = {}
    if force_remove:
        # A replaced environment is unreachable by configuration and must not
        # retain persist-mode lifecycle semantics. Persistent storage can remain
        # owned by the stable profile/target storage identity while the obsolete
        # runtime is removed.
        attrs = ["_persist_across_processes"]
        if not preserve_storage:
            attrs.extend(["_persistent", "persistent_filesystem"])
        for attr in attrs:
            if hasattr(env, attr):
                try:
                    ownership_attrs[attr] = getattr(env, attr)
                    setattr(env, attr, False)
                except (AttributeError, TypeError):
                    pass

    try:
        if hasattr(env, "cleanup"):
            cleanup = env.cleanup
            kwargs = {}
            if force_remove:
                try:
                    if "force_remove" in inspect.signature(cleanup).parameters:
                        kwargs["force_remove"] = True
                except (TypeError, ValueError):
                    pass
            result = cleanup(**kwargs)
        elif hasattr(env, "stop"):
            result = env.stop()
        elif hasattr(env, "terminate"):
            result = env.terminate()
        else:
            return

        if inspect.isawaitable(result):
            import asyncio

            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(result)
                loop.close()
            except Exception:
                try:
                    close = getattr(result, "close", None)
                    if close is not None:
                        close()
                except Exception:
                    pass
                raise

        wait_fn = getattr(env, "wait_for_cleanup", None)
        if wait_fn is not None and not wait_fn(timeout=60.0):
            raise RuntimeError("environment cleanup did not finish within 60 seconds")
    except BaseException:
        # Cleanup can fail before the obsolete runtime is actually removed. In
        # that case the caller restores the handle to the active cache, so also
        # restore its persistence ownership flags rather than leaving a live
        # runtime reachable but disowned by this process.
        for attr, value in ownership_attrs.items():
            try:
                setattr(env, attr, value)
            except (AttributeError, TypeError):
                pass
        raise


class _EnvironmentReplacementError(RuntimeError):
    """Base error for a fail-closed named-target runtime replacement."""


class _EnvironmentReplacementBusyError(_EnvironmentReplacementError):
    """The previous stable-storage runtime still has active users."""


class _EnvironmentReplacementCleanupError(_EnvironmentReplacementError):
    """The previous stable-storage runtime could not be retired safely."""


def _prepare_environment_replacement(
    env: Any,
    environment_key: Hashable,
    *,
    target_name: str,
) -> bool:
    """Retire an idle stable-storage runtime before creating its replacement.

    Persistent Docker generations share one storage identity, so the obsolete
    runtime must be gone before the replacement container is created. The
    caller holds the per-environment creation lock; this helper owns the shared
    detach/cleanup/restore handoff used by terminal, file, and execute_code.
    """
    from tools.terminal_tool import _active_environments, _env_lock, _last_activity
    if env is None:
        return False
    if _environment_replacement_is_busy(env, environment_key):
        raise _EnvironmentReplacementBusyError(
            f"Execution target {target_name!r} changed while its persistent "
            "Docker runtime is still active. Wait for its commands/background "
            "processes to finish, then retry."
        )
    if not _environment_has_stable_storage(env):
        return False

    with _env_lock:
        if _active_environments.get(environment_key) is not env:
            raise _EnvironmentReplacementBusyError(
                f"Execution target {target_name!r} changed again while its "
                "previous runtime was being retired. Retry the operation."
            )
        owned_keys = [
            (key, key in _last_activity, _last_activity.get(key, 0.0))
            for key, candidate in list(_active_environments.items())
            if candidate is env
        ]
        for key, _, _ in owned_keys:
            _active_environments.pop(key, None)
            _last_activity.pop(key, None)

    try:
        _cleanup_environment_resource(
            env,
            force_remove=True,
            preserve_storage=True,
        )
    except BaseException as exc:
        with _env_lock:
            for key, had_activity, activity in owned_keys:
                if key not in _active_environments:
                    _active_environments[key] = env
                    if had_activity:
                        _last_activity[key] = activity
        if isinstance(exc, Exception):
            raise _EnvironmentReplacementCleanupError(
                "Could not retire the previous persistent Docker runtime for "
                f"execution target {target_name!r}: {exc}"
            ) from exc
        raise
    return True


def _retire_replaced_environment(env: Any, task_key: Hashable) -> None:
    """Defer teardown until no operation/process can still reference *env*."""
    from tools.terminal_tool import _retired_environments, _retired_environments_lock
    if env is None:
        return
    with _retired_environments_lock:
        if all(existing_env is not env for _, existing_env, _ in _retired_environments):
            _retired_environments.append((task_key, env, time.time()))


def _collect_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> list[tuple[Hashable, Any]]:
    """Detach retired resources that are old enough and no longer in use."""
    from tools.terminal_tool import _retired_environments, _retired_environments_lock
    now = time.time()
    ready: list[tuple[Hashable, Any]] = []
    keep: list[tuple[Hashable, Any, float]] = []
    try:
        from tools.process_registry import process_registry
    except ImportError:
        process_registry = None

    with _retired_environments_lock:
        candidates = list(_retired_environments)
        _retired_environments.clear()

    for retired_key, env, retired_at in candidates:
        if task_key is not None and retired_key != task_key:
            keep.append((retired_key, env, retired_at))
            continue
        busy = False
        if require_idle:
            busy = _active_turns_for_environment_key(retired_key) > 0
            if not busy and process_registry is not None:
                busy = (process_registry.has_active_processes(retired_key)
                        or _has_active_environment_process(env))
        if busy or now - retired_at < min_age_seconds:
            keep.append((retired_key, env, retired_at))
        else:
            ready.append((retired_key, env))

    # Merge records retired concurrently while we performed potentially slow
    # process liveness checks. Avoid duplicate records by environment identity.
    ready_ids = {id(env) for _, env in ready}
    with _retired_environments_lock:
        concurrent = [
            record for record in _retired_environments
            if id(record[1]) not in ready_ids
        ]
        seen = {id(record[1]) for record in concurrent}
        concurrent.extend(
            record for record in keep
            if id(record[1]) not in seen
        )
        _retired_environments[:] = concurrent
    return ready


def _cleanup_retired_environments(
    *,
    task_key: Hashable | None = None,
    min_age_seconds: float = 60.0,
    require_idle: bool = True,
) -> int:
    """Force-remove retired environments selected by lifecycle policy."""
    ready = _collect_retired_environments(
        task_key=task_key,
        min_age_seconds=min_age_seconds,
        require_idle=require_idle,
    )
    cleaned = 0
    for retired_key, env in ready:
        try:
            _cleanup_environment_resource(
                env,
                force_remove=True,
                preserve_storage=_environment_has_stable_storage(env),
            )
            cleaned += 1
            logger.info("Cleaned retired environment for task: %s", retired_key)
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "not found" in error_str.lower():
                cleaned += 1
                logger.info("Retired environment for task %s was already gone", retired_key)
            else:
                logger.warning(
                    "Error cleaning retired environment for task %s: %s",
                    retired_key, exc,
                )
                _retire_replaced_environment(env, retired_key)
    return cleaned
