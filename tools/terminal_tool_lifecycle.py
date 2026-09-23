"""Sandbox lifecycle for the terminal tool: idle reaping, teardown, manual/atexit
cleanup, and the lazy ensure_task_env bring-up. The env cache dicts and locks
stay in tools.terminal_tool (tests patch them there) and are read through it
at call time.

Split out of ``tools/terminal_tool.py``; every public/patched name is re-imported there,
so ``tools.terminal_tool.<name>`` keeps resolving (and monkeypatching) as before.
"""
from __future__ import annotations

import glob
import logging
import inspect
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Hashable
from tools.environments.singularity import _get_scratch_dir
from tools.terminal_tool_backends import (
    _container_config_from_config,
    _ssh_config_from_config,
)
from tools.terminal_tool_config import _quiet, _CONTAINER_BACKENDS, _is_unusable_container_cwd

def _environment_is_persistent(env: Any) -> bool:
    return bool(getattr(env, "_session_scoped", False) or getattr(env, "_persistent", False)
                or getattr(env, "persistent_filesystem", False))

# Log-record parity with the origin module.
logger = logging.getLogger("tools.terminal_tool")


# Advisory disk-usage check; cached so the recursive scan doesn't run on
# every command (a result up to 5 minutes stale is harmless).
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}

_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _scratch_paths():
    return glob.glob(str(_get_scratch_dir() / "hermes-*"))


def _check_disk_usage_warning():
    """True when hermes scratch dirs exceed the warning threshold (cached, advisory)."""
    from tools.terminal_tool import DISK_USAGE_WARNING_THRESHOLD_GB
    if time.monotonic() - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        total_bytes = 0
        for path in _scratch_paths():
            for f in Path(path).rglob('*'):
                if f.is_file():
                    with _quiet("Could not stat file %s", f, exc=OSError):
                        total_bytes += f.stat().st_size
        total_gb = total_bytes / (1024 ** 3)
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = time.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception:
        # Don't update cache on error so the next call retries.
        logger.debug("Disk usage warning check failed", exc_info=True)
        return False


def _create_configured_env(
    config: Dict[str, Any], env_type: str, *, image: str, cwd: str, timeout: int,
    task_id: str, host_cwd: Optional[str], local_config: Optional[dict] = None,
    container_config: Optional[dict] = None, ssh_config: Optional[dict] = None,
    session_scoped: bool = False,
):
    """``_create_environment`` with the ssh/container kwargs shaped from *config*
    (shared by the terminal tool and the lazy :func:`ensure_task_env` bring-up)."""
    from tools.terminal_tool import _create_environment
    from tools.terminal_tool_config import _is_container_backend
    return _create_environment(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        ssh_config=ssh_config if ssh_config is not None else (_ssh_config_from_config(config) if env_type == "ssh" else None),
        container_config=(
            container_config if container_config is not None else
            (_container_config_from_config(config) if _is_container_backend(env_type) else None)
        ),
        local_config=local_config, task_id=task_id, host_cwd=host_cwd,
        session_scoped=session_scoped,
    )


def _cleanup_env(env: Any, *, force_remove: Optional[bool] = None) -> None:
    """Tear down one environment via cleanup()/stop()/terminate(), whichever it has.

    ``force_remove`` is forwarded to ``cleanup()`` only when given and the backend's
    signature accepts it (``DockerEnvironment``, issue #20561; other backends don't).
    Shared by ``cleanup_vm``, the idle reaper and the prompt-time backend probe so
    the signature check lives in one place.
    """
    if hasattr(env, 'cleanup'):
        if force_remove is not None and "force_remove" in inspect.signature(env.cleanup).parameters:
            env.cleanup(force_remove=force_remove)
        else:
            env.cleanup()
    elif hasattr(env, 'stop'):
        env.stop()
    elif hasattr(env, 'terminate'):
        env.terminate()


def _teardown_env(env: Any, task_id: str, *, force_remove: Optional[bool] = None, done_msg: str = "Cleaned up inactive environment for task: %s") -> None:
    """``_cleanup_env`` plus outcome logging. A 404/"not found" error means the
    sandbox is already gone — logged at info."""
    try:
        _cleanup_env(env, force_remove=force_remove)
        logger.info(done_msg, task_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _clear_file_ops_cache(task_id: str) -> None:
    """Invalidate the file_ops cache entry so ShellFileOperations can't reference a dead sandbox."""
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass


def _unregister_env(task_id: str):
    """Pop *task_id* from the env cache, activity map and creation locks; return
    the env (or None). Callers run the (slow) teardown OUTSIDE the lock —
    Modal/Docker teardown can block 10-15s and would stall every concurrent
    terminal/file tool call."""
    from tools.terminal_tool import (
        _active_environments, _creation_locks, _creation_locks_lock, _env_lock,
        _last_activity,
    )
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)
    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)
    return env


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    from tools.terminal_tool import _active_environments, _creation_locks, _creation_locks_lock, _env_lock, _last_activity
    from tools.terminal_tool_target_lifecycle import _active_turns_for_environment_key, _cleanup_retired_environments
    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if _active_turns_for_environment_key(task_id) > 0:
                # An active tool or overlapping logical turn owns this runtime.
                # Refresh activity so it gets a complete idle window afterward.
                _last_activity[task_id] = current_time
                continue
            tracked_env = _active_environments.get(task_id)
            effective_lifetime = getattr(
                tracked_env, "_hermes_lifetime_seconds", lifetime_seconds,
            )
            if current_time - last_time > effective_lifetime:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)

    # Replaced environments are no longer selectable. Give concurrent foreground
    # calls a one-minute grace period, then force-remove them once no tool turn or
    # background process still references their task scope.
    _cleanup_retired_environments(min_age_seconds=60.0, require_idle=True)


def get_active_env(task_id: str, target: Optional[str] = None):
    """Return the active BaseEnvironment for *task_id*, or None."""
    from tools.terminal_tool import _active_environments, _env_lock, _environment_scope_key, _resolve_container_task_id, _target_resolution
    resolution = _target_resolution(target)
    lookup = _environment_scope_key(
        _resolve_container_task_id(task_id, config=resolution.config),
        resolution,
    )
    raw_lookup = _environment_scope_key(task_id, resolution)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(raw_lookup)


def ensure_task_env(
    task_id: Optional[str] = None,
    target: Optional[str] = None,
):
    """Lazily create and cache the sandbox environment for *task_id*.

    This is used by non-terminal consumers such as ``vision_analyze``. It must
    resolve and stamp the same named-target identity as ``terminal_tool``;
    otherwise a first-use image read can create an unscoped environment that a
    later terminal call cannot find or safely reuse.
    """
    from tools.terminal_tool import _active_environments, _apply_task_cwd_override, _create_environment, _creation_locks, _creation_locks_lock, _docker_environment_is_session_scoped, _env_lock, _environment_scope_key, _get_env_config, _last_activity, _resolve_container_task_id, _resolve_task_host_cwd, _start_cleanup_thread, _target_resolution, get_session_cwd, resolve_task_overrides
    from tools.terminal_tool_target_lifecycle import _EnvironmentReplacementError, _build_environment_constructor_configs, _cleanup_environment_resource, _environment_has_stable_storage, _environment_matches_target, _prepare_environment_replacement, _record_environment_lifetime, _record_environment_target, _retire_replaced_environment
    try:
        resolution = _target_resolution(target)
        config = (
            _get_env_config(dict(resolution.config))
            if resolution.named else _get_env_config()
        )
    except Exception as exc:  # best-effort bring-up
        logger.warning("Lazy environment target resolution failed: %s", exc)
        return None

    env_type = config["env_type"]
    if env_type == "local":
        return None

    raw_task_id = task_id or "default"
    base_task_id = _resolve_container_task_id(raw_task_id, config=config)
    effective_task_id = _environment_scope_key(base_task_id, resolution)
    raw_environment_key = _environment_scope_key(raw_task_id, resolution)
    backend_task_id = resolution.backend_task_id(base_task_id)

    def _find_existing():
        with _env_lock:
            for key in (effective_task_id, raw_environment_key):
                candidate = _active_environments.get(key)
                if _environment_matches_target(candidate, resolution):
                    _last_activity[key] = time.time()
                    return key, candidate
        return None, None

    _, existing = _find_existing()
    if existing is not None:
        return existing

    overrides = resolve_task_overrides(task_id, config=config)
    if env_type == "docker":
        image = overrides.get("docker_image") or config["docker_image"]
    elif env_type == "singularity":
        image = overrides.get("singularity_image") or config["singularity_image"]
    elif env_type == "modal":
        image = overrides.get("modal_image") or config["modal_image"]
    elif env_type == "daytona":
        image = overrides.get("daytona_image") or config["daytona_image"]
    else:
        image = ""

    cwd_override = (
        overrides.get("cwd")
        if (
            not resolution.named
            or (resolution.is_default and resolution.backend != "ssh")
        )
        else None
    )
    cwd = cwd_override or get_session_cwd(
        task_id, _resolution=resolution,
    ) or config["cwd"]
    cwd = _apply_task_cwd_override(config, cwd, cwd_override)
    host_cwd = _resolve_task_host_cwd(config, task_id)
    if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
        cwd = "/workspace" if host_cwd else config["cwd"]

    _start_cleanup_thread()
    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(
            effective_task_id, threading.Lock(),
        )

    with task_lock:
        _, existing = _find_existing()
        if existing is not None:
            return existing

        with _env_lock:
            stale_key = (
                effective_task_id
                if effective_task_id in _active_environments
                else raw_environment_key
            )
            stale_env = _active_environments.get(stale_key)
        try:
            _prepare_environment_replacement(
                stale_env,
                stale_key,
                target_name=resolution.target,
            )
        except _EnvironmentReplacementError as exc:
            logger.warning("Lazy environment replacement blocked: %s", exc)
            return None

        try:
            container_config, ssh_config, local_config = (
                _build_environment_constructor_configs(
                    config, resolution, base_task_id,
                )
            )
            from tools import terminal_tool_backends
            constructor = _create_environment if resolution.named else terminal_tool_backends._create_environment
            new_env = constructor(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                ssh_config=ssh_config,
                container_config=container_config,
                local_config=local_config,
                task_id=backend_task_id,
                host_cwd=host_cwd,
                session_scoped=_docker_environment_is_session_scoped(
                    config,
                    raw_task_id,
                    base_task_id,
                ),
            )
            _record_environment_lifetime(new_env, config)
            _record_environment_target(new_env, resolution)
        except Exception as exc:  # noqa: BLE001 — best-effort bring-up
            logger.warning(
                "Lazy %s environment init failed for task %s: %s",
                env_type, effective_task_id, exc,
            )
            return None

        publish_error = None
        if resolution.named:
            try:
                from tools.execution_targets import (
                    execution_target_config_is_frozen,
                    resolve_live_execution_target,
                )

                live_resolution = (
                    resolution
                    if execution_target_config_is_frozen()
                    else resolve_live_execution_target(target)
                )
                if live_resolution.security_scope != resolution.security_scope:
                    publish_error = (
                        f"Execution target {resolution.target!r} changed while "
                        "its environment was being created."
                    )
            except Exception as exc:
                publish_error = str(exc)

        if publish_error is not None:
            _cleanup_environment_resource(
                new_env,
                force_remove=True,
                preserve_storage=_environment_has_stable_storage(new_env),
            )
            logger.warning("Lazy environment publish failed: %s", publish_error)
            return None

        replaced_envs = []
        with _env_lock:
            current = _active_environments.get(effective_task_id)
            if current is not None and current is not new_env:
                replaced_envs.append((effective_task_id, current))
            if raw_environment_key != effective_task_id:
                raw_env = _active_environments.get(raw_environment_key)
                if (
                    raw_env is not None
                    and raw_env is not new_env
                    and not _environment_matches_target(raw_env, resolution)
                ):
                    _active_environments.pop(raw_environment_key, None)
                    _last_activity.pop(raw_environment_key, None)
                    replaced_envs.append((raw_environment_key, raw_env))
            _active_environments[effective_task_id] = new_env
            _last_activity[effective_task_id] = time.time()

        seen_replaced = set()
        for replaced_key, replaced_env in replaced_envs:
            if id(replaced_env) in seen_replaced:
                continue
            seen_replaced.add(id(replaced_env))
            _retire_replaced_environment(replaced_env, replaced_key)

        logger.info(
            "%s environment lazily initialized for task %s",
            env_type, effective_task_id,
        )
        return new_env


def is_persistent_env(task_id: str, target: Optional[str] = None) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.

    Session-scoped docker containers (per-session isolation mode) also count
    as persistent HERE: their lifetime is the SESSION, not the turn — they
    are removed by ``AIAgent.close()`` → ``cleanup_vm`` at session teardown
    and by the idle reaper, not per-turn.
    """
    env = get_active_env(task_id, target=target)
    if env is None:
        return False
    if getattr(env, "_session_scoped", False):
        return True
    return _environment_is_persistent(env)


def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    from tools.terminal_tool import _active_environments
    from tools.terminal_tool_target_lifecycle import _cleanup_retired_environments
    task_ids = list(_active_environments.keys())
    cleaned = 0

    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)

    cleaned += _cleanup_retired_environments(
        min_age_seconds=0.0, require_idle=False,
    )

    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)

    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(
    task_id: Hashable,
    *,
    force_remove: bool = False,
    preserve_persistent: bool = False,
    target: Optional[str] = None,
    include_collapsed: bool = False,
):
    """Manually clean up a specific environment by task_id.

    *force_remove* (default False) is forwarded to backends that accept it
    — currently only ``DockerEnvironment``. ``preserve_persistent`` is used
    by per-turn cleanup to keep each persistent named sibling live while
    removing only non-persistent targets. The default of False matches
    session-lifecycle semantics: this function is called from
    ``AIAgent.close()`` (TUI session close, gateway session teardown) and the
    per-turn cleanup branch for non-persistent envs, both of which should
    honor the user's persist-mode preference. Stopping the container here
    would defeat the "ONE long-lived container shared across sessions"
    contract — exactly the bug Ben reported when the container was killed
    on every TUI session close.

    Pass ``force_remove=True`` for actual user-initiated teardown
    (e.g. ``/reset``-style flows that haven't been wired yet, or future
    "destroy my sandbox" commands).

    The idle reaper passes the env through ``env.cleanup()`` directly (not
    via this function), so persist-mode idle envs are similarly no-op'd —
    only the orphan reaper at next startup reclaims them.
    """
    from tools.terminal_tool import _active_environments, _creation_locks, _creation_locks_lock, _env_lock, _last_activity, _resolve_container_task_id, _target_resolution
    from tools.terminal_tool_target_lifecycle import _cleanup_retired_environments
    # Direct tuple keys are used by global/idle cleanup. For a raw task,
    # omitted target cleans every target scope owned by that exact raw key;
    # an explicit target cleans exactly that scope. Do not collapse arbitrary
    # subagent ids to "default" here: legacy cleanup_vm(child_id) never tore
    # down the parent's shared environment, and doing so in named mode would
    # let a delegate's close race/disrupt its parent.
    if isinstance(task_id, tuple):
        keys = [task_id]
    elif target is None:
        try:
            resolution = _target_resolution(None)
            scoped_task_id = resolution.scope_task_key(task_id)
            collapsed_task_id = _resolve_container_task_id(
                str(task_id),
                config=resolution.config,
            )
            scoped_collapsed_task_id = resolution.scope_task_key(collapsed_task_id)
        except Exception:
            scoped_task_id = task_id
            collapsed_task_id = task_id
            scoped_collapsed_task_id = task_id
        matching_task_ids = {task_id, scoped_task_id}
        if include_collapsed:
            matching_task_ids.update({
                collapsed_task_id, scoped_collapsed_task_id,
            })
        with _env_lock:
            keys = [
                key for key in _active_environments
                if key in matching_task_ids
                or (
                    isinstance(key, tuple) and len(key) == 2
                    and key[0] in matching_task_ids
                )
            ]
        if not keys:
            keys = [task_id]
    else:
        resolution = _target_resolution(target)
        if resolution.named:
            keys = [resolution.environment_key(task_id)]
        else:
            keys = [task_id]

    active_process_keys = set()
    if preserve_persistent:
        try:
            from tools.process_registry import process_registry

            active_process_keys = {
                key for key in keys
                if process_registry.has_active_processes(key)
            }
        except Exception:
            logger.debug(
                "Failed to inspect active processes before cleanup",
                exc_info=True,
            )

    envs = []
    removed_keys = []
    with _env_lock:
        for key in keys:
            existing = _active_environments.get(key)
            if key in active_process_keys:
                continue
            if (
                preserve_persistent
                and existing is not None
                and _environment_is_persistent(existing)
            ):
                continue
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            removed_keys.append(key)
            if env is not None:
                envs.append((key, env))

    # Clean up per-task creation lock
    with _creation_locks_lock:
        for key in removed_keys:
            _creation_locks.pop(key, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        for key in removed_keys:
            clear_file_ops_cache(key)
    except ImportError:
        pass

    for key in keys:
        _cleanup_retired_environments(
            task_key=key,
            min_age_seconds=0.0,
            require_idle=preserve_persistent,
        )

    if not envs:
        return

    for key, env in envs:
        try:
            if hasattr(env, 'cleanup'):
                # Pass force_remove only if the env's cleanup() accepts it
                # (DockerEnvironment after issue #20561; other backends don't).
                import inspect
                sig = inspect.signature(env.cleanup)
                if "force_remove" in sig.parameters:
                    env.cleanup(force_remove=force_remove)
                else:
                    env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Manually cleaned up environment for task: %s", key)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", key)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", key, e)


def _evict_environment_for_task(
    task_id: Optional[str], target: Optional[str] = None,
) -> None:
    """Drop any cached environment for *task_id* (and its collapsed key).

    Used when a backend reports an infrastructure failure: keeping the dead
    env cached would make every subsequent call fail against a stale
    connection, defeating automatic recovery.
    """
    from tools.terminal_tool import _active_environments, _env_lock, _environment_scope_key, _get_env_config, _last_activity, _resolve_container_task_id, _target_resolution
    resolution = _target_resolution(target)
    config = (
        _get_env_config(dict(resolution.config))
        if resolution.named else _get_env_config()
    )
    raw_task_id = task_id or "default"
    base_task_id = _resolve_container_task_id(raw_task_id, config=config)
    keys = {
        _environment_scope_key(base_task_id, resolution),
        _environment_scope_key(raw_task_id, resolution),
    }
    evicted = []
    seen = set()
    with _env_lock:
        for key in keys:
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            if env is not None and id(env) not in seen:
                seen.add(id(env))
                evicted.append(env)
    for env in evicted:
        try:
            env.cleanup()
        except Exception:
            logger.debug("cleanup of degraded environment failed", exc_info=True)


def get_environment_for_target_scope(
    task_id: str, target: str, runtime_scope: str,
):
    """Find the active/retired environment that produced a scoped result."""
    from tools.terminal_tool import _active_environments, _env_lock, _profile_scoped_task_key, _resolve_container_task_id, _retired_environments, _retired_environments_lock, _target_resolution
    raw = task_id or "default"
    try:
        resolution = _target_resolution(target)
        collapsed = _resolve_container_task_id(
            raw,
            config=resolution.config,
        )
    except Exception:
        collapsed = _resolve_container_task_id(raw)
    bases = {
        _profile_scoped_task_key(raw),
        _profile_scoped_task_key(collapsed),
        _profile_scoped_task_key("default"),
    }

    def _matches(key: Hashable, env: Any) -> bool:
        belongs = key in bases or (
            isinstance(key, tuple) and len(key) == 2 and key[0] in bases
        )
        return bool(
            belongs
            and getattr(env, "_hermes_target_name", None) == target
            and getattr(env, "_hermes_target_scope", None) == runtime_scope
        )

    with _env_lock:
        for key, env in _active_environments.items():
            if _matches(key, env):
                return env
    with _retired_environments_lock:
        for key, env, _retired_at in _retired_environments:
            if _matches(key, env):
                return env
    return None
