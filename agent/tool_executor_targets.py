"""Target-affined checkpoints, output storage and context hints."""
from __future__ import annotations

_TARGET_RESULT_TOOLS = {
    "terminal", "read_file", "write_file", "patch", "search_files", "execute_code", "process",
}

def _ensure_file_checkpoint(agent, function_name: str, function_args: dict, effective_task_id: str) -> None:
    """Checkpoint the same workspace path that the file tool will mutate, resolved the way
    file tools do (against the task's live cwd, which differs from the process cwd in Docker)."""
    from agent.tool_executor import os
    file_path = function_args.get("path", "")
    if not file_path:
        return
    from agent.file_safety import is_nt_namespace_path
    from tools.execution_targets import resolve_execution_target
    from tools.file_tools_paths import _resolve_path_for_task, container_backend_for_task

    # The raw NT-namespace check precedes all path resolution (which can itself
    # initiate outbound SMB authentication on Windows).
    if is_nt_namespace_path(file_path):
        return
    resolution = resolve_execution_target(function_args.get("execution_target"))
    if resolution.backend != "local" or (
        not resolution.named and container_backend_for_task(effective_task_id or "default") is not None
    ):
        return  # Never checkpoint a container/remote path on the host.
    if resolution.named:
        from tools.terminal_tool import get_session_cwd
        cwd = (get_session_cwd(effective_task_id or "default", _resolution=resolution)
               or resolution.config.get("cwd"))
        # Match the file tool's tilde/MSYS conversion and host symlink
        # resolution, but anchor a relative path to this selected target.
        from tools.file_tools_paths import _anchor, _host_text
        resolved_path = _anchor(_host_text(file_path, False),
                                lambda: cwd or os.getcwd(), False)
    else:
        resolved_path = _resolve_path_for_task(file_path, effective_task_id or "default")
    agent._checkpoint_mgr.ensure_checkpoint(
        agent._checkpoint_mgr.get_working_dir_for_path(str(resolved_path)), f"before {function_name}",
    )


def _result_target_identity(tool_name: str, args: dict, result) -> tuple[str | None, str | None]:
    """Use the producer's target/scope, not the current mutable default alias."""
    from agent.tool_executor import json
    if tool_name not in _TARGET_RESULT_TOOLS:
        return None, None
    payload = None
    if isinstance(result, str) and result.lstrip().startswith("{"):
        try:
            payload = json.loads(result)
        except ValueError:
            pass
    if isinstance(payload, dict) and isinstance(payload.get("target"), str):
        return payload["target"], payload.get("runtime_scope")
    selector = args.get("execution_target")
    if isinstance(selector, str) and selector:
        return selector, None
    try:
        from tools.execution_targets import resolve_execution_target
        resolution = resolve_execution_target()
        if resolution.named:
            return resolution.target, resolution.security_scope
    except (ValueError, OSError):
        pass
    return None, None


def _result_environment(task_id: str, target: str | None, scope: str | None):
    """Never spill a named target's data to the default backend on a miss."""
    from agent.tool_executor import get_active_env
    if not target:
        return get_active_env(task_id)
    try:
        if scope:
            from tools.terminal_tool import get_environment_for_target_scope
            return get_environment_for_target_scope(task_id, target, scope)
        return get_active_env(task_id, target=target)
    except (ValueError, OSError):
        return None


def _active_env_for_tool_result(task_id: str, tool_name: str, args: dict, result=None):
    """Find the producer of a result, never a changed default target.

    Process output belongs to its background session even if a named target
    was later reconfigured; spilling to the host/default environment would
    cross the execution boundary.
    """
    from agent.tool_executor import _result_environment, _result_target_identity
    if tool_name == "process" and args.get("session_id"):
        try:
            from tools.process_registry import process_registry
            record = process_registry.get(args["session_id"])
            if record is not None and record.env_ref is not None:
                return record.env_ref
        except (ValueError, OSError):
            pass
        return None
    target, scope = _result_target_identity(tool_name, args, result)
    try:
        return _result_environment(task_id, target, scope)
    except (ValueError, OSError):
        return None


def _selected_local_target_cwd(
    task_id: str, tool_name: str, args: dict, *, runtime_scope: str | None = None,
) -> str | None:
    """Host cwd only when the producing target is still the same local runtime."""
    if tool_name not in _TARGET_RESULT_TOOLS:
        return None
    try:
        from tools.execution_targets import resolve_execution_target
        from tools.terminal_tool import get_session_cwd
        resolution = resolve_execution_target(args.get("execution_target"))
        if resolution.backend != "local" or (
            runtime_scope is not None and resolution.security_scope != runtime_scope
        ):
            return None
        return (get_session_cwd(task_id, _resolution=resolution)
                or resolution.config.get("cwd"))
    except (ValueError, OSError):
        return None


def _target_subdirectory_hints(
    agent, task_id: str, tool_name: str, args: dict,
    target: str | None, *, runtime_scope: str | None = None,
) -> str | None:
    """Inject context from the *actual local target*, never a remote host path."""
    from agent.tool_executor import Path, _selected_local_target_cwd
    if tool_name not in _TARGET_RESULT_TOOLS:
        tracker = getattr(agent, "_subdirectory_hints", None)
        return tracker.check_tool_call(tool_name, args) if tracker else None
    from tools.execution_targets import resolve_execution_target
    try:
        resolution = resolve_execution_target(target)
    except (ValueError, OSError):
        return None
    if resolution.backend != "local" or (
        runtime_scope is not None and runtime_scope != resolution.security_scope
    ):
        return None
    cwd = _selected_local_target_cwd(
        task_id, tool_name, {**args, "execution_target": resolution.target},
        runtime_scope=runtime_scope,
    )
    if not cwd:
        tracker = getattr(agent, "_subdirectory_hints", None)
        return (tracker.check_tool_call(tool_name, args)
                if not resolution.named and tracker else None)
    tracker = getattr(agent, "_subdirectory_hints", None)
    if tracker is None or not getattr(tracker, "enabled", True):
        return None
    if not resolution.named and str(tracker.working_dir) == str(Path(cwd).resolve()):
        return tracker.check_tool_call(tool_name, args)
    from agent.subdirectory_hints import SubdirectoryHintTracker
    trackers = getattr(agent, "_target_subdirectory_hint_trackers", None)
    if trackers is None:
        trackers = {}
        agent._target_subdirectory_hint_trackers = trackers
    key = (resolution.target, resolution.security_scope, str(Path(cwd).resolve()))
    if key not in trackers:
        trackers[key] = SubdirectoryHintTracker(
            cwd, enabled=getattr(tracker, "enabled", True),
        )
    return trackers[key].check_tool_call(tool_name, args)


def _persisted_target_hint(content, target: str | None, scope: str | None):
    from agent.tool_executor import PERSISTED_OUTPUT_TAG, _persisted_target_hint, json
    if isinstance(content, list):
        return [{**part, "text": _persisted_target_hint(part.get("text", ""), target, scope)}
                if isinstance(part, dict) and part.get("type") == "text" else part
                for part in content]
    if target and isinstance(content, str) and PERSISTED_OUTPUT_TAG in content and "Execution target for this saved output:" not in content:
        selector = f"execution_target={json.dumps(target)}"
        if scope:
            selector += f", runtime_scope={json.dumps(scope)}"
        return content + f"\nExecution target for this saved output: {json.dumps(target)}. Pass {selector} to `read_file`."
    return content
