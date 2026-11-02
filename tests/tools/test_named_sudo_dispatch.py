"""Terminal dispatch must bind credentials to the approved target, not just name its result."""
import json

import pytest


@pytest.mark.parametrize("background", [False, True])
def test_terminal_dispatch_binds_sudo_to_selected_target(monkeypatch, background):
    from tools import terminal_tool as terminal, terminal_tool_sudo as sudo
    from tools.execution_targets import execution_target_config_scope, resolve_execution_target
    from types import SimpleNamespace
    config = {"terminal": {"default_target": "alpha", "targets": {
        "alpha": {"backend": "local", "cwd": "/workspace", "sudo_password": "fixture-alpha"},
        "beta": {"backend": "local", "cwd": "/workspace", "sudo_password": "fixture-beta"}}}}
    monkeypatch.setattr(terminal, "_active_environments", {})
    monkeypatch.setattr(terminal, "_last_activity", {})
    monkeypatch.setattr(terminal, "_creation_locks", {})
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal, "_check_all_guards", lambda *a, **kw: {"approved": True})
    monkeypatch.setattr(terminal, "_create_environment", lambda **kw: SimpleNamespace(cwd=kw['cwd'],env={}))
    observed=[]
    def dispatch(*a, **kw):
        observed.append(sudo._sudo_execution_context.get())
        return json.dumps({"output": "done", "exit_code": 0})
    monkeypatch.setattr(terminal, "_run_foreground", dispatch)
    monkeypatch.setattr(terminal, "spawn_background_process", dispatch)
    with execution_target_config_scope(config):
        expected = resolve_execution_target("beta")
        result=json.loads(terminal.terminal_tool("printf harmless", task_id="owner", execution_target="beta",background=background))
    assert result["exit_code"] == 0
    assert observed == [("beta", "local", True, "fixture-beta", expected.security_scope)]
    assert sudo._sudo_execution_context.get() is None
