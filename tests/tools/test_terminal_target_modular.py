"""Regression coverage for named targets on the modular terminal backend."""
import json

import pytest
from tools.terminal_tool_lifecycle import get_active_env


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    import tools.execution_targets as targets
    import tools.terminal_tool as tt
    targets.set_execution_target_config_source(None)
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    config = {"terminal": {"default_target": "alpha", "targets": {
        "alpha": {"backend": "local", "cwd": str(alpha)},
        "beta": {"backend": "local", "cwd": str(beta)},
    }}}
    monkeypatch.setattr(targets, "_load_merged_config", lambda: config)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})
    monkeypatch.setattr(tt, "_creation_locks", {})
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_session_cwd_specs", {})
    monkeypatch.setattr(tt, "_retired_environments", [])
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *_a, **_kw: {"approved": True})
    yield tt, targets, config, alpha, beta
    for env in tt._active_environments.values():
        env.cleanup()
    targets.set_execution_target_config_source(None)


def test_named_targets_choose_independent_cwd_cache_and_report_metadata(isolated):
    tt, targets, _config, alpha, beta = isolated
    first = json.loads(tt.terminal_tool("pwd", task_id="s"))
    second = json.loads(tt.terminal_tool("pwd", task_id="s", execution_target="beta"))
    assert first["output"] == str(alpha)
    assert second["output"] == str(beta)
    assert (first["target"], second["target"]) == ("alpha", "beta")
    assert first["backend"] == second["backend"] == "local"
    assert len(tt._active_environments) == 2
    assert tt.get_session_cwd("s", "alpha") == str(alpha)
    assert tt.get_session_cwd("s", "beta") == str(beta)


def test_repointed_alias_replaces_runtime_not_cwd_or_environment(isolated):
    tt, _targets, config, alpha, beta = isolated
    first = json.loads(tt.terminal_tool("pwd", task_id="s"))
    previous = get_active_env("s", "alpha")
    config["terminal"]["targets"]["alpha"]["cwd"] = str(beta)
    second = json.loads(tt.terminal_tool("pwd", task_id="s"))
    assert first["output"] == str(alpha)
    assert second["output"] == str(beta)
    assert first["runtime_scope"] != second["runtime_scope"]
    assert get_active_env("s", "alpha") is not previous
    assert tt.get_session_cwd("s", "alpha") == str(beta)


def test_unknown_target_fails_closed_without_falling_back(isolated):
    tt, *_ = isolated
    result = json.loads(tt.terminal_tool("pwd", task_id="s", execution_target="missing"))
    assert result["status"] == "error"
    assert "Unknown execution target" in result["error"]
    assert not tt._active_environments


def test_target_guard_sees_actual_routing_identity(monkeypatch, isolated):
    tt, *_ = isolated
    seen = []
    def deny(_command, _backend, **kwargs):
        seen.append(kwargs)
        return {"approved": False, "message": "denied"}
    monkeypatch.setattr(tt, "_check_all_guards", deny)
    result = json.loads(tt.terminal_tool("pwd", task_id="s", execution_target="beta"))
    assert result["status"] == "blocked"
    assert seen[0]["execution_target"] == "beta"
    assert seen[0]["execution_target_named"] is True
    assert seen[0]["execution_target_scope"]


def test_legacy_flat_target_keeps_string_environment_key(monkeypatch, isolated):
    tt, targets, _config, *_ = isolated
    monkeypatch.setattr(targets, "_load_merged_config", lambda: {"terminal": {"backend": "local"}})
    result = json.loads(tt.terminal_tool("pwd", task_id="s"))
    assert result["exit_code"] == 0
    assert "default" in tt._active_environments
    assert not isinstance(next(iter(tt._active_environments)), tuple)


def test_named_background_spawn_carries_target_generation(isolated):
    from tools.process_registry import process_registry
    tt, targets, _config, *_ = isolated
    spawned = json.loads(tt.terminal_tool(
        "sleep 0.1", task_id="s", execution_target="beta", background=True,
    ))
    assert spawned["error"] is None
    assert spawned["target"] == "beta"
    assert spawned["runtime_scope"] == targets.resolve_execution_target("beta").security_scope
    session = process_registry.get(spawned["session_id"])
    assert session.target == "beta"
    assert session.runtime_scope == spawned["runtime_scope"]
    assert session.environment_task_key == ("default", "beta")
    finished = process_registry.wait(spawned["session_id"], timeout=5)
    assert finished["exit_code"] == 0


def test_named_sudo_cache_never_reads_another_target_or_generation(isolated):
    from tools.terminal_tool_sudo import (
        _reset_cached_sudo_passwords, _get_cached_sudo_password,
        _scoped_sudo_execution, _set_cached_sudo_password,
        _transform_sudo_command,
    )
    _reset_cached_sudo_passwords()
    with _scoped_sudo_execution("alpha", "local", named=True, target_scope="v1"):
        _set_cached_sudo_password("alpha-only")
    with _scoped_sudo_execution("beta", "local", named=True, target_scope="v1"):
        assert _get_cached_sudo_password() == ""
    with _scoped_sudo_execution("alpha", "local", named=True, target_scope="v2"):
        assert _get_cached_sudo_password() == ""
    with _scoped_sudo_execution("alpha", "local", named=True, target_scope="v1"):
        assert _get_cached_sudo_password() == "alpha-only"
        command, stdin = _transform_sudo_command("sudo id")
        assert "alpha-only" not in command
        assert stdin == "alpha-only\n"
    _reset_cached_sudo_passwords()
