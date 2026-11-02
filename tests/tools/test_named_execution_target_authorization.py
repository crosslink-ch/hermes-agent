"""Behavior contracts split by scenario."""
from __future__ import annotations
import json
from types import SimpleNamespace
from tools import approval_context
from tests.tools.named_target_fixtures import _named_config

pytest_plugins = ["tests.tools.named_target_fixtures"]

def test_model_facing_handlers_reject_removed_target_selector(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha)}),
    )

    terminal_result = json.loads(terminal_mod._handle_terminal({
        "command": "pwd", "target": "alpha",
    }, task_id="removed-selector"))
    write_result = json.loads(file_mod._handle_write_file({
        "path": "alias.txt", "content": "data\n", "target": "alpha",
    }, task_id="removed-selector"))

    assert "does not accept 'target'" in terminal_result["error"]
    assert "does not accept 'target'" in write_result["error"]
    assert not (alpha / "alias.txt").exists()


def test_sudo_cache_and_nopasswd_probe_are_target_scoped(
    monkeypatch, isolated_target_state,
):
    from tools import terminal_tool_sudo as sudo_mod

    sudo_mod._reset_cached_sudo_passwords()
    sudo_mod._set_cached_sudo_password("local-secret", "local", "local")
    assert sudo_mod._get_cached_sudo_password("local", "local") == "local-secret"
    assert sudo_mod._get_cached_sudo_password("devbox", "ssh") == ""

    # Current backends supply their own sudo -n probe, never a host-global
    # subprocess probe. A selected target must use only its supplied callback.
    sudo_mod._reset_cached_sudo_passwords()
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(sudo_mod, "_in_delegated_child_context", lambda: False)
    calls = []
    with sudo_mod._scoped_sudo_execution("devbox", "ssh"):
        command, password = sudo_mod._transform_sudo_command(
            "sudo id", sudo_nopasswd_check=lambda: calls.append("devbox") or True,
        )
        assert (command, password) == ("sudo id", None)
    with sudo_mod._scoped_sudo_execution("local", "local"):
        command, password = sudo_mod._transform_sudo_command(
            "sudo id", sudo_nopasswd_check=lambda: calls.append("local") or True,
        )
        assert (command, password) == ("sudo id", None)
    assert calls == ["devbox", "local"]

    monkeypatch.setenv("SUDO_PASSWORD", "default-secret")
    with sudo_mod._scoped_sudo_execution(
        "devbox", "ssh", named=True, sudo_password="target-secret",
    ):
        transformed, selected_password = sudo_mod._transform_sudo_command("sudo id")
    assert "target-secret" not in transformed
    assert "default-secret" not in transformed
    assert selected_password == "target-secret\n"

    monkeypatch.delenv("SUDO_PASSWORD")
    with sudo_mod._scoped_sudo_execution("devbox", "ssh", named=True):
        sudo_mod._set_cached_sudo_password("cached-target")
        _, cached_password = sudo_mod._transform_sudo_command("sudo id")
    assert cached_password == "cached-target\n"

    with sudo_mod._scoped_sudo_execution(
        "devbox", "ssh", named=True, target_scope="config-v2",
    ):
        sudo_mod._set_cached_sudo_password("stale-secret")
    assert sudo_mod._invalidate_cached_sudo_on_auth_failure(
        "sudo id", "sudo: authentication failed", "devbox", "ssh", "config-v2",
    ) is True
    assert sudo_mod._get_cached_sudo_password("devbox", "ssh", "config-v2") == ""


def test_command_approval_payload_and_observer_include_target_metadata(monkeypatch):
    from tools import approval as approval_mod

    first = approval_mod._execution_scoped_pattern_key(
        "danger", "devbox", True, "scope-one",
    )
    second = approval_mod._execution_scoped_pattern_key(
        "danger", "devbox", True, "scope-two",
    )
    assert first != second

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_mod, "detect_hardline_command", lambda command: (False, None))
    monkeypatch.setattr(approval_mod, "_check_sudo_stdin_guard", lambda command: (False, None))
    monkeypatch.setattr(approval_mod, "_match_user_deny_rule", lambda command: None)
    monkeypatch.setattr(approval_mod, "_command_matches_permanent_allowlist", lambda command: False)
    monkeypatch.setattr(
        approval_mod, "detect_dangerous_command",
        lambda command: (True, "danger", "dangerous command"),
    )
    session_key = "approval-target-session"
    token = approval_context.set_current_session_key(session_key)
    seen = {}
    hooks = []

    def notify(data):
        seen.update(data)
        with approval_mod._lock:
            entry = approval_mod._gateway_queues[session_key][-1]
            entry.result = "deny"
            entry.event.set()

    monkeypatch.setattr(approval_context, "_fire_approval_hook", lambda name, **data: hooks.append((name, data)))
    approval_mod.register_gateway_notify(session_key, notify)
    try:
        approval_mod.check_all_command_guards(
            "danger", "local", execution_target="alpha", execution_backend="local",
        )
    finally:
        approval_mod.unregister_gateway_notify(session_key)
        approval_context.reset_current_session_key(token)

    assert seen["target"] == "alpha"
    assert seen["backend"] == "local"
    assert "alpha" in seen["description"] and "local" in seen["description"]
    pre_hook = next(data for name, data in hooks if name == "pre_approval_request")
    assert pre_hook["target"] == "alpha"
    assert pre_hook["backend"] == "local"


def test_execute_code_approval_payload_includes_target_metadata(monkeypatch):
    from tools import approval as approval_mod

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval_mod, "is_approved", lambda *args: False)
    monkeypatch.setattr(approval_mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(
        approval_mod, "is_current_session_yolo_enabled", lambda: False,
    )
    seen = {}
    monkeypatch.setattr(
        approval_mod, "_await_gateway_decision",
        lambda session_key, notify_cb, approval_data, surface: (
            seen.update(approval_data) or {"resolved": True, "choice": "once"}
        ),
    )
    session_token = approval_context.set_current_session_key("target-approval")
    try:
        with approval_mod._lock:
            approval_mod._gateway_notify_cbs["target-approval"] = lambda data: None
        result = approval_mod.check_execute_code_guard(
            "print('ok')", "ssh", execution_target="devbox",
            execution_backend="ssh",
        )
    finally:
        approval_context.reset_current_session_key(session_token)
        with approval_mod._lock:
            approval_mod._gateway_notify_cbs.pop("target-approval", None)

    assert result["approved"] is True
    assert seen["target"] == "devbox"
    assert seen["backend"] == "ssh"
    assert "devbox" in seen["description"]


def test_gateway_script_guard_reads_selected_named_target_cwd(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "restart.sh").write_text("#!/bin/sh\necho safe\n")
    (beta / "restart.sh").write_text("#!/bin/sh\nhermes gateway restart\n")
    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process", lambda: True,
    )

    result = json.loads(terminal_mod.terminal_tool(
        "bash restart.sh", task_id="gateway-guard", execution_target="beta",
    ))

    assert result["status"] == "error"
    assert "cannot restart, stop, or uninstall the gateway" in result["error"]


def test_checkpoint_alias_flip_pins_dispatch_generation(monkeypatch, tmp_path):
    from agent import tool_executor
    import tools.execution_targets as targets_mod
    from tools.file_tools import read_file_tool, write_file_tool

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "sample.txt").write_text("before")
    config_a = _named_config({"dev": str(first)}, default="dev")
    config_b = _named_config({"dev": str(second)}, default="dev")
    targets_mod.set_execution_target_config_source(config_a)
    # Current main requires a full read before overwriting an existing file.
    # Establish the baseline before the checkpoint deliberately changes the alias.
    initial = read_file_tool("sample.txt", task_id="target-race", execution_target="dev")
    assert "before" in initial
    checkpoints = []

    class CheckpointManager:
        enabled = True

        @staticmethod
        def get_working_dir_for_path(path):
            return str(first)

        def ensure_checkpoint(self, cwd, reason):
            checkpoints.append((cwd, reason))
            targets_mod.set_execution_target_config_source(config_b)

    class Guardrails:
        @staticmethod
        def before_call(_name, _args):
            return SimpleNamespace(allows_execution=True)

    agent = SimpleNamespace(
        session_id="session",
        _current_turn_id="turn",
        _current_api_request_id="request",
        _tool_guardrails=Guardrails(),
        _guardrail_block_result=lambda decision: decision,
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=CheckpointManager(),
        _touch_activity=lambda *_args, **_kwargs: None,
    )

    managed = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name="write_file",
        function_args={
            "path": "sample.txt",
            "content": "data",
            "execution_target": "dev",
        },
        effective_task_id="target-race",
        tool_call_id="call-1",
        execute=lambda args: write_file_tool(
            args["path"], args["content"],
            task_id="target-race",
            execution_target=args["execution_target"],
        ),
    )

    result = json.loads(managed.result)
    assert checkpoints[0][0] == str(first)
    assert result["cwd"] == str(first)
    assert result["target"] == "dev"
    assert (first / "sample.txt").read_text() == "data"
    assert not (second / "sample.txt").exists()
    assert targets_mod.resolve_execution_target("dev").config["cwd"] == str(second)
    targets_mod.set_execution_target_config_source(None)


def test_guardrail_alias_flip_pins_authorized_dispatch_generation(monkeypatch, tmp_path):
    from agent import tool_executor
    import tools.execution_targets as targets_mod
    from tools.file_tools import write_file_tool

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    config_a = _named_config({"dev": str(first)}, default="dev")
    config_b = _named_config({"dev": str(second)}, default="dev")
    targets_mod.set_execution_target_config_source(config_a)

    class Guardrails:
        @staticmethod
        def before_call(_name, _args):
            targets_mod.set_execution_target_config_source(config_b)
            return SimpleNamespace(allows_execution=True)

    agent = SimpleNamespace(
        session_id="session",
        _current_turn_id="turn",
        _current_api_request_id="request",
        _tool_guardrails=Guardrails(),
        _guardrail_block_result=lambda decision: decision,
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _touch_activity=lambda *_args, **_kwargs: None,
    )

    managed = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name="write_file",
        function_args={
            "path": "sample.txt",
            "content": "authorized-generation",
            "execution_target": "dev",
        },
        effective_task_id="target-race",
        tool_call_id="call-guardrail",
        execute=lambda args: write_file_tool(
            args["path"], args["content"],
            task_id="target-race",
            execution_target=args["execution_target"],
        ),
    )

    result = json.loads(managed.result)
    assert result["cwd"] == str(first)
    assert (first / "sample.txt").read_text() == "authorized-generation"
    assert not (second / "sample.txt").exists()
    assert targets_mod.resolve_execution_target("dev").config["cwd"] == str(second)
    targets_mod.set_execution_target_config_source(None)


def test_pre_tool_hook_selector_rewrite_is_revalidated_before_authorization(monkeypatch):
    from agent import tool_executor

    guardrail_calls = []
    dispatch_calls = []

    class Guardrails:
        @staticmethod
        def before_call(name, args):
            guardrail_calls.append((name, dict(args)))
            return SimpleNamespace(allows_execution=True)

    monkeypatch.setattr(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        lambda _name, args, **_kwargs: (
            None,
            {**args, "target": "content"},
        ),
    )
    agent = SimpleNamespace(
        session_id="session",
        _current_turn_id="turn",
        _current_api_request_id="request",
        _tool_guardrails=Guardrails(),
        _guardrail_block_result=lambda decision: decision,
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=False),
        _touch_activity=lambda *_args, **_kwargs: None,
    )

    managed = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name="write_file",
        function_args={"path": "sample.txt", "content": "data"},
        effective_task_id="selector-rewrite",
        tool_call_id="call-selector-rewrite",
        execute=lambda args: dispatch_calls.append(dict(args)) or "dispatched",
    )

    assert managed.blocked is True
    assert "does not accept 'target'" in str(managed.result)
    assert guardrail_calls == []
    assert dispatch_calls == []


def test_removed_target_selector_blocks_before_agent_authorization(monkeypatch):
    from agent import tool_executor

    class Guardrails:
        @staticmethod
        def before_call(_name, _args):
            raise AssertionError("guardrails must not see an invalid selector")

    agent = SimpleNamespace(
        session_id="session",
        _current_turn_id="turn",
        _current_api_request_id="request",
        _tool_guardrails=Guardrails(),
        quiet_mode=True,
        tool_progress_mode="off",
        verbose_logging=False,
        tool_progress_callback=None,
        tool_start_callback=None,
        _checkpoint_mgr=SimpleNamespace(enabled=True),
        _touch_activity=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.resolve_pre_tool_block",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pre-tool hooks must not see an invalid selector")
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "_begin_tool_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("progress/checkpoint preflight must not start")
        ),
    )
    execute_calls = []

    managed = tool_executor._run_agent_tool_execution_middleware(
        agent,
        function_name="write_file",
        function_args={
            "path": "private.txt",
            "content": "private",
            "target": "alpha",
        },
        effective_task_id="invalid-selector",
        tool_call_id="call-invalid-selector",
        execute=lambda args: execute_calls.append(args),
    )

    assert "does not accept 'target'" in json.loads(managed.result)["error"]
    assert managed.blocked is True
    assert execute_calls == []

