"""Behavior contracts split by scenario."""
from __future__ import annotations
import json
import pytest
from tests.tools.named_target_fixtures import _named_config

pytest_plugins = ["tests.tools.named_target_fixtures"]

def test_execute_code_inherits_target_and_rejects_nested_override(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )
    source = code_mod.generate_hermes_tools_module(
        ["write_file", "read_file", "search_files"],
    )
    namespace = {}
    exec(source, namespace)
    calls = []
    namespace["_call"] = lambda name, args: calls.append((name, args)) or {}

    namespace["write_file"]("nested.txt", "alpha-default\n")
    namespace["write_file"](
        "nested.txt", "beta-explicit\n", execution_target="beta",
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'target'"):
        namespace["write_file"]("nested.txt", "invalid\n", target="beta")
    namespace["search_files"]("needle")

    inherited_write = code_mod._inherit_execution_target(
        calls[0][0], calls[0][1], "alpha",
    )
    inherited_search = code_mod._inherit_execution_target(
        calls[2][0], calls[2][1], "alpha",
    )
    assert inherited_write["execution_target"] == "alpha"
    assert inherited_search["execution_target"] == "alpha"
    with pytest.raises(ValueError, match="cannot select 'beta'"):
        code_mod._inherit_execution_target(calls[1][0], calls[1][1], "alpha")
    with pytest.raises(ValueError, match="does not accept 'target'"):
        code_mod._inherit_execution_target(
            "write_file", {"path": "x", "target": "beta"}, "alpha",
        )

    remote_config = _named_config({"alpha": str(alpha), "beta": str(beta)})
    remote_config["terminal"]["targets"]["alpha"]["backend"] = "ssh"
    remote_config["terminal"]["targets"]["alpha"]["ssh_host"] = "example.invalid"
    remote_config["terminal"]["targets"]["alpha"]["ssh_user"] = "agent"
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: remote_config)
    forwarded = {}
    monkeypatch.setattr(code_mod, "_get_execution_mode", lambda: "project")
    monkeypatch.setattr(
        code_mod, "_execute_remote",
        lambda code, task_id, enabled_tools, reset=False, target=None, mode="strict",
        expected_target_scope=None, expected_target_config=None: (
            forwarded.update(
                target=target, mode=mode, scope=expected_target_scope,
                config=expected_target_config,
            )
            or json.dumps({"status": "success"})
        ),
    )

    result = json.loads(code_mod.execute_code(
        "print('ok')", execution_target="alpha",
    ))
    assert result["status"] == "success"
    assert forwarded["target"] == "alpha"
    assert forwarded["mode"] == "project"
    assert forwarded["scope"] == targets_mod.resolve_execution_target("alpha").security_scope
    assert forwarded["config"]["terminal"]["targets"]["alpha"]["backend"] == "ssh"
    with pytest.raises(ValueError, match="runtime_scope"):
        code_mod._inherit_execution_target(
            "read_file",
            {"path": "saved.txt", "runtime_scope": "old-scope"},
            "alpha",
            forwarded["scope"],
        )


def test_execute_code_rpc_rejects_alias_repointed_during_run(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import code_execution_tool as code_mod

    config = {
        "terminal": {
            "default_target": "alpha",
            "targets": {"alpha": {"backend": "local", "cwd": "/old"}},
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    approved_scope = targets_mod.resolve_execution_target("alpha").security_scope
    config["terminal"]["targets"]["alpha"] = {
        "backend": "ssh", "ssh_host": "new.example", "ssh_user": "agent",
        "cwd": "/new",
    }

    with pytest.raises(ValueError, match="changed while execute_code was running"):
        code_mod._inherit_execution_target(
            "write_file", {"path": "marker", "content": "x"},
            "alpha", approved_scope,
        )


def test_execute_code_rpc_live_recheck_ignores_frozen_dispatch_snapshot(
    isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import code_execution_tool as code_mod

    approved = {
        "terminal": {
            "default_target": "alpha",
            "targets": {"alpha": {"backend": "local", "cwd": "/old"}},
        },
    }
    live = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {
                    "backend": "ssh",
                    "ssh_host": "new.example",
                    "ssh_user": "agent",
                    "cwd": "/new",
                },
            },
        },
    }
    approved_scope = targets_mod.resolve_execution_target(
        "alpha", config=approved,
    ).security_scope
    targets_mod.set_execution_target_config_source(live)

    with targets_mod.execution_target_config_scope(approved):
        with pytest.raises(
            ValueError, match="changed while execute_code was running",
        ):
            code_mod._inherit_execution_target(
                "write_file", {"path": "marker", "content": "x"},
                "alpha", approved_scope,
            )


def test_execute_code_rpc_dispatch_uses_frozen_approved_target_config(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import code_execution_tool as code_mod

    live = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {
                    "backend": "ssh", "ssh_host": "new.example",
                    "ssh_user": "agent",
                },
            },
        },
    }
    approved = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {"backend": "local", "cwd": "/approved"},
            },
        },
    }

    targets_mod.set_execution_target_config_source(live)

    def handler(_name, args, task_id=None):
        resolved = targets_mod.resolve_execution_target(args["execution_target"])
        current = targets_mod.resolve_live_execution_target(args["execution_target"])
        return resolved.backend, resolved.config.get("cwd"), current.backend

    assert code_mod._dispatch_rpc_tool(
        handler, "write_file", {"execution_target": "alpha"}, "task", approved,
    ) == ("local", "/approved", "ssh")
    assert targets_mod.resolve_execution_target("alpha").backend == "ssh"


def test_execute_code_routes_real_nested_file_call_to_selected_local_target(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    config = _named_config({"alpha": str(alpha), "beta": str(beta)})
    # Real merged configs contain defaults for unrelated backends.  Freezing the
    # approved target must preserve them as inherited top-level settings rather
    # than reclassifying them as explicit local-target fields.
    config["terminal"]["modal_mode"] = "ephemeral"
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    targets_mod.set_execution_target_config_source(config)

    result = json.loads(code_mod.execute_code(
        "from hermes_tools import write_file\n"
        "print(write_file('from-code.txt', 'beta-via-rpc\\n'))\n",
        task_id="execute-target-session",
        enabled_tools=["write_file"],
        execution_target="beta",
    ))

    assert result["status"] == "success"
    assert result["target"] == "beta"
    assert result["backend"] == "local"
    assert (beta / "from-code.txt").read_text(encoding="utf-8") == "beta-via-rpc\n"
    assert not (alpha / "from-code.txt").exists()


def test_remote_project_mode_executes_from_target_session_cwd(
    monkeypatch, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    config = _named_config({"alpha": "/srv/configured", "beta": "/b"})
    config["terminal"]["targets"]["alpha"].update({
        "backend": "ssh",
        "ssh_host": "example.invalid",
        "ssh_user": "agent",
    })
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    terminal_mod, _ = isolated_target_state
    terminal_mod.record_session_cwd(
        "remote-project", "/srv/session-project", target="alpha",
    )

    calls = []

    class FakeEnv:
        cwd = "/srv/configured"

        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            calls.append((command, cwd))
            if cwd is not None:
                self.cwd = cwd
            if "command -v python3" in command:
                return {"output": "OK\n", "returncode": 0}
            return {"output": "done\n", "returncode": 0}

        def get_temp_dir(self):
            return "/remote-tmp"

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    env = FakeEnv()
    monkeypatch.setattr(
        code_mod, "_get_or_create_env",
        lambda task_id, target=None, expected_target_scope=None: (env, "ssh"),
    )
    monkeypatch.setattr(code_mod, "_ship_file_to_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_mod.threading, "Thread", DummyThread)
    monkeypatch.setattr(
        code_mod, "_load_config", lambda: {"timeout": 30, "max_tool_calls": 5},
    )

    result = json.loads(code_mod._execute_remote(
        "print('ok')", "remote-project", [], target="alpha", mode="project",
    ))
    repeated = json.loads(code_mod._execute_remote(
        "print('again')", "remote-project", [], target="alpha", mode="project",
    ))

    assert result["status"] == repeated["status"] == "success"
    assert result["cwd"] == repeated["cwd"] == "/srv/session-project"
    script_calls = [
        item for item in calls
        if "python3 /remote-tmp/hermes_exec_" in item[0]
    ]
    assert len(script_calls) == 2
    assert all(cwd == "/srv/session-project" for _, cwd in script_calls)
