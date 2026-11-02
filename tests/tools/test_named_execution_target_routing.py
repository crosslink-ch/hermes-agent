from __future__ import annotations

import json
from types import SimpleNamespace
import threading
import time

import pytest
import subprocess
from tools import terminal_tool_lifecycle as terminal_lifecycle
from tools import terminal_tool_target_lifecycle as target_lifecycle
from tools import file_tools_read_tracking as read_tracking
from tools import approval_context

from tests.tools.named_target_fixtures import _named_config

pytest_plugins = ["tests.tools.named_target_fixtures"]


def test_named_docker_target_uses_selected_config_for_terminal_key(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    config = {
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": "/tmp"},
                "sandbox": {
                    "backend": "docker",
                    "cwd": "/workspace",
                    "container_persistent": False,
                    "docker_mount_cwd_to_workspace": False,
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    created = []

    class FakeEnvironment:
        backend = "docker"
        cwd = "/workspace"
        _persistent = True

        def execute(self, _command, **_kwargs):
            return {"output": "ok\n", "returncode": 0}

        def cleanup(self, force_remove=False):
            del force_remove

        def wait_for_cleanup(self, timeout):
            del timeout
            return True

    def fake_create(**kwargs):
        created.append(dict(kwargs))
        return FakeEnvironment()

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    result = json.loads(terminal_mod.terminal_tool(
        "pwd",
        task_id="named-session",
        execution_target="sandbox",
    ))
    resolution = targets_mod.resolve_execution_target("sandbox")

    assert result["exit_code"] == 0
    assert created[0]["task_id"] == resolution.backend_task_id("named-session")
    assert created[0]["session_scoped"] is True


@pytest.mark.parametrize("builder", ["file", "code"])
def test_named_docker_target_uses_selected_config_for_nonterminal_builder(
    builder, monkeypatch, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": "/tmp"},
                "sandbox": {
                    "backend": "docker",
                    "cwd": "/workspace",
                    "container_persistent": False,
                    "docker_mount_cwd_to_workspace": False,
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    created = []

    class FakeEnvironment:
        backend = "docker"
        cwd = "/workspace"
        _persistent = True

        def cleanup(self, force_remove=False):
            del force_remove

        def wait_for_cleanup(self, timeout):
            del timeout
            return True

    def fake_create(**kwargs):
        created.append(dict(kwargs))
        return FakeEnvironment()

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    if builder == "file":
        file_mod._get_file_ops("named-session", "sandbox")
    else:
        code_mod._get_or_create_env("named-session", "sandbox")
    resolution = targets_mod.resolve_execution_target("sandbox")

    assert created[0]["task_id"] == resolution.backend_task_id("named-session")
    assert created[0]["session_scoped"] is True


def test_same_task_reuses_within_target_and_isolates_across_targets(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    first = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-a"))
    second = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-b", execution_target="alpha"))
    third = json.loads(terminal_mod.terminal_tool("pwd", task_id="child-a", execution_target="beta"))

    assert first["output"] == second["output"] == str(alpha)
    assert third["output"] == str(beta)
    assert set(terminal_mod._active_environments) == {("default", "alpha"), ("default", "beta")}


def test_config_change_replaces_environment_instead_of_relabeling_it(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "devbox",
            "targets": {
                "devbox": {"backend": "local", "cwd": "/local/project"},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    created = []
    executed_cwds = []
    cleanup_calls = []
    cleanup_waits = []

    class FakeEnvironment:
        def __init__(self, backend):
            self.backend = backend
            self._persistent = True
            self.cwd = "/remote/project" if backend == "ssh" else "/local/project"

        def execute(self, command, **kwargs):
            executed_cwds.append(kwargs.get("cwd"))
            return {"output": f"executed-on-{self.backend}\n", "returncode": 0}

        def cleanup(self, force_remove=False):
            cleanup_calls.append((self.backend, force_remove, self._persistent))

        def wait_for_cleanup(self, timeout):
            cleanup_waits.append(timeout)
            return True

    def fake_create(**kwargs):
        env = FakeEnvironment(kwargs["env_type"])
        created.append(env)
        return env

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    first = json.loads(terminal_mod.terminal_tool("pwd", task_id="session"))
    first_file_ops = file_mod._get_file_ops("session", "devbox")
    terminal_mod.record_session_cwd(
        "session", "/local/old-working-tree", target="devbox",
    )
    config["terminal"]["targets"]["devbox"] = {
        "backend": "ssh",
        "cwd": "/remote/project",
        "ssh_host": "devbox.example",
        "ssh_user": "agent",
    }
    second = json.loads(terminal_mod.terminal_tool("pwd", task_id="session"))
    second_file_ops = file_mod._get_file_ops("session", "devbox")

    assert first["output"].strip() == "executed-on-local"
    assert first["backend"] == "local"
    assert second["output"].strip() == "executed-on-ssh"
    assert second["backend"] == "ssh"
    assert [env.backend for env in created] == ["local", "ssh"]
    assert first_file_ops.env is created[0]
    assert second_file_ops.env is created[1]
    assert executed_cwds == ["/local/project", "/remote/project"]
    assert len(terminal_mod._retired_environments) == 1
    assert terminal_mod._retired_environments[0][1] is created[0]
    assert terminal_mod._cleanup_retired_environments(
        min_age_seconds=0.0, require_idle=False,
    ) == 1
    assert cleanup_calls == [("local", True, False)]
    assert cleanup_waits == [60.0]
    assert terminal_mod._retired_environments == []


def test_persistent_storage_replacement_waits_for_active_runtime(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state
    env = SimpleNamespace(_hermes_stable_storage=True)
    monkeypatch.setattr(terminal_mod, "_active_turn_counts", {"session": 2})

    assert terminal_mod._environment_replacement_is_busy(
        env, ("session", "devbox"),
    ) is True


@pytest.mark.parametrize("builder", ["terminal", "file", "code"])
def test_environment_creation_fails_closed_if_target_changes_before_publish(
    builder, monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import code_execution_tool as code_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "alpha",
            "targets": {"alpha": {"backend": "local", "cwd": "/old"}},
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    cleaned = []

    class FakeEnvironment:
        cwd = "/old"

        def cleanup(self, force_remove=False):
            cleaned.append(force_remove)

        def wait_for_cleanup(self, timeout):
            return True

    def fake_create(**kwargs):
        config["terminal"]["targets"]["alpha"] = {
            "backend": "local", "cwd": "/new",
        }
        return FakeEnvironment()

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    if builder == "terminal":
        result = json.loads(terminal_mod.terminal_tool(
            "pwd", task_id="publish-race", execution_target="alpha",
        ))
        assert "changed while its environment was being created" in result["error"]
    elif builder == "file":
        with pytest.raises(RuntimeError, match="changed while its environment"):
            file_mod._get_file_ops("publish-race", "alpha")
    else:
        with pytest.raises(ValueError, match="changed while its environment"):
            code_mod._get_or_create_env("publish-race", "alpha")

    assert terminal_mod._active_environments == {}
    assert cleaned == [True]


def test_ssh_target_routes_terminal_and_file_adapter_to_same_environment(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "backend": "local",
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": "/workspace/local"},
                "devbox": {
                    "backend": "ssh",
                    "cwd": "/srv/project",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    created = []

    class FakeSshEnvironment:
        def __init__(self, cwd):
            self.cwd = cwd
            self.is_persistent = True

        def execute(self, command, cwd="", timeout=None, **kwargs):
            return {"output": self.cwd + "\n", "returncode": 0}

        def cleanup(self):
            pass

    def fake_create(**kwargs):
        created.append(kwargs)
        return FakeSshEnvironment(kwargs["cwd"])

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    result = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", execution_target="devbox",
    ))
    file_ops = file_mod._get_file_ops("session", "devbox")

    assert result["target"] == "devbox"
    assert result["backend"] == "ssh"
    assert result["output"] == "/srv/project"
    assert created[0]["env_type"] == "ssh"
    assert created[0]["ssh_config"]["host"] == "devbox.example.com"
    assert created[0]["cwd"] == "/srv/project"
    assert file_ops.env is terminal_mod._active_environments[("default", "devbox")]


def test_ssh_paths_stay_remote_relative_and_ignore_host_workspace_override(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "backend": "local",
            "default_target": "devbox",
            "targets": {
                "devbox": {
                    "backend": "ssh",
                    "cwd": ".",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
                "container": {"backend": "docker", "cwd": "."},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    terminal_mod.register_task_env_overrides(
        "session", {"cwd": str(tmp_path)},
    )

    assert file_mod._authoritative_workspace_root("session", "devbox") == "."
    assert file_mod._backend_operation_path(
        "relative.txt", str(tmp_path / "relative.txt"), "session", "devbox",
    ) == "relative.txt"
    assert file_mod._backend_operation_path(
        "~/notes.txt", str(tmp_path / "notes.txt"), "session", "devbox",
    ) == "~/notes.txt"
    assert file_mod._authoritative_workspace_root(
        "session", "container",
    ) == "/root"

    terminal_mod.record_session_cwd(
        "session-a", "/srv/a", target="devbox",
    )
    terminal_mod.record_session_cwd(
        "session-b", "/srv/b", target="devbox",
    )
    assert file_mod._backend_operation_path(
        "relative.txt", "/host/wrong", "session-a", "devbox",
    ) == "/srv/a/relative.txt"
    assert file_mod._backend_operation_path(
        "relative.txt", "/host/wrong", "session-b", "devbox",
    ) == "/srv/b/relative.txt"
    rewritten = file_mod._backend_v4a_patch(
        "*** Begin Patch\n*** Update File: relative.txt\n*** End Patch",
        "session-a",
        "devbox",
    )
    assert "*** Update File: /srv/a/relative.txt" in rewritten


def test_fresh_ssh_file_scope_does_not_inherit_shared_environment_cwd(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools.file_operations import ShellFileOperations

    _, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "devbox",
            "targets": {
                "devbox": {
                    "backend": "ssh",
                    "cwd": ".",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    calls = []

    class SharedSshEnvironment:
        cwd = "/other-session"

        def execute(self, command, cwd=None, **kwargs):
            calls.append(cwd)
            return {"output": "", "returncode": 0}

    shared_ops = ShellFileOperations(SharedSshEnvironment())
    monkeypatch.setattr(file_mod, "_get_file_ops", lambda *args, **kwargs: shared_ops)
    resolution = targets_mod.resolve_execution_target("devbox")

    fresh_ops = file_mod._file_ops_for_resolution("fresh-session", resolution)
    fresh_ops._exec("true")
    terminal_mod, _ = isolated_target_state
    terminal_mod.record_session_cwd(
        "recorded-session", "/srv/recorded", target="devbox",
    )
    recorded_ops = file_mod._file_ops_for_resolution(
        "recorded-session", resolution,
    )
    recorded_ops._exec("true")

    assert calls == [".", "/srv/recorded"]


def test_legacy_ssh_search_preserves_relative_path(monkeypatch, isolated_target_state):
    from tools.file_operations import SearchResult

    _, file_mod = isolated_target_state
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "~")
    captured = {}

    class FakeOps:
        def search(self, **kwargs):
            captured.update(kwargs)
            return SearchResult(matches=[], total_count=0)

    monkeypatch.setattr(file_mod, "_get_file_ops", lambda *args, **kwargs: FakeOps())

    result = json.loads(file_mod.search_tool("needle", path="."))

    assert not result.get("error")
    assert captured["path"] == "."


def test_ssh_reads_never_use_host_mtime_dedup(
    monkeypatch, tmp_path, isolated_target_state,
):
    _, file_mod = isolated_target_state
    import tools.execution_targets as targets_mod
    from tools.file_operations import ReadResult

    host_twin = tmp_path / "same.txt"
    host_twin.write_text("host\n", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "devbox",
            "targets": {
                "devbox": {
                    "backend": "ssh",
                    "cwd": str(tmp_path),
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)

    class FakeOps:
        calls = 0

        @classmethod
        def read_file(cls, path, offset, limit):
            cls.calls += 1
            return ReadResult(
                content=f"remote-{cls.calls}", total_lines=1, file_size=8,
            )

    monkeypatch.setattr(
        file_mod, "_get_file_ops",
        lambda task_id, target=None, _resolution=None: FakeOps(),
    )

    first = json.loads(file_mod.read_file_tool(
        "same.txt", task_id="session", execution_target="devbox",
    ))
    second = json.loads(file_mod.read_file_tool(
        "same.txt", task_id="session", execution_target="devbox",
    ))

    assert first["content"] != second["content"]
    assert second.get("status") != "unchanged"
    assert FakeOps.calls == 2


def test_target_specific_cd_does_not_cross_talk_to_terminal_or_file_tools(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha_sub = alpha / "sub"
    beta_sub = beta / "sub"
    alpha_sub.mkdir(parents=True)
    beta_sub.mkdir(parents=True)
    (alpha_sub / "which.txt").write_text("alpha\n", encoding="utf-8")
    (beta / "which.txt").write_text("beta-root\n", encoding="utf-8")
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    cd_result = json.loads(terminal_mod.terminal_tool(
        "cd sub", task_id="session", execution_target="alpha",
    ))
    beta_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", execution_target="beta",
    ))
    alpha_read = json.loads(file_mod.read_file_tool(
        "which.txt", task_id="session", execution_target="alpha",
    ))
    beta_read = json.loads(file_mod.read_file_tool(
        "which.txt", task_id="session", execution_target="beta",
    ))

    assert cd_result["exit_code"] == 0
    assert beta_pwd["output"] == str(beta)
    assert "alpha" in alpha_read["content"]
    assert "beta-root" in beta_read["content"]


def test_local_targets_route_write_read_patch_search_and_cache_separately(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    _, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    write_alpha = json.loads(file_mod.write_file_tool(
        "shared.txt", "alpha-value\n", task_id="session",
        execution_target="alpha",
    ))
    write_beta = json.loads(file_mod.write_file_tool(
        "shared.txt", "beta-value\n", task_id="session",
        execution_target="beta",
    ))
    patch_alpha = json.loads(file_mod.patch_tool(
        path="shared.txt", old_string="alpha-value\n", new_string="alpha-patched\n",
        task_id="session", execution_target="alpha",
    ))
    search_alpha = json.loads(file_mod.search_tool(
        "alpha-patched", path=".", task_id="session", execution_target="alpha",
    ))
    search_beta = json.loads(file_mod.search_tool(
        "beta-value", path=".", task_id="session", execution_target="beta",
    ))

    assert not write_alpha.get("error") and not write_beta.get("error")
    assert not patch_alpha.get("error")
    assert (alpha / "shared.txt").read_text(encoding="utf-8") == "alpha-patched\n"
    assert (beta / "shared.txt").read_text(encoding="utf-8") == "beta-value\n"
    assert search_alpha["matches"] and search_beta["matches"]
    assert write_alpha["target"] == patch_alpha["target"] == "alpha"
    assert search_beta["target"] == "beta"
    assert set(file_mod._file_ops_cache) == {("default", "alpha"), ("default", "beta")}
    assert ("session", "alpha") in file_mod._read_tracker
    assert ("session", "beta") in file_mod._read_tracker


def test_real_config_loader_routes_terminal_and_files_between_two_local_targets(
    monkeypatch, tmp_path, isolated_target_state,
):
    import hermes_yaml as yaml

    import tools.file_tools as file_mod
    import tools.terminal_tool as terminal_mod

    alpha = tmp_path / "configured-alpha"
    beta = tmp_path / "configured-beta"
    alpha.mkdir()
    beta.mkdir()
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "terminal": {
                "default_target": "alpha",
                "targets": {
                    "alpha": {"backend": "local", "cwd": str(alpha)},
                    "beta": {"backend": "local", "cwd": str(beta)},
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    alpha_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="configured-session", execution_target="alpha",
    ))
    beta_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="configured-session", execution_target="beta",
    ))
    assert alpha_pwd["output"] == str(alpha)
    assert beta_pwd["output"] == str(beta)

    assert json.loads(file_mod.write_file_tool(
        "same.txt", "from alpha\n", task_id="configured-session", execution_target="alpha",
    ))["target"] == "alpha"
    assert json.loads(file_mod.write_file_tool(
        "same.txt", "from beta\n", task_id="configured-session", execution_target="beta",
    ))["target"] == "beta"
    assert "from alpha" in json.loads(file_mod.read_file_tool(
        "same.txt", task_id="configured-session", execution_target="alpha",
    ))["content"]
    assert "from beta" in json.loads(file_mod.read_file_tool(
        "same.txt", task_id="configured-session", execution_target="beta",
    ))["content"]


def test_workspace_override_applies_only_to_default_named_target(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    host_workspace = tmp_path / "host-workspace"
    alpha.mkdir()
    beta.mkdir()
    host_workspace.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha), "beta": str(beta)}),
    )

    # ACP/TUI/gateway surfaces register the host workspace as a task override.
    # It should seed the configured default target, not replace an explicit
    # remote/container target's own cwd.
    terminal_mod.register_task_env_overrides(
        "session", {"cwd": str(host_workspace)},
    )

    alpha_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", execution_target="alpha",
    ))
    beta_pwd = json.loads(terminal_mod.terminal_tool(
        "pwd", task_id="session", execution_target="beta",
    ))
    beta_write = json.loads(file_mod.write_file_tool(
        "target-only.txt", "beta\n", task_id="session", execution_target="beta",
    ))

    assert alpha_pwd["output"] == str(host_workspace)
    assert beta_pwd["output"] == str(beta)
    assert beta_write["resolved_path"] == str(beta / "target-only.txt")
    assert (beta / "target-only.txt").read_text(encoding="utf-8") == "beta\n"
    assert not (host_workspace / "target-only.txt").exists()


def test_unknown_target_errors_are_returned_by_execution_and_file_tools(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: _named_config({"alpha": str(alpha)}),
    )

    results = [
        json.loads(terminal_mod.terminal_tool("pwd", execution_target="missing")),
        json.loads(file_mod.read_file_tool("x.txt", execution_target="missing")),
        json.loads(file_mod.write_file_tool(
            "x.txt", "x", execution_target="missing",
        )),
        json.loads(file_mod.search_tool(
            "x", path=".", execution_target="missing",
        )),
        json.loads(code_mod.execute_code(
            "print('x')", execution_target="missing",
        )),
    ]

    for result in results:
        assert "missing" in result["error"]
        assert "Available targets: 'alpha'" in result["error"]
    assert not (alpha / "x.txt").exists()


def test_public_execution_routing_signatures_have_no_target_parameter(
    isolated_target_state,
):
    import inspect
    import tools.code_execution_tool as code_mod

    terminal_mod, file_mod = isolated_target_state
    for function in (
        terminal_mod.terminal_tool,
        file_mod.read_file_tool,
        file_mod.write_file_tool,
        file_mod.patch_tool,
        code_mod.execute_code,
    ):
        parameters = inspect.signature(function).parameters
        assert "execution_target" in parameters
        assert "target" not in parameters

    search_parameters = inspect.signature(file_mod.search_tool).parameters
    assert "target" in search_parameters
    assert "execution_target" in search_parameters


def test_cleanup_without_target_removes_all_task_scopes_and_explicit_removes_one(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state

    class FakeEnv:
        def __init__(self):
            self.cleaned = 0

        def cleanup(self):
            self.cleaned += 1

    config = _named_config({"alpha": "/a", "beta": "/b"})
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    alpha = FakeEnv()
    beta = FakeEnv()
    terminal_mod._active_environments.update({
        ("default", "alpha"): alpha,
        ("default", "beta"): beta,
    })
    terminal_mod._last_activity.update({
        ("default", "alpha"): time.time(),
        ("default", "beta"): time.time(),
    })

    # Unregistered delegate ids collapse to the parent's "default" scope for
    # execution, but closing a delegate must preserve legacy cleanup semantics
    # and must not tear down the parent's shared environments.
    terminal_lifecycle.cleanup_vm("child-a")
    terminal_lifecycle.cleanup_vm("child-a", target="alpha")
    assert alpha.cleaned == beta.cleaned == 0

    terminal_lifecycle.cleanup_vm("default", target="alpha")
    assert alpha.cleaned == 1 and beta.cleaned == 0
    assert set(terminal_mod._active_environments) == {("default", "beta")}

    terminal_lifecycle.cleanup_vm("default")
    assert beta.cleaned == 1
    assert terminal_mod._active_environments == {}


def test_per_turn_cleanup_preserves_only_persistent_named_siblings(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state

    class FakeEnv:
        def __init__(self, persistent):
            self._persistent = persistent
            self.cleaned = 0

        def cleanup(self):
            self.cleaned += 1

    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: _named_config({"alpha": "/a", "beta": "/b"}),
    )
    key_a = ("default", "alpha")
    key_b = ("default", "beta")
    persistent = FakeEnv(True)
    ephemeral = FakeEnv(False)
    terminal_mod._active_environments.update({
        key_a: persistent,
        key_b: ephemeral,
    })
    terminal_mod._last_activity.update({key_a: 1.0, key_b: 1.0})
    persistent_ops = object()
    ephemeral_ops = object()
    file_mod._file_ops_cache.update({
        key_a: persistent_ops,
        key_b: ephemeral_ops,
    })

    terminal_mod.register_environment_turn("turn-a")
    terminal_mod.register_environment_turn("turn-b")
    assert terminal_mod.release_environment_turn("turn-a") == 1
    terminal_lifecycle.cleanup_vm(
        "turn-a", preserve_persistent=True, include_collapsed=False,
    )
    assert key_b in terminal_mod._active_environments

    assert terminal_mod.release_environment_turn("turn-b") == 0
    terminal_lifecycle.cleanup_vm(
        "turn-b", preserve_persistent=True, include_collapsed=True,
    )

    assert terminal_mod._active_environments[key_a] is persistent
    assert key_b not in terminal_mod._active_environments
    assert persistent.cleaned == 0
    assert ephemeral.cleaned == 1
    assert file_mod._file_ops_cache[key_a] is persistent_ops
    assert key_b not in file_mod._file_ops_cache

    from tools.process_registry import process_registry

    active_env = FakeEnv(False)
    terminal_mod._active_environments[key_b] = active_env
    monkeypatch.setattr(
        process_registry, "has_active_processes", lambda key: key == key_b,
    )
    terminal_lifecycle.cleanup_vm(
        "turn-c", preserve_persistent=True, include_collapsed=True,
    )
    assert terminal_mod._active_environments[key_b] is active_env
    assert active_env.cleaned == 0


def test_complete_logical_turn_cleanup_waits_for_overlapping_turn(
    monkeypatch, isolated_target_state,
):
    """Cleanup releases only its own logical-turn lease.

    Two gateway turns can overlap while their child task ids collapse onto the
    same shared target environments. The first finisher must not tear down an
    ephemeral sibling still in use by the other complete logical turn.
    """
    from agent import chat_completion_helpers as chat_helpers

    terminal_mod, _ = isolated_target_state
    vm_calls = []
    browser_calls = []
    runtime = SimpleNamespace(
        cleanup_vm=lambda task_id, **kwargs: vm_calls.append((task_id, kwargs)),
        cleanup_browser=lambda task_id: browser_calls.append(task_id),
    )
    monkeypatch.setattr(chat_helpers, "_ra", lambda: runtime)
    fake_agent = SimpleNamespace(verbose_logging=False)

    with terminal_mod.logical_environment_turn("turn-a"):
        assert terminal_mod.active_environment_turns("turn-a") == 1
        with terminal_mod.logical_environment_turn("turn-b"):
            assert terminal_mod.active_environment_turns("turn-b") == 2
            chat_helpers.cleanup_task_resources(fake_agent, "turn-b")
            assert terminal_mod.active_environment_turns("turn-a") == 1
            assert vm_calls[-1] == (
                "turn-b",
                {"preserve_persistent": True, "include_collapsed": False},
            )
        chat_helpers.cleanup_task_resources(fake_agent, "turn-a")
        assert terminal_mod.active_environment_turns("turn-a") == 0
        assert vm_calls[-1] == (
            "turn-a",
            {"preserve_persistent": True, "include_collapsed": True},
        )

    # Both context-manager finalizers are idempotent after cleanup released
    # their leases; no negative count or leaked logical-turn owner remains.
    assert terminal_mod.active_environment_turns("turn-a") == 0
    assert browser_calls == ["turn-b", "turn-a"]


def test_logical_environment_turn_exception_releases_once(isolated_target_state):
    terminal_mod, _ = isolated_target_state

    with pytest.raises(RuntimeError, match="boom"):
        with terminal_mod.logical_environment_turn("turn-exception"):
            assert terminal_mod.active_environment_turns("turn-exception") == 1
            raise RuntimeError("boom")

    assert terminal_mod.active_environment_turns("turn-exception") == 0


def test_local_persistent_config_marks_environment(
    monkeypatch, isolated_target_state,
):
    class FakeLocal:
        def __init__(self, cwd, timeout, persistent=True):
            self.cwd = cwd
            self.timeout = timeout
            self._persistent = persistent

    import tools.terminal_tool_backends as backends
    from tools.terminal_tool_lifecycle import _environment_is_persistent

    monkeypatch.setattr(backends, "_LocalEnvironment", FakeLocal)
    env = backends._create_environment(
        "local", "", "/workspace", 30,
        local_config={"persistent": True},
    )

    assert _environment_is_persistent(env)


def test_process_metadata_survives_status_list_and_checkpoint(monkeypatch, tmp_path):
    import tools.process_registry as process_mod

    checkpoint = tmp_path / "processes.json"
    monkeypatch.setattr(process_mod, "CHECKPOINT_PATH", checkpoint)
    registry = process_mod.ProcessRegistry()
    session = process_mod.ProcessSession(
        id="proc_targeted",
        command="sleep 30",
        task_id="default",
        target="devbox",
        backend="ssh",
        timeout_seconds=7,
        environment_task_key="profile-a:default",
        pid=4321,
        host_start_time=99,
        started_at=time.time(),
    )
    registry._running[session.id] = session

    assert registry.has_active_processes(("profile-a:default", "devbox"))
    assert registry.poll(session.id)["target"] == "devbox"
    assert registry.list_sessions()[0]["backend"] == "ssh"
    registry._write_checkpoint()
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw[0]["target"] == "devbox"
    assert raw[0]["timeout_seconds"] == 7
    assert raw[0]["environment_task_key"] == "profile-a:default"
    assert raw[0]["backend"] == "ssh"

    recovered = process_mod.ProcessRegistry()
    monkeypatch.setattr(recovered, "_host_pid_is_ours", lambda pid, started: True)
    assert recovered.recover_from_checkpoint() == 1
    recovered_session = recovered.get(session.id)
    assert recovered_session is not None
    assert recovered_session.target == "devbox"
    assert recovered_session.backend == "ssh"
    assert recovered_session.timeout_seconds == 7
    assert recovered_session.environment_task_key == "profile-a:default"

    recovered_session.exited = True
    waited = recovered.wait(session.id, timeout=99)
    assert "configured limit of 7s" in waited["timeout_note"]


def test_tool_output_persistence_uses_the_result_target(monkeypatch):
    import agent.tool_executor as executor

    default_env = object()
    beta_env = object()

    def fake_get_active_env(_task_id, target=None):
        return beta_env if target == "beta" else default_env

    monkeypatch.setattr(executor, "get_active_env", fake_get_active_env)

    result = json.dumps({"target": "beta", "output": "x"})
    assert executor._active_env_for_tool_result(
        "session", "terminal", {}, result,
    ) is beta_env

    import tools.terminal_tool as terminal_mod

    scoped_env = object()
    monkeypatch.setattr(
        terminal_mod, "get_environment_for_target_scope",
        lambda task_id, target, scope: scoped_env,
    )
    scoped_result = json.dumps({
        "target": "beta", "runtime_scope": "old-scope", "output": "x",
    })
    assert executor._active_env_for_tool_result(
        "session", "terminal", {}, scoped_result,
    ) is scoped_env

    from tools.process_registry import process_registry

    producing_env = object()
    monkeypatch.setattr(
        process_registry, "get",
        lambda session_id: SimpleNamespace(env_ref=producing_env),
    )
    assert executor._active_env_for_tool_result(
        "session", "process", {"session_id": "proc-old"},
        json.dumps({"target": "beta", "output": "large"}),
    ) is producing_env

    selected = {"content": "large", "tool_call_id": "call-beta"}
    ordinary = {"content": "web result", "tool_call_id": "call-web"}
    messages = [selected, ordinary]
    identities = {"call-beta": ("beta", None, beta_env)}
    captured = []

    def fake_enforce(turn_messages, *, env_resolver=None, config=None):
        assert callable(env_resolver)
        captured.extend(env_resolver(message) for message in turn_messages)

    monkeypatch.setattr(executor, "enforce_turn_budget", fake_enforce)
    agent = SimpleNamespace(
        _execution_target_by_tool_call=identities,
        _apply_pending_steer_to_tool_results=lambda *_args: None,
    )
    executor._finalize_tool_batch(
        agent, messages, "session", len(messages), executor.DEFAULT_BUDGET,
    )
    assert captured == [beta_env, default_env]
    assert identities == {}
    assert "_execution_target" not in selected

    persisted = executor._persisted_target_hint(
        f"{executor.PERSISTED_OUTPUT_TAG}\nfull output saved", "beta", None,
    )
    assert 'Execution target for this saved output: "beta"' in persisted
    assert 'execution_target="beta"' in persisted
    scoped = executor._persisted_target_hint(
        f"saved {executor.PERSISTED_OUTPUT_TAG}", "beta", "scope-v1",
    )
    assert 'runtime_scope="scope-v1"' in scoped

    monkeypatch.setattr(
        executor, "get_active_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad target config")),
    )
    assert executor._active_env_for_tool_result(
        "session", "terminal", {"execution_target": "broken"},
    ) is None
    captured.clear()
    identities["call-broken"] = ("broken", None, None)
    executor._finalize_tool_batch(
        agent, [{"content": "error", "tool_call_id": "call-broken"}],
        "session", 1, executor.DEFAULT_BUDGET,
    )
    assert captured == [None]
    assert identities == {}


def test_subdirectory_hints_follow_local_target_and_skip_remote_host(
    monkeypatch, tmp_path, isolated_target_state,
):
    from types import SimpleNamespace

    import agent.tool_executor as executor
    import tools.execution_targets as targets_mod

    local_root = tmp_path / "local-target"
    subdir = local_root / "src"
    subdir.mkdir(parents=True)
    (subdir / "AGENTS.md").write_text("LOCAL TARGET HINT", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "local",
            "targets": {
                "local": {"backend": "local", "cwd": str(local_root)},
                "devbox": {
                    "backend": "ssh",
                    "cwd": "/srv/project",
                    "ssh_host": "devbox.example.com",
                    "ssh_user": "agent",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)

    class DefaultTracker:
        def __init__(self):
            self.calls = 0

        def check_tool_call(self, *args, **kwargs):
            self.calls += 1
            return "WRONG HOST HINT"

    default_tracker = DefaultTracker()
    agent = SimpleNamespace(_subdirectory_hints=default_tracker)
    local_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, "local",
    )
    remote_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, "devbox",
    )
    config["terminal"]["default_target"] = "devbox"
    omitted_remote_hint = executor._target_subdirectory_hints(
        agent, "session", "read_file", {"path": "src/main.py"}, None,
    )

    assert isinstance(local_hint, str)
    assert "LOCAL TARGET HINT" in local_hint
    assert executor._selected_local_target_cwd(
        "session", "write_file", {"execution_target": "local"},
    ) == str(local_root)
    assert executor._selected_local_target_cwd(
        "session", "terminal", {"execution_target": "devbox"},
    ) is None
    assert remote_hint is None
    assert omitted_remote_hint is None
    assert default_tracker.calls == 0


def test_subdirectory_hints_reject_repointed_runtime_scope(
    monkeypatch, tmp_path, isolated_target_state,
):
    from types import SimpleNamespace

    import agent.tool_executor as executor
    import tools.execution_targets as targets_mod

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    (second / "src").mkdir(parents=True)
    (second / "src" / "AGENTS.md").write_text(
        "REPLACEMENT TARGET HINT",
        encoding="utf-8",
    )
    config_a = _named_config({"dev": str(first)}, default="dev")
    config_b = _named_config({"dev": str(second)}, default="dev")
    targets_mod.set_execution_target_config_source(config_a)
    producing_scope = targets_mod.resolve_execution_target("dev").security_scope
    targets_mod.set_execution_target_config_source(config_b)

    try:
        hint = executor._target_subdirectory_hints(
            SimpleNamespace(_subdirectory_hints=None),
            "session",
            "read_file",
            {"path": "src/main.py"},
            "dev",
            runtime_scope=producing_scope,
        )
    finally:
        targets_mod.set_execution_target_config_source(None)

    assert hint is None


def test_targetless_intervening_tool_resets_all_target_read_trackers(
    monkeypatch, isolated_target_state,
):
    import model_tools
    import tools.execution_targets as targets_mod

    _, file_mod = isolated_target_state
    assert "memory" not in model_tools._TARGET_SELECTOR_TOOLS
    assert "terminal" in model_tools._TARGET_SELECTOR_TOOLS
    for key in ("session", ("session", "alpha"), ("session", "beta")):
        file_mod._read_tracker[key] = {
            "last_key": "same",
            "consecutive": 4,
            "dedup_hits": {"same": 2},
        }

    read_tracking.notify_other_tool_call("session")

    for data in file_mod._read_tracker.values():
        assert data["last_key"] is None
        assert data["consecutive"] == 0
        assert data["dedup_hits"] == {}

    monkeypatch.setattr(targets_mod, "_active_profile_scope", lambda: "profile-a")
    profile_key = ("profile-profile-a:session", "alpha")
    file_mod._read_tracker[profile_key] = {
        "dedup": {"region": (1.0, 2, "hash")},
        "dedup_hits": {"region": 3},
        "dedup_generation_reads": {"region"},
    }
    read_tracking.reset_file_dedup("session")
    # Fork main retains mtimes across compression; only the generation and
    # stub-loop counters reset, including for the current profile's targets.
    assert file_mod._read_tracker[profile_key]["dedup"] == {"region": (1.0, 2, "hash")}
    assert file_mod._read_tracker[profile_key]["dedup_hits"] == {}
    assert file_mod._read_tracker[profile_key]["dedup_generation_reads"] == set()


def test_requirements_keep_tools_registered_when_any_target_is_usable(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    config = {
        "terminal": {
            "default_target": "broken-docker",
            "targets": {
                "broken-docker": {"backend": "docker", "cwd": "/workspace"},
                "local": {"backend": "local", "cwd": "/workspace/local"},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    checked = []
    monkeypatch.setattr(
        terminal_mod,
        "_check_terminal_config_requirements",
        lambda cfg: checked.append(cfg["env_type"]) or cfg["env_type"] == "local",
    )

    assert terminal_mod.check_terminal_requirements() is True
    assert checked == ["local"]


def test_provider_backed_local_target_has_runtime_scoped_file_namespace(
    isolated_target_state,
):
    _, file_mod = isolated_target_state
    resolution = SimpleNamespace(
        backend="local",
        named=True,
        target="local-runtime",
        profile_scope="worker",
        provider="hetzner",
        security_scope="server-123",
        config={"backend": "local", "cwd": "/workspace"},
    )

    namespace = file_mod._file_state_namespace(
        "session",
        "local-runtime",
        _resolution=resolution,
    )

    assert namespace == "profile-worker:runtime-server-123:local"


def test_local_target_aliases_share_file_state_lock_namespace(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import file_state

    _, file_mod = isolated_target_state
    shared = tmp_path / "shared"
    shared.mkdir()
    path = shared / "same.txt"
    path.write_text("v1", encoding="utf-8")
    config = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {"backend": "local", "cwd": str(shared)},
                "alias": {"backend": "local", "cwd": str(shared)},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    assert file_mod._file_state_namespace("reader", "alpha") is None
    assert file_mod._file_state_namespace("writer", "alias") is None

    registry = file_state.FileStateRegistry()
    monkeypatch.setattr(file_state, "_registry", registry)
    file_state.record_read(("reader", "alpha"), path, namespace=None)
    file_state.note_write(("writer", "alias"), path, namespace=None)
    warning = file_state.check_stale(("reader", "alpha"), path, namespace=None)
    assert warning is not None
    assert "writer" in warning


def test_ssh_target_aliases_share_file_state_lock_namespace(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import file_state

    _, file_mod = isolated_target_state
    ssh = {
        "backend": "ssh",
        "ssh_host": "devbox.example",
        "ssh_user": "agent",
        "ssh_port": 2222,
        "cwd": "/srv/project",
    }
    config = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": dict(ssh),
                "alias": {**ssh, "cwd": "/different/root"},
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)

    alpha = file_mod._file_state_namespace("reader", "alpha")
    alias = file_mod._file_state_namespace("writer", "alias")
    assert alpha == alias
    assert alpha.startswith("ssh:")

    alpha_resolution = targets_mod.resolve_execution_target("alpha")
    alias_resolution = targets_mod.resolve_execution_target("alias")
    alpha_key = alpha_resolution.file_coordination_key("reader")
    alias_key = alias_resolution.file_coordination_key("writer")
    path = "/srv/project/shared.txt"
    registry = file_state.FileStateRegistry()
    registry.record_read(alpha_key, path, namespace=alpha, stat_path=False)
    registry.note_write(alias_key, path, namespace=alias, stat_path=False)
    warning = registry.check_stale(alpha_key, path, namespace=alpha)

    assert warning is not None
    assert "writer" in warning
    assert registry._lock_for(path, alpha) is registry._lock_for(path, alias)


def test_persistent_docker_replacements_share_storage_lock_namespace(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    _, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "sandbox",
            "timeout": 30,
            "targets": {
                "sandbox": {
                    "backend": "docker",
                    "container_persistent": True,
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    before = file_mod._file_state_namespace("session", "sandbox")
    config["terminal"]["timeout"] = 90
    after = file_mod._file_state_namespace("session", "sandbox")

    assert before == after
    assert before.startswith("docker-storage:")


@pytest.mark.parametrize("builder", ["terminal", "file", "execute_code"])
def test_idle_persistent_docker_hot_edit_cleans_runtime_before_replacement(
    monkeypatch, isolated_target_state, builder,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "sandbox",
            "timeout": 30,
            "targets": {
                "sandbox": {
                    "backend": "docker",
                    "container_persistent": True,
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    created = []
    cleanup_calls = []

    class FakeEnvironment:
        cwd = "/workspace"

        def __init__(self):
            self.cleaned = False

        def execute(self, command, **kwargs):
            return {"output": "ok\n", "returncode": 0}

        def cleanup(self, force_remove=False):
            self.cleaned = True
            cleanup_calls.append(force_remove)

        def wait_for_cleanup(self, timeout):
            return True

    def fake_create(**kwargs):
        if created:
            assert created[0].cleaned is True
        env = FakeEnvironment()
        created.append(env)
        return env

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    def build():
        if builder == "terminal":
            result = json.loads(terminal_mod.terminal_tool(
                "pwd", task_id="session", execution_target="sandbox",
            ))
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
            return result
        if builder == "file":
            return file_mod._get_file_ops("session", "sandbox")
        return code_mod._get_or_create_env("session", "sandbox")

    first = build()
    config["terminal"]["timeout"] = 90
    terminal_mod._active_turn_counts["default"] = 2
    expected_error = ValueError if builder == "execute_code" else RuntimeError
    with pytest.raises(expected_error, match="still active"):
        build()
    assert len(created) == 1
    assert cleanup_calls == []
    assert terminal_mod._active_environments[("default", "sandbox")] is created[0]

    terminal_mod._active_turn_counts.clear()
    second = build()

    if builder == "terminal":
        assert first["output"].strip() == second["output"].strip() == "ok"
    assert len(created) == 2
    assert cleanup_calls == [True]
    assert terminal_mod._retired_environments == []


@pytest.mark.parametrize("builder", ["terminal", "file", "execute_code"])
def test_persistent_docker_hot_edit_cleanup_failure_restores_runtime_ownership(
    monkeypatch, isolated_target_state, builder,
):
    import tools.code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "sandbox",
            "timeout": 30,
            "targets": {
                "sandbox": {
                    "backend": "docker",
                    "container_persistent": True,
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    created = []

    class FakeEnvironment:
        cwd = "/workspace"

        def __init__(self):
            self._persist_across_processes = True
            self._persistent = True

        def execute(self, command, **kwargs):
            return {"output": "ok\n", "returncode": 0}

        def cleanup(self, force_remove=False):
            assert force_remove is True
            assert self._persist_across_processes is False
            assert self._persistent is True
            raise RuntimeError("lease handoff failed")

    def fake_create(**kwargs):
        env = FakeEnvironment()
        created.append(env)
        return env

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)

    def build():
        if builder == "terminal":
            result = json.loads(terminal_mod.terminal_tool(
                "pwd", task_id="session", execution_target="sandbox",
            ))
            if result.get("status") == "error":
                raise RuntimeError(result["error"])
            return result
        if builder == "file":
            return file_mod._get_file_ops("session", "sandbox")
        return code_mod._get_or_create_env("session", "sandbox")

    first = build()
    old_env = created[0]
    config["terminal"]["timeout"] = 90

    if builder == "terminal":
        assert first["output"].strip() == "ok"
    expected_error = ValueError if builder == "execute_code" else RuntimeError
    with pytest.raises(expected_error, match="Could not retire"):
        build()
    assert len(created) == 1
    assert terminal_mod._active_environments[("default", "sandbox")] is old_env
    assert old_env._persist_across_processes is True
    assert old_env._persistent is True


def test_local_background_process_keeps_producing_target_generation(
    monkeypatch, tmp_path, isolated_target_state,
):
    import agent.tool_executor as executor
    import tools.execution_targets as targets_mod
    from tools.process_registry import ProcessSession, process_registry

    terminal_mod, _ = isolated_target_state
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    config = _named_config({"alpha": str(alpha)})
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    captured = {}

    def fake_spawn_local(**kwargs):
        captured.update(kwargs)
        return ProcessSession(
            id="proc-alpha",
            command=kwargs["command"],
            pid=123,
            cwd=kwargs["cwd"],
            target=kwargs["target"],
            backend=kwargs["backend"],
            runtime_scope=kwargs["runtime_scope"],
            env_ref=kwargs["env_ref"],
        )

    monkeypatch.setattr(process_registry, "spawn_local", fake_spawn_local)
    result = json.loads(terminal_mod.terminal_tool(
        "sleep 1", task_id="session", execution_target="alpha", background=True,
    ))
    producing_env = captured["env_ref"]
    producing_scope = targets_mod.resolve_execution_target("alpha").security_scope

    assert result["session_id"] == "proc-alpha"
    assert captured["runtime_scope"] == producing_scope
    assert producing_env is terminal_mod._active_environments[("default", "alpha")]

    config["terminal"]["targets"]["alpha"]["cwd"] = str(tmp_path / "new-alpha")
    session = ProcessSession(
        id="proc-alpha",
        command="sleep 1",
        target="alpha",
        backend="local",
        runtime_scope=producing_scope,
        env_ref=producing_env,
    )
    monkeypatch.setattr(process_registry, "get", lambda session_id: session)
    process_result = json.dumps({
        "target": "alpha",
        "runtime_scope": producing_scope,
        "output": "large",
    })
    assert executor._active_env_for_tool_result(
        "session", "process", {"session_id": "proc-alpha"}, process_result,
    ) is producing_env


def test_retired_environment_waits_for_base_task_turn_scope(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state
    env = object()
    now = time.time()
    terminal_mod._active_turn_counts["bench"] = 1
    terminal_mod._retired_environments.append((("bench", "alpha"), env, now - 120))
    monkeypatch.setattr(
        terminal_mod, "_cleanup_environment_resource",
        lambda *args, **kwargs: pytest.fail("active environment was cleaned"),
    )

    assert terminal_mod._cleanup_retired_environments(
        min_age_seconds=0.0, require_idle=True,
    ) == 0
    assert terminal_mod._retired_environments[0][1] is env


def test_legacy_flat_config_still_uses_plain_string_keys(
    monkeypatch, tmp_path, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    monkeypatch.setattr(
        targets_mod, "_load_merged_config",
        lambda: {"terminal": {"backend": "local", "cwd": str(tmp_path), "timeout": 60}},
    )

    result = json.loads(terminal_mod.terminal_tool("pwd", task_id="legacy"))

    assert result["exit_code"] == 0
    assert "default" in terminal_mod._active_environments
    assert all(not isinstance(key, tuple) for key in terminal_mod._active_environments)


def test_named_local_read_reports_target_for_full_and_deduplicated_reads(tmp_path):
    import tools.execution_targets as targets_mod
    from tools.file_tools import read_file_tool

    (tmp_path / "sample.txt").write_text("routed read")
    targets_mod.set_execution_target_config_source(
        _named_config({"read-box": str(tmp_path)}, default="read-box")
    )
    try:
        for _ in range(2):
            result = json.loads(read_file_tool(
                "sample.txt", task_id="target-read-metadata", execution_target="read-box",
            ))
            assert result["target"] == "read-box"
            assert result["backend"] == "local"
            assert result["cwd"] == str(tmp_path)
    finally:
        targets_mod.set_execution_target_config_source(None)


def test_current_tool_leases_do_not_block_own_replacement_but_other_users_do(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state
    environment_key = ("default", "dev")
    env = SimpleNamespace(_hermes_stable_storage=True)

    with terminal_mod.logical_environment_turn("turn-a"):
        with terminal_mod.environment_turn_usage(
            "turn-a", environment_key=environment_key,
        ):
            assert not terminal_mod._environment_replacement_is_busy(
                env, environment_key,
            )
            other_key = terminal_mod.register_environment_turn("turn-b")
            try:
                assert terminal_mod._environment_replacement_is_busy(
                    env, environment_key,
                )
            finally:
                target_lifecycle._release_environment_turn_key(other_key)


def test_idle_reaper_and_deferred_cleanup_wait_for_file_only_tool_lease(
    monkeypatch, isolated_target_state,
):
    terminal_mod, _ = isolated_target_state
    cleaned = []
    deferred = []
    environment_key = ("default", "dev")

    class Environment:
        def cleanup(self, force_remove=False):
            cleaned.append(force_remove)

    terminal_mod._active_environments[environment_key] = Environment()
    terminal_mod._last_activity[environment_key] = 0.0
    monkeypatch.setattr(
        terminal_lifecycle,
        "cleanup_vm",
        lambda task_id, **kw: deferred.append((task_id, kw)),
    )

    with terminal_mod.logical_environment_turn("file-turn"):
        with terminal_mod.environment_turn_usage(
            "file-turn", environment_key=environment_key,
        ):
            terminal_mod._cleanup_inactive_envs(lifetime_seconds=0)
            assert environment_key in terminal_mod._active_environments
            assert cleaned == []
            terminal_mod.release_logical_environment_turn_for_cleanup("file-turn")
            terminal_mod.defer_environment_turn_cleanup("file-turn")
            assert deferred == []
        assert deferred == [(
            "file-turn",
            {"preserve_persistent": True, "include_collapsed": True},
        )]


def test_negative_not_found_cache_is_target_scoped_and_skips_host_oracle(
    monkeypatch, isolated_target_state,
):
    _, file_mod = isolated_target_state
    alpha_state = ("default", "alpha")
    beta_state = ("default", "beta")
    remote_path = "/srv/project/missing.txt"
    file_mod._record_not_found(
        "read_file", remote_path, alpha_state, '{"error": "missing"}',
    )
    monkeypatch.setattr(
        file_mod.os.path,
        "exists",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("remote cache must not inspect the host filesystem")
        ),
    )

    assert file_mod._check_not_found_cache(
        "read_file", remote_path, alpha_state, check_host_filesystem=False,
    )
    assert file_mod._check_not_found_cache(
        "read_file", remote_path, beta_state, check_host_filesystem=False,
    ) is None


@pytest.mark.parametrize("builder", ["terminal", "file", "code"])
def test_all_environment_builders_forward_docker_shm_size(
    builder, monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod
    from tools import code_execution_tool as code_mod

    terminal_mod, file_mod = isolated_target_state
    config = {
        "terminal": {
            "default_target": "sandbox",
            "targets": {
                "sandbox": {
                    "backend": "docker",
                    "cwd": "/workspace",
                    "docker_image": "python:3.12-slim",
                    "docker_shm_size": "768m",
                },
            },
        },
    }
    monkeypatch.setattr(targets_mod, "_load_merged_config", lambda: config)
    captured = []

    class FakeEnvironment:
        cwd = "/workspace"
        _persistent = False

        def execute(self, _command, **_kwargs):
            return {"output": "/workspace\n", "returncode": 0}

        def cleanup(self, force_remove=False):
            return None

        def wait_for_cleanup(self, timeout):
            return True

    def fake_create(**kwargs):
        captured.append(kwargs["container_config"])
        return FakeEnvironment()

    monkeypatch.setattr(terminal_mod, "_create_environment", fake_create)
    if builder == "terminal":
        result = json.loads(terminal_mod.terminal_tool(
            "pwd", task_id="shm-terminal", execution_target="sandbox",
        ))
        assert result["exit_code"] == 0
    elif builder == "file":
        file_mod._get_file_ops("shm-file", "sandbox")
    else:
        code_mod._get_or_create_env("shm-code", "sandbox")

    assert captured[0]["docker_shm_size"] == "768m"


def test_file_tool_lease_key_uses_explicit_nondefault_target(
    monkeypatch, isolated_target_state,
):
    import tools.execution_targets as targets_mod

    terminal_mod, _ = isolated_target_state
    monkeypatch.setattr(
        targets_mod,
        "_load_merged_config",
        lambda: _named_config({"alpha": "/alpha", "beta": "/beta"}),
    )
    expected = targets_mod.resolve_execution_target("beta").session_key("default")

    assert terminal_mod.execution_environment_turn_key(
        "write_file", {"execution_target": "beta"}, task_id="child-task",
    ) == expected
    assert terminal_mod.execution_environment_turn_key(
        "process", {"session_id": "proc_123"}, task_id="child-task",
    ) == target_lifecycle._turn_scope_key("child-task")
