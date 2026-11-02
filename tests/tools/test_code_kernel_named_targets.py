"""Named target authority on the newer persistent-kernel dispatch path."""
import json
from pathlib import Path
import subprocess

import pytest

from tools import code_kernel_remote as kernels


class ShellTransport:
    """A real subprocess transport with a private staging directory."""
    def __init__(self, root, scope):
        self.root = root
        self._hermes_target_scope = scope

    def get_temp_dir(self):
        return str(self.root)

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        result = subprocess.run(["bash", "-c", command], cwd=cwd or self.root,
                                input=stdin_data, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"output": result.stdout, "returncode": result.returncode}


def test_frozen_target_keeps_inherited_backend_defaults(tmp_path):
    from tools.code_execution_tool import _frozen_target_config
    from tools.execution_targets import execution_target_config_scope, resolve_execution_target
    config = {"terminal": {"default_target": "alpha", "modal_mode": "auto",
                          "targets": {"alpha": {"backend": "local", "cwd": str(tmp_path)}}}}
    selected = resolve_execution_target("alpha", config=config)
    with execution_target_config_scope(_frozen_target_config(selected)):
        frozen = resolve_execution_target("alpha")
    assert frozen.security_scope == selected.security_scope


@pytest.mark.platforms("posix")
def test_remote_kernel_never_reuses_another_target_environment(tmp_path):
    envs = [ShellTransport(tmp_path, scope) for scope in ("alpha", "beta")]
    def run(env, code):
        return kernels.execute_in_remote_kernel(
            code, env=env, env_type="ssh", task_env_id="same-owner",
            sandbox_tools=frozenset(), timeout=15, max_tool_calls=0, reset=False)
    try:
        first = run(envs[0], "value='alpha'; print(value)")
        second = run(envs[1], "print(globals().get('value', 'fresh'))")
        again = run(envs[0], "print(value)")
        assert first["status"] == second["status"] == again["status"] == "success"
        assert second["stdout"].strip() == "fresh", second
        assert again["stdout"].strip() == "alpha", again
    finally:
        kernels.shutdown_all_remote_kernels()


@pytest.mark.platforms("posix")
def test_named_remote_project_cells_keep_state_and_selected_cwd(tmp_path, monkeypatch):
    from tools import code_execution_tool as code
    from tools.execution_targets import execution_target_config_scope
    workspace = tmp_path / "project space"
    workspace.mkdir()
    env = ShellTransport(tmp_path, "alpha")
    config = {"terminal": {"default_target": "alpha", "targets": {
        "alpha": {"backend": "ssh", "ssh_host": "fixture.invalid", "ssh_user": "agent", "cwd": str(workspace)}}}}
    monkeypatch.setattr(code, "_get_or_create_env", lambda *a: (env, "ssh"))
    try:
        with execution_target_config_scope(config):
            first = json.loads(code._execute_remote("value=42", "owner", [], target="alpha", mode="project"))
            second = json.loads(code._execute_remote("import os; print(value, os.getcwd())", "owner", [], target="alpha", mode="project"))
        assert first["status"] == second["status"] == "success", second
        assert f"42 {workspace}" in second["output"]
        assert second["kernel"]["reused"] is True
    finally:
        kernels.shutdown_all_remote_kernels()
