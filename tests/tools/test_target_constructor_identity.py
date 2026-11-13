"""File-first creation uses the same immutable backend identity as terminal-first."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend", ["ssh", "docker"])
def test_file_first_creation_keeps_named_constructor_identity(tmp_path, monkeypatch, backend):
    from tools import file_tools, terminal_tool
    from tools.execution_targets import execution_target_config_scope, resolve_execution_target
    target = {"backend": backend, "cwd": "/workspace"}
    if backend == "ssh":
        target.update(ssh_host="fixture.invalid", ssh_user="agent")
    cfg = {"terminal": {"default_target": "named", "docker_image_pinned": True,
                        "docker_snap_compat": True, "targets": {"named": target}}}
    seen = {}
    def construct(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(cwd="/workspace")
    monkeypatch.setattr(terminal_tool, "_create_environment", construct)
    with execution_target_config_scope(cfg):
        selected = resolve_execution_target("named")
        file_tools._create_terminal_env_for_file_ops("owner", "default", selected, selected.backend_task_id("default"))
    if backend == "ssh":
        assert seen["ssh_config"]["runtime_scope"] == selected.security_scope
    else:
        assert seen["container_config"]["storage_task_id"] == selected.storage_task_id("default")
        assert seen["container_config"]["docker_image_pinned"] is True
        assert seen["container_config"]["docker_snap_compat"] is True
