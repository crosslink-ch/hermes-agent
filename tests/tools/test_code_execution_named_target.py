def test_execute_code_rpc_uses_frozen_approved_target_config():
    from tools import code_execution_tool as code_mod
    import tools.execution_targets as targets_mod

    approved = {
        "terminal": {
            "default_target": "alpha",
            "targets": {
                "alpha": {"backend": "local", "cwd": "/approved"},
            },
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
                },
            },
        },
    }
    targets_mod.set_execution_target_config_source(live)

    def handler(_name, args, task_id=None):
        selected = targets_mod.resolve_execution_target(args["execution_target"])
        current = targets_mod.resolve_live_execution_target(args["execution_target"])
        return selected.backend, selected.config.get("cwd"), current.backend

    try:
        assert code_mod._dispatch_rpc_tool(
            handler,
            "write_file",
            {"execution_target": "alpha"},
            "task",
            approved,
        ) == ("local", "/approved", "ssh")
    finally:
        targets_mod.set_execution_target_config_source(None)


def test_generated_stubs_default_to_approved_target_and_frozen_config_denies_pivot():
    import pytest
    from tools import code_execution_tool as code_mod
    from tools.execution_targets import resolve_execution_target, execution_target_config_scope

    config = {"terminal": {"default_target": "alpha", "targets": {
        "alpha": {"backend": "local"}, "beta": {"backend": "local"}}}}
    approved = resolve_execution_target("beta", config=config)
    frozen = code_mod._frozen_target_config(approved)
    with execution_target_config_scope(frozen):
        assert resolve_execution_target("beta").security_scope == approved.security_scope
        with pytest.raises(ValueError):
            resolve_execution_target("alpha")
        token = code_mod._bound_code_target.set("beta")
        try:
            source = code_mod.generate_hermes_tools_module(["read_file", "write_file"])
        finally:
            code_mod._bound_code_target.reset(token)
        assert "execution_target: str = 'beta'" in source
    assert code_mod._inherit_execution_target("search_files", {"target": "files"}, "beta") == {
        "target": "files", "execution_target": "beta"}
    with pytest.raises(ValueError, match="cannot select"):
        code_mod._inherit_execution_target("write_file", {"execution_target": "alpha"}, "beta")
