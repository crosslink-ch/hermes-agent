"""Shared test fixtures and builders; no collected tests."""
from __future__ import annotations
import pytest
from tools import file_tools_read_tracking as read_tracking

def _named_config(cwds: dict[str, str], default: str = "alpha") -> dict:
    return {
        "terminal": {
            "timeout": 60,
            "lifetime_seconds": 3600,
            "default_target": default,
            "targets": {
                name: {"backend": "local", "cwd": cwd}
                for name, cwd in cwds.items()
            },
        },
    }


@pytest.fixture
def isolated_target_state(monkeypatch):
    import tools.file_tools as file_mod
    import tools.execution_targets as targets_mod
    import tools.terminal_tool as terminal_mod

    targets_mod.set_execution_target_config_source(None)

    monkeypatch.setattr(terminal_mod, "_active_environments", {})
    monkeypatch.setattr(terminal_mod, "_last_activity", {})
    monkeypatch.setattr(terminal_mod, "_creation_locks", {})
    monkeypatch.setattr(terminal_mod, "_session_cwd", {})
    monkeypatch.setattr(terminal_mod, "_session_cwd_specs", {})
    monkeypatch.setattr(terminal_mod, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_mod, "_active_turn_counts", {})
    monkeypatch.setattr(terminal_mod, "_deferred_environment_cleanups", {})
    monkeypatch.setattr(terminal_mod, "_retired_environments", [])
    monkeypatch.setattr(terminal_mod, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(file_mod, "_file_ops_cache", {})
    read_tracker = {}
    patch_failures = {}
    monkeypatch.setattr(read_tracking, "_read_tracker", read_tracker)
    monkeypatch.setattr(file_mod, "_read_tracker", read_tracker)
    monkeypatch.setattr(read_tracking, "_patch_failure_tracker", patch_failures)
    monkeypatch.setattr(file_mod, "_patch_failure_tracker", patch_failures)
    monkeypatch.setattr(
        terminal_mod,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )
    yield terminal_mod, file_mod
    targets_mod.set_execution_target_config_source(None)
    for env in list(terminal_mod._active_environments.values()):
        try:
            env.cleanup()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Named-target fixture cleanup failed")
