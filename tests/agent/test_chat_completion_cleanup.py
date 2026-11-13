"""Per-turn cleanup: headed browsers are left to the idle reaper; the VM is always released."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("headed", [False, True])
def test_browser_cleanup_follows_headed_mode(monkeypatch, headed):
    from agent import chat_completion_helpers
    from tools import browser_session_cloud, terminal_tool_target_lifecycle

    runtime = SimpleNamespace(cleanup_vm=Mock(), cleanup_browser=Mock())
    monkeypatch.setattr(chat_completion_helpers, "_ra", lambda: runtime)
    monkeypatch.setattr(browser_session_cloud, "is_headed_mode", lambda: headed)
    monkeypatch.setattr(terminal_tool_target_lifecycle, "release_logical_environment_turn_for_cleanup", lambda _task: None)
    monkeypatch.setattr(terminal_tool_target_lifecycle, "active_environment_turns", lambda _task: 0)

    chat_completion_helpers.cleanup_task_resources(SimpleNamespace(verbose_logging=False), "task")

    runtime.cleanup_vm.assert_called_once_with("task", preserve_persistent=True, include_collapsed=True)
    assert runtime.cleanup_browser.call_count == (0 if headed else 1)
