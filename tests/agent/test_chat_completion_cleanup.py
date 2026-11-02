"""Headed-browser cleanup fallback parses a boolean, not token presence."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("value,headed", [("false", False), ("0", False), ("", False), ("true", True), ("yes", True)])
def test_browser_cleanup_fallback_respects_boolean_value(monkeypatch, value, headed):
    from agent import chat_completion_helpers
    from tools import browser_tool_cloud, terminal_tool_target_lifecycle

    def optional_browser_unavailable():
        raise ImportError("optional browser integration unavailable")

    runtime = SimpleNamespace(cleanup_vm=Mock(), cleanup_browser=Mock())
    monkeypatch.setattr(chat_completion_helpers, "_ra", lambda: runtime)
    monkeypatch.setattr(browser_tool_cloud, "_is_headed_mode", optional_browser_unavailable)
    monkeypatch.setattr(terminal_tool_target_lifecycle, "release_logical_environment_turn_for_cleanup", lambda _task: None)
    monkeypatch.setattr(terminal_tool_target_lifecycle, "active_environment_turns", lambda _task: 0)
    monkeypatch.setenv("AGENT_BROWSER_HEADED", value)

    chat_completion_helpers.cleanup_task_resources(SimpleNamespace(verbose_logging=False), "task")

    runtime.cleanup_vm.assert_called_once_with("task", preserve_persistent=True, include_collapsed=True)
    assert runtime.cleanup_browser.call_count == (0 if headed else 1)
