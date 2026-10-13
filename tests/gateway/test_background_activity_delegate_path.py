"""Real delegate_task production lifecycle with an offline child model."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.run_turn_runner import TurnRunner


def test_actual_delegate_task_child_keeps_reporting_after_parent_reply(
    tmp_path, monkeypatch
):
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task
    from tools import async_delegation as ad
    from gateway.session_context import set_session_vars, clear_session_vars
    from agent.tool_executor import execute_tool_calls_sequential
    from tests.agent.test_run_agent import _mock_assistant_msg, _mock_tool_call

    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated-bot", extra={"base_url": "http://127.0.0.1"}
        )
    )
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="agent:main:thechat:dm:owned",
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
    )
    ctx = SimpleNamespace(
        _background_activity=turn,
        _structured_progress_supported=False,
        _run_still_current=lambda: False,
    )
    observer = TurnRunner(None, ctx)
    ready, release, reported = (threading.Event() for _ in range(3))
    with patch("agent.process_bootstrap.OpenAI"):
        parent = AIAgent(
            api_key="isolated-test",
            model="offline-test",
            base_url="http://127.0.0.1:1",
            enabled_toolsets=["terminal", "delegation"],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            session_id="session-A",
        )
    parent.tool_progress_callback = (
        lambda event, name=None, preview=None, args=None, **kw: (
            observer._record_structured_tool_progress(event, name, preview, **kw)
        )
    )

    def offline_child(self, *args, **kwargs):
        ready.set()
        assert release.wait(10)
        calls = _mock_assistant_msg(
            tool_calls=[
                _mock_tool_call(
                    "terminal",
                    json.dumps({"command": "printf offline-child-result"}),
                    "native-child-call",
                )
            ]
        )
        messages = []
        execute_tool_calls_sequential(
            self, calls, messages, kwargs.get("task_id", "child"), finalize=False
        )
        assert "offline-child-result" in str(messages[-1]["content"])
        reported.set()
        return {
            "final_response": "offline-child-result",
            "completed": True,
            "messages": messages,
            "api_calls": 1,
        }

    monkeypatch.setattr(AIAgent, "run_conversation", offline_child)
    tokens = set_session_vars(
        platform="thechat",
        chat_id="owned",
        chat_type="dm",
        session_key="agent:main:thechat:dm:owned",
        session_id="session-A",
        async_delivery=True,
    )
    ad._reset_for_tests()
    try:
        handle = json.loads(
            delegate_task(
                goal="offline native child acceptance",
                parent_agent=parent,
                background=True,
            )
        )
        assert handle["status"] == "dispatched", handle
        assert ready.wait(5)
        turn.parent_returned()
        turn.finish()
        assert turn.snapshot()["parent"]["state"] == "waiting_worker"
        assert turn.snapshot()["workers"][0]["state"] == "running"
        release.set()
        assert reported.wait(5)
        # The lifecycle callback follows the fake model's return; wait on the real async ledger.
        from tools.process_registry import process_registry

        while True:
            event = process_registry.completion_queue.get(timeout=10)
            if event.get("delegation_id") == handle["delegation_id"]:
                break
        assert event["status"] == "completed", event
        snapshot = turn.snapshot()
        assert snapshot["parent"]["state"] == "verifying"
        assert snapshot["workers"][0]["state"] == "finished"
        assert snapshot["activities"][0]["state"] == "finished"
        assert snapshot["activities"][0]["workerId"] == snapshot["workers"][0]["id"]
        assert a.activity_tasks_for_completion(event) == (turn.task_id,)
    finally:
        release.set()
        reported.wait(5)
        clear_session_vars(tokens)
        parent.close()
        ad._reset_for_tests()
