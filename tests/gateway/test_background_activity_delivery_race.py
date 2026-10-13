"""A child exit racing original final delivery still needs a verification turn."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.session import SessionSource
from gateway.run_turn_runner import TurnRunner
from tools.delegate_tool_progress import _build_child_progress_callback


@pytest.mark.asyncio
async def test_child_exit_before_original_delivery_cannot_skip_verification(tmp_path):
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated-bot", extra={"base_url": "http://127.0.0.1"}
        )
    )
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    key = adapter._source_session_key(source)
    turn = adapter.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
        event_message_id="opening",
    )
    ctx = SimpleNamespace(
        _background_activity=turn,
        _structured_progress_supported=False,
        _run_still_current=lambda: False,
    )
    observer = TurnRunner(None, ctx)
    relay = _build_child_progress_callback(
        0,
        "private goal",
        SimpleNamespace(tool_progress_callback=observer.activity_only_progress),
        subagent_id="child",
    )
    relay("subagent.start", delegation_id="original-completion")
    turn.parent_returned()
    relay("subagent.complete", status="completed")
    await adapter.on_processing_complete(
        MessageEvent(source=source, text="parent reply", message_id="opening"),
        ProcessingOutcome.SUCCESS,
    )
    assert turn.snapshot()["parent"]["state"] == "verifying", (
        "An exit racing original delivery is not a parent verification execution"
    )
    assert adapter.activity_tasks_for_completion({
        "session_key": key,
        "parent_session_id": "session-A",
        "delegation_id": "original-completion",
    }) == (turn.task_id,)
    verify = adapter.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        verification=True,
        task_ids=(turn.task_id,),
        event_message_id="followup",
    )
    assert verify.execution_id != turn.execution_id
    verify.parent_returned()
    await adapter.on_processing_complete(
        MessageEvent(
            source=source, text="verification", message_id="followup", internal=True
        ),
        ProcessingOutcome.SUCCESS,
    )
    assert turn.snapshot()["parent"]["state"] == "finished"
