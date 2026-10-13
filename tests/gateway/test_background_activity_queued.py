"""Queued completion verification must retain its binding through delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import pytest
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


@pytest.mark.asyncio
async def test_inband_completion_reuses_task_and_defers_terminal_until_delivery(
    tmp_path,
):
    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    key = a._source_session_key(source)
    first = a.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
        event_message_id="opening",
    )
    first.event("subagent.start", subagent_id="worker")
    first.finish()
    first.event("subagent.complete", subagent_id="worker", status="completed")
    from tests.gateway.test_discord_triggering_note_persistence import _runner

    gateway = _runner()
    gateway.hooks = SimpleNamespace(loaded_hooks=False)
    gateway._MAX_INTERRUPT_DEPTH = 8
    gateway._run_agent_deliver_first_response = AsyncMock()
    gateway._is_goal_continuation_event = lambda event: False
    gateway._session_key_for_source = lambda source: key
    gateway._prepare_profile_scoped_inbound_message_text = AsyncMock(
        return_value="verify"
    )
    gateway._reply_anchor_for_event = lambda event: event.message_id
    gateway._delivery_adapter_for = lambda source: a
    gateway._intake_adapter_for = lambda source: a
    gateway._resolve_profile_home_for_source = lambda source: tmp_path
    gateway._run_agent_progress_threading = lambda *args: (None, None, None)
    gateway._refresh_agent_cache_message_count = AsyncMock()
    verification = []

    async def run(**kwargs):
        assert kwargs.get("background_activity_task_ids") == (first.task_id,), (
            "Queued completion loses original binding before recursive turn"
        )
        ctx = TurnContext(
            source=source,
            session_key=key,
            session_id="session-A",
            persist_user_display_kind="internal_notification",
            background_activity_task_ids=kwargs["background_activity_task_ids"],
        )
        gateway._run_agent_bind_turn_wiring(
            ctx, TurnRunner(gateway, ctx), source, kwargs["event_message_id"], False
        )
        verification.append(ctx._background_activity)
        ctx._background_activity.parent_returned()
        return {"final_response": "verified", "messages": [], "completed": True}

    gateway._run_agent = run
    outer = SimpleNamespace(
        source=source,
        session_id="session-A",
        session_key=key,
        run_generation=1,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata=None,
        context_prompt="",
        result_holder=[None],
    )
    pending = MessageEvent(
        text="verify", source=source, message_id="completion", internal=True
    )
    pending._background_activity_task_ids = (first.task_id,)
    await GatewayRunner._run_agent_queued_followup(
        gateway,
        outer,
        a,
        "verify",
        pending,
        "parent reply",
        {"interrupted": True, "messages": []},
        None,
    )
    assert first.snapshot()["parent"]["state"] == "verifying", (
        "In-band processing hook is not final delivery"
    )
    assert verification[0].execution_id != first.execution_id
    await a.on_processing_complete(
        MessageEvent(text="opening", source=source, message_id="opening"),
        ProcessingOutcome.SUCCESS,
    )
    assert first.snapshot()["parent"]["state"] == "finished"
