"""A TheChat branch thread starts from its parent's conversation on the production path.

The real gateway admits every messaging turn through the session authority's durable FIFO, which
creates the thread's session before the turn runs and replays the turn from the stored payload
(without the platform's raw event). The branch has to be applied at admission, while the original
event and its ``sessionIntent`` are still in hand; applied later it never fires.

Real GatewayRunner, SessionAuthority, session store and SQLite; the model call is stubbed.
"""

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_authority import initialize_session_authority
from tests.e2e.test_gateway_core_session_flows import (
    dispatch,
    make_runner_and_adapter,
    session_db_call,
    transcript_contents,
)

pytestmark = pytest.mark.asyncio

BRANCH_INTENT = {
    "sessionIntent": {
        "type": "branch",
        "fromThreadId": "source-thread",
        "title": "Child Branch",
    }
}


def thechat_event(adapter, text, *, thread_id, message_id, raw_message=None):
    """An inbound event whose source the adapter built, as the admission's provenance requires."""
    source = adapter.build_source(
        chat_id="e2e-chat-1", user_id="e2e-user-1", user_name="E2E Tester", chat_type="dm",
        thread_id=thread_id)
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source,
                        message_id=message_id, raw_message=raw_message or {})


async def test_thechat_branch_intent_applies_through_session_authority_admission(monkeypatch):
    runner, adapter = make_runner_and_adapter(monkeypatch)
    runner._is_user_authorized = lambda source, **_: True  # the admission path passes keywords
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="e2e-branch")
    runner._wire_adapter_handlers(adapter)

    histories: dict[str, list] = {}
    stub_run_agent = runner._run_agent

    async def recording_run_agent(**kwargs):
        histories[kwargs["message"]] = list(kwargs.get("history") or [])
        return await stub_run_agent(**kwargs)

    runner._run_agent = recording_run_agent

    try:
        parent_event = thechat_event(
            adapter, "parent hello", thread_id="source-thread", message_id="parent-message-1")
        assert await dispatch(adapter, parent_event) == "e2e echo: parent hello"
        parent_entry = runner.session_store.get_session_for_source(parent_event.source)
        assert parent_entry is not None

        branch_event = thechat_event(
            adapter, "child first message", thread_id="branch-thread", message_id="branch-message-1",
            raw_message=BRANCH_INTENT)
        assert await dispatch(adapter, branch_event) == "e2e echo: child first message"

        # The model saw the parent's conversation on the branch's first turn.
        first_history = [(m.get("role"), m.get("content")) for m in histories["child first message"]]
        assert ("user", "parent hello") in first_history
        assert ("assistant", "e2e echo: parent hello") in first_history

        child_entry = runner.session_store.get_session_for_source(branch_event.source)
        assert child_entry is not None and child_entry.session_id != parent_entry.session_id
        child_row = await session_db_call(runner, "get_session", child_entry.session_id)
        assert child_row["parent_session_id"] == parent_entry.session_id
        assert child_row["thread_id"] == "branch-thread"
        assert await session_db_call(runner, "get_session_title", child_entry.session_id) == "Child Branch"

        # TheChat keeps sending the branch marker on every message of the thread; it is applied once.
        second_event = thechat_event(
            adapter, "child second message", thread_id="branch-thread", message_id="branch-message-2",
            raw_message=BRANCH_INTENT)
        assert await dispatch(adapter, second_event) == "e2e echo: child second message"
        assert runner.session_store.get_session_for_source(branch_event.source).session_id == child_entry.session_id

        child_transcript = transcript_contents(runner, child_entry.session_id)
        assert child_transcript.count(("user", "parent hello")) == 1
        assert ("user", "child first message") in child_transcript
        assert ("user", "child second message") in child_transcript
        assert ("user", "child first message") not in transcript_contents(runner, parent_entry.session_id)
    finally:
        authority.db.close()
