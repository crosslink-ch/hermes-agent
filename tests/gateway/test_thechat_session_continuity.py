from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _RecordingResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"ok": True, "messageId": "msg-1"}


class _RecordingClient:
    def __init__(self):
        self.posts = []

    async def post(self, path, *, json=None):
        self.posts.append((path, json))
        return _RecordingResponse()


@pytest.mark.asyncio
async def test_thechat_adapter_does_not_send_session_payload_on_messages():
    adapter = TheChatAdapter(PlatformConfig())
    client = _RecordingClient()
    adapter._client = client  # type: ignore[assignment]
    adapter._contexts["chat-1"] = {
        "conversation_id": "conv-1",
        "invocation_id": "inv-1",
        "session": {"sessionId": "session-1", "sessionKey": "key-1"},
    }

    result = await adapter.send(
        "chat-1",
        "hello",
        metadata={"session": {"sessionId": "session-2", "sessionKey": "key-1"}},
    )

    assert result.success is True
    assert client.posts[-1][0] == "/hermes-platform/messages"
    assert "session" not in client.posts[-1][1]


@pytest.mark.asyncio
async def test_thechat_adapter_title_progress_only_sends_title():
    adapter = TheChatAdapter(PlatformConfig())
    client = _RecordingClient()
    adapter._client = client  # type: ignore[assignment]
    context = {"conversation_id": "conv-1", "invocation_id": "inv-1", "bot_id": "bot-1"}

    result = await adapter.send_session_title_update(
        "chat-1",
        "Investigate checkout",
        "session-1",
        session_key="key-1",
        metadata={"session": {"sessionId": "session-1", "sessionKey": "key-1"}},
        context=context,
    )

    assert result.success is True
    path, payload = client.posts[-1]
    assert path == "/hermes-platform/invocations/inv-1/progress"
    assert "session" not in payload
    assert payload["type"] == "session.title"
    assert payload["payload"] == {"title": "Investigate checkout"}


@pytest.mark.asyncio
async def test_thechat_ignores_non_current_session_intent_shapes():
    runner = object.__new__(GatewayRunner)
    runner._session_db = None
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="thechat:conversation:1",
        user_id="user-1",
    )
    current_entry = SimpleNamespace(
        session_id="current-session",
        session_key="thechat-key",
    )

    for raw_message in (
        {"session_intent": {"type": "branch", "fromThreadId": "source-thread"}},
        {"sessionIntent": {"type": "fork", "fromThreadId": "source-thread"}},
        {"sessionIntent": {"type": "resume", "sessionId": "old-session"}},
    ):
        event = MessageEvent(
            text="continue",
            message_type=MessageType.TEXT,
            source=source,
            raw_message=raw_message,
        )

        result = await runner._apply_thechat_session_intent_async(
            event,
            source,
            current_entry,
        )

        assert result is current_entry
        assert not hasattr(event, "hermes_session")


@pytest.mark.asyncio
async def test_thechat_branch_resolves_parent_from_current_source_thread_contract():
    runner = object.__new__(GatewayRunner)
    created = {}
    appended = []
    titled = {}

    runner._session_db = SimpleNamespace(
        resolve_resume_session_id=lambda session_id: session_id,
        get_session=lambda session_id: {"id": session_id},
        get_session_title=lambda session_id: "Original",
        get_next_title_in_lineage=lambda title: f"{title} #2",
        create_session=lambda **kwargs: created.update(kwargs),
        append_message=lambda **kwargs: appended.append(kwargs),
        set_session_title=lambda session_id, title: titled.update({session_id: title}),
        _session_lineage_root_to_tip=lambda session_id: ["parent-thread-session", session_id],
    )
    runner.config = {}
    parent_sources = []

    def get_session_for_source(source):
        parent_sources.append(source)
        return SimpleNamespace(session_id="parent-thread-session")

    def switch_session(session_key, target_session_id):
        return SimpleNamespace(
            session_id=target_session_id,
            session_key=session_key,
        )

    runner.session_store = SimpleNamespace(
        get_session_for_source=get_session_for_source,
        load_transcript=lambda session_id: [{"role": "user", "content": "hello"}],
        switch_session=switch_session,
    )
    runner._clear_session_boundary_security_state = lambda session_key: None
    runner._evict_cached_agent = lambda session_key: None

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="thechat:conversation:1",
        user_id="user-1",
        thread_id="branch-thread",
    )
    event = MessageEvent(
        text="try another approach",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={
            "sessionIntent": {
                "type": "branch",
                "fromThreadId": "source-thread",
                "title": "Alternative",
            },
        },
    )
    current_entry = SimpleNamespace(
        session_id="placeholder-session",
        session_key="branch-thread-key",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 1, 12, 0, 0),
    )

    result = await runner._apply_thechat_session_intent_async(
        event,
        source,
        current_entry,
    )

    assert parent_sources[0].thread_id == "source-thread"
    assert created["parent_session_id"] == "parent-thread-session"
    assert created["model_config"] == {"_branched_from": "parent-thread-session"}
    assert created["session_id"] == result.session_id
    assert appended[0]["session_id"] == result.session_id
    assert titled[result.session_id] == "Alternative"
    assert result.session_key == "branch-thread-key"
    session_reference = getattr(event, "hermes_session")
    assert session_reference["sessionId"] == result.session_id
    assert session_reference["reason"] == "branch.created"


@pytest.mark.asyncio
async def test_thechat_persisted_branch_marker_is_ignored_after_branch_exists():
    runner = object.__new__(GatewayRunner)
    runner._session_db = None
    runner.session_store = SimpleNamespace(
        get_session_for_source=lambda source: (_ for _ in ()).throw(
            AssertionError("existing branch thread must not re-resolve parent")
        )
    )

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="thechat:conversation:1",
        user_id="user-1",
        thread_id="branch-thread",
    )
    event = MessageEvent(
        text="continue",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={
            "sessionIntent": {
                "type": "branch",
                "fromThreadId": "source-thread",
            },
        },
    )
    current_entry = SimpleNamespace(
        session_id="already-branched-session",
        session_key="branch-thread-key",
        created_at=datetime(2026, 1, 1, 12, 0, 0),
        updated_at=datetime(2026, 1, 1, 12, 5, 0),
    )

    result = await runner._apply_thechat_session_intent_async(
        event,
        source,
        current_entry,
    )

    assert result is current_entry
    assert not hasattr(event, "hermes_session")
