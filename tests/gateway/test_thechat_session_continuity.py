from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_thechat_adapter_normalizes_session_payload_from_event_context():
    adapter = TheChatAdapter(PlatformConfig())
    context = {
        "event": SimpleNamespace(
            hermes_session={
                "session_id": "session-1",
                "session_key": "key-1",
                "lineage_root_id": "root-1",
            },
        ),
    }

    payload = adapter._session_payload_for_context(
        context,
        reason="message.delivered",
    )

    assert payload == {
        "sessionId": "session-1",
        "sessionKey": "key-1",
        "lineageRootId": "root-1",
        "reason": "message.delivered",
        "source": "hermes",
    }
    assert context["session"] == payload


def test_thechat_adapter_prefers_live_event_session_over_cached_context_session():
    adapter = TheChatAdapter(PlatformConfig())
    context = {
        "session": {
            "sessionId": "stale-parent",
            "sessionKey": "key-1",
            "lineageRootId": "root-1",
        },
        "event": SimpleNamespace(
            hermes_session={
                "sessionId": "compressed-child",
                "sessionKey": "key-1",
                "lineageRootId": "root-1",
            },
        ),
    }

    payload = adapter._session_payload_for_context(
        context,
        reason="invocation.completed",
    )

    assert payload is not None
    assert payload["sessionId"] == "compressed-child"
    assert payload["reason"] == "invocation.completed"
    assert context["session"] == payload


def test_thechat_continuity_session_id_switches_gateway_session():
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(
        resolve_resume_session_id=lambda session_id: f"{session_id}-tip",
        get_session=lambda session_id: {"id": session_id},
        _session_lineage_root_to_tip=lambda session_id: ["root-1", session_id],
    )
    switched_entry = SimpleNamespace(
        session_id="session-2-tip",
        session_key="thechat-key",
    )
    runner.session_store = SimpleNamespace(
        switch_session=lambda session_key, target_session_id: switched_entry,
    )
    runner._clear_session_boundary_security_state = lambda session_key: None
    runner._evict_cached_agent = lambda session_key: None

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="thechat:conversation:1",
        user_id="user-1",
    )
    event = MessageEvent(
        text="continue",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"continuity": {"sessionId": "session-2"}},
    )
    current_entry = SimpleNamespace(
        session_id="session-1",
        session_key="thechat-key",
    )

    result = runner._apply_platform_session_continuity(
        event,
        source,
        current_entry,
    )

    assert result is switched_entry
    assert event.hermes_session["sessionId"] == "session-2-tip"
    assert event.hermes_session["lineageRootId"] == "root-1"
    assert event.hermes_session["reason"] == "continuity.resumed"


def test_thechat_pending_branch_creates_branch_session_before_agent_run():
    runner = object.__new__(GatewayRunner)
    created = {}
    appended = []
    titled = {}

    def create_session(**kwargs):
        created.update(kwargs)

    def append_message(**kwargs):
        appended.append(kwargs)

    runner._session_db = SimpleNamespace(
        resolve_resume_session_id=lambda session_id: session_id,
        get_session=lambda session_id: {"id": session_id},
        get_session_title=lambda session_id: "Original",
        get_next_title_in_lineage=lambda title: f"{title} #2",
        create_session=create_session,
        append_message=append_message,
        set_session_title=lambda session_id, title: titled.update({session_id: title}),
        _session_lineage_root_to_tip=lambda session_id: ["parent-1", session_id],
    )
    runner.config = {}

    def switch_session(session_key, target_session_id):
        return SimpleNamespace(
            session_id=target_session_id,
            session_key=session_key,
        )

    runner.session_store = SimpleNamespace(
        load_transcript=lambda session_id: [{"role": "user", "content": "hello"}],
        switch_session=switch_session,
    )
    runner._clear_session_boundary_security_state = lambda session_key: None
    runner._evict_cached_agent = lambda session_key: None

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="thechat:conversation:1",
        user_id="user-1",
    )
    event = MessageEvent(
        text="try another approach",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={
            "continuity": {
                "branchFromSessionId": "parent-1",
                "branchTitle": "Alternative",
            },
        },
    )
    current_entry = SimpleNamespace(
        session_id="old-session",
        session_key="thechat-key",
    )

    result = runner._apply_platform_session_continuity(
        event,
        source,
        current_entry,
    )

    assert result.session_key == "thechat-key"
    assert result.session_id != "old-session"
    assert created["parent_session_id"] == "parent-1"
    assert created["session_id"] == result.session_id
    assert appended[0]["session_id"] == result.session_id
    assert titled[result.session_id] == "Alternative"
    assert event.hermes_session["sessionId"] == result.session_id
    assert event.hermes_session["reason"] == "branch.created"
