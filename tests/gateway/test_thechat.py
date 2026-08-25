import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, cast

import httpx
import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.platforms import thechat
from gateway.platforms.thechat import TheChatAdapter


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
INVOCATION_ID = "22222222-2222-4222-8222-222222222222"
OTHER_INVOCATION_ID = "33333333-3333-4333-8333-333333333333"
OTHER_CONVERSATION_ID = "44444444-4444-4444-8444-444444444444"
SESSION_KEY = "agent:main:thechat:dm:chat-1"


class _FakeResponse:
    def __init__(self, data=None):
        self._data = data or {"messageId": "thechat-message-1"}
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self):
        self.posts = []
        self.closed = False

    async def get(self, path, **_kwargs):
        return _FakeResponse({"ok": True})

    async def post(self, path, json):
        self.posts.append({"path": path, "json": json})
        if path == "/bots/me/webhook":
            return _FakeResponse({"webhookSecret": "whsec-test"})
        return _FakeResponse()

    async def aclose(self):
        self.closed = True


class _PollingClient(_FakeClient):
    def __init__(self, event):
        super().__init__()
        self.event = event
        self.gets = []

    async def get(self, path, **kwargs):
        self.gets.append({"path": path, **kwargs})
        if path == "/hermes-platform/events":
            return _FakeResponse({"events": [self.event]})
        return _FakeResponse({"ok": True})


def _make_adapter():
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={
                "base_url": "http://thechat.test",
                "webhook_url": "http://gateway.test/thechat/webhook",
            },
        )
    )
    adapter._client = cast(httpx.AsyncClient, _FakeClient())
    return adapter


async def _drain_webhook_tasks(adapter):
    while adapter._webhook_tasks:
        await asyncio.gather(*list(adapter._webhook_tasks))


def test_config_loads_only_current_thechat_token_location(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("THECHAT_BASE_URL", "https://thechat.test")
    monkeypatch.setenv("THECHAT_HERMES_BOT_TOKEN", "obsolete-token")
    monkeypatch.delenv("THECHAT_BOT_TOKEN", raising=False)

    config = load_gateway_config()

    assert Platform.THECHAT not in config.platforms

    monkeypatch.setenv("THECHAT_BOT_TOKEN", "current-token")
    config = load_gateway_config()
    thechat_config = config.platforms[Platform.THECHAT]

    assert thechat_config.token == "current-token"
    assert thechat_config.extra["base_url"] == "https://thechat.test"
    assert "token" not in thechat_config.extra


def _context(invocation_id):
    return {
        "invocation_id": invocation_id,
        "bot_id": "bot-1",
        "bot_name": "Hermes",
        "conversation_id": CONVERSATION_ID,
        "conversation_name": "Hermes DM",
        "chat_type": "dm",
        "thread_id": None,
    }


def _event(
    adapter,
    message_id="message-1",
    chat_id=CONVERSATION_ID,
    thread_id=None,
):
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="dm",
        user_id="user-1",
        user_name="User",
        message_id=message_id,
        thread_id=thread_id,
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


def _platform_item(
    *,
    text="hello from TheChat",
    thread_id=None,
):
    return {
        "id": INVOCATION_ID,
        "chatId": CONVERSATION_ID,
        "chatType": "dm",
        "threadId": thread_id,
        "invocationId": INVOCATION_ID,
        "messageId": "message-1",
        "text": text,
        "sender": {"id": "user-1", "name": "User"},
        "bot": {"id": "bot-1", "userId": "bot-user-1", "name": "Hermes"},
        "conversation": {
            "id": CONVERSATION_ID,
            "type": "direct",
            "name": "Hermes DM",
            "workspaceId": "workspace-1",
        },
    }


class _WebhookRequest:
    def __init__(self, body, headers):
        self._body = body
        self.headers = headers

    async def read(self):
        return self._body.encode()


def _signed_webhook_request(adapter, payload, *, timestamp=None):
    body = json.dumps(payload, separators=(",", ":"))
    timestamp = int(time.time()) if timestamp is None else timestamp
    signature = hmac.new(
        adapter._webhook_secret.encode(),
        f"{timestamp}.{body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return _WebhookRequest(
        body,
        {
            "X-Webhook-Timestamp": str(timestamp),
            "X-Webhook-Signature": signature,
        },
    )


def _interaction_payload(
    *,
    interaction_id="progress-event-1",
    request_type="approval.request",
    request_id="approval-request-1",
    invocation_id=INVOCATION_ID,
    conversation_id=CONVERSATION_ID,
    thread_id=None,
    session_key=SESSION_KEY,
    response="once",
):
    return {
        "type": "thechat.hermes_platform.interaction",
        "interaction": {
            "id": interaction_id,
            "requestType": request_type,
            "requestId": request_id,
            "invocationId": invocation_id,
            "conversationId": conversation_id,
            "threadId": thread_id,
            "sessionKey": session_key,
            "response": response,
        },
    }


@pytest.mark.asyncio
async def test_send_uses_reply_to_context_instead_of_latest_chat_context():
    adapter = _make_adapter()
    first = _context("invocation-first")
    second = _context("invocation-second")
    adapter._contexts[CONVERSATION_ID] = second
    adapter._event_contexts["message-first"] = first
    adapter._event_contexts["message-second"] = second

    result = await adapter.send(CONVERSATION_ID, "first response", reply_to="message-first")

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"] == {
        "conversationId": CONVERSATION_ID,
        "content": "first response",
        "attachmentIds": [],
        "invocationId": "invocation-first",
        "botId": "bot-1",
    }
    assert first["delivered"] is True
    assert "delivered" not in second


@pytest.mark.asyncio
async def test_send_does_not_use_notify_metadata_as_completion_signal():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts[CONVERSATION_ID] = context

    result = await adapter.send(CONVERSATION_ID, "final response", metadata={"notify": True})

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"]["attachmentIds"] == []


@pytest.mark.asyncio
async def test_send_can_post_without_invocation_context():
    adapter = _make_adapter()
    chat_id = CONVERSATION_ID

    result = await adapter.send(chat_id, "cron says hello")

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"] == {
        "conversationId": chat_id,
        "content": "cron says hello",
        "attachmentIds": [],
    }


@pytest.mark.asyncio
async def test_send_rejects_obsolete_composite_chat_id():
    adapter = _make_adapter()

    result = await adapter.send(
        "thechat:workspace:workspace-1:conversation:conversation-1:bot:bot-1",
        "obsolete target",
    )

    assert result.success is False
    assert result.retryable is False
    assert "conversation UUID" in (result.error or "")
    assert cast(_FakeClient, adapter._client).posts == []


@pytest.mark.asyncio
async def test_send_keeps_thread_metadata_after_invocation_context_cleanup():
    adapter = _make_adapter()
    context = _context("invocation-original")
    context["thread_id"] = "thread-1"
    context_key = adapter._context_key(CONVERSATION_ID, "thread-1")
    adapter._contexts[context_key] = context
    adapter._event_contexts["message-original"] = context

    await adapter.on_processing_complete(
        _event(
            adapter,
            message_id="message-original",
            chat_id=CONVERSATION_ID,
            thread_id="thread-1",
        ),
        ProcessingOutcome.SUCCESS,
    )

    assert context_key not in adapter._contexts
    assert "message-original" not in adapter._event_contexts
    client = cast(_FakeClient, adapter._client)
    client.posts.clear()

    result = await adapter.send(
        CONVERSATION_ID,
        "late clarify response",
        metadata={
            "thread_id": "thread-1",
            "message_id": "message-original",
        },
    )

    assert result.success is True
    assert client.posts[0]["path"] == "/hermes-platform/messages"
    assert client.posts[0]["json"] == {
        "conversationId": CONVERSATION_ID,
        "content": "late clarify response",
        "attachmentIds": [],
        "threadId": "thread-1",
    }


@pytest.mark.asyncio
async def test_processing_cancel_reports_cancelled_without_chat_message():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts[CONVERSATION_ID] = context
    adapter._event_contexts["message-first"] = context

    await adapter.on_processing_complete(
        _event(adapter, message_id="message-first"),
        ProcessingOutcome.CANCELLED,
    )

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/cancelled",
            "json": {"reason": "Hermes gateway cancelled the message"},
        }
    ]
    assert CONVERSATION_ID not in adapter._contexts


@pytest.mark.asyncio
async def test_processing_success_without_delivery_marks_invocation_completed():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts[CONVERSATION_ID] = context
    adapter._event_contexts["message-first"] = context

    await adapter.on_processing_complete(
        _event(adapter, message_id="message-first"),
        ProcessingOutcome.SUCCESS,
    )

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/completed",
            "json": {"reason": "Hermes gateway completed without a chat response"},
        }
    ]


@pytest.mark.asyncio
async def test_processing_success_after_delivery_marks_invocation_completed():
    adapter = _make_adapter()
    context = _context("invocation-first")
    context["delivered"] = True
    adapter._contexts[CONVERSATION_ID] = context
    adapter._event_contexts["message-first"] = context

    await adapter.on_processing_complete(
        _event(adapter, message_id="message-first"),
        ProcessingOutcome.SUCCESS,
    )

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/completed",
            "json": {"reason": "Hermes gateway completed"},
        }
    ]


@pytest.mark.asyncio
async def test_active_stop_command_marks_thechat_command_invocation_completed():
    adapter = _make_adapter()
    stop_context = _context("invocation-stop")
    adapter._contexts[CONVERSATION_ID] = stop_context
    adapter._event_contexts["message-stop"] = stop_context

    async def handle(event):
        assert event.text == "/stop"
        return "Stopped."

    adapter.set_message_handler(handle)
    event = _event(adapter, message_id="message-stop")
    event.text = "/stop"
    event.message_type = MessageType.COMMAND
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/messages",
            "json": {
                "conversationId": CONVERSATION_ID,
                "content": "Stopped.",
                "attachmentIds": [],
                "invocationId": "invocation-stop",
                "botId": "bot-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-stop/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert CONVERSATION_ID not in adapter._contexts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_text", "response_text"),
    [
        ("/approve", "Command approved. The agent is resuming..."),
        ("/steer prefer the smaller fix", "Steer queued."),
    ],
)
async def test_active_non_canceling_commands_mark_thechat_invocation_completed(
    command_text,
    response_text,
):
    adapter = _make_adapter()
    command_context = _context("invocation-command")
    adapter._contexts[CONVERSATION_ID] = command_context
    adapter._event_contexts["message-command"] = command_context

    async def handle(event):
        assert event.text == command_text
        return response_text

    adapter.set_message_handler(handle)
    event = _event(adapter, message_id="message-command")
    event.text = command_text
    event.message_type = MessageType.COMMAND
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/messages",
            "json": {
                "conversationId": CONVERSATION_ID,
                "content": response_text,
                "attachmentIds": [],
                "invocationId": "invocation-command",
                "botId": "bot-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-command/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert CONVERSATION_ID not in adapter._contexts
    assert "message-command" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_active_queue_command_ack_marks_invocation_completed_immediately():
    adapter = _make_adapter()
    command_context = _context("invocation-queue")
    adapter._contexts[CONVERSATION_ID] = command_context
    adapter._event_contexts["message-queue"] = command_context

    async def handle(event):
        queued_event = _event(adapter, message_id=event.message_id)
        queued_event.text = "run tests"
        adapter._pending_messages[session_key] = queued_event
        return "Queued for the next turn."

    adapter.set_message_handler(handle)
    event = _event(adapter, message_id="message-queue")
    event.text = "/queue run tests"
    event.message_type = MessageType.COMMAND
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/messages",
            "json": {
                "conversationId": CONVERSATION_ID,
                "content": "Queued for the next turn.",
                "attachmentIds": [],
                "invocationId": "invocation-queue",
                "botId": "bot-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-queue/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert CONVERSATION_ID not in adapter._contexts
    assert "message-queue" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_busy_command_ack_marks_thechat_invocation_completed():
    adapter = _make_adapter()
    command_context = _context("invocation-command")
    adapter._contexts[CONVERSATION_ID] = command_context
    adapter._event_contexts["message-command"] = command_context

    async def handle(_event):
        return "unused"

    async def busy_handler(event, session_key):
        adapter._pending_messages[session_key] = event
        await adapter._send_with_retry(
            chat_id=event.source.chat_id,
            content="Interrupting current task.",
            reply_to=event.message_id,
            metadata={"message_id": event.message_id},
        )
        return True

    adapter.set_message_handler(handle)
    adapter.set_busy_session_handler(busy_handler)
    event = _event(adapter, message_id="message-command")
    event.text = "/appro e"
    event.message_type = MessageType.COMMAND
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/messages",
            "json": {
                "conversationId": CONVERSATION_ID,
                "content": "Interrupting current task.",
                "attachmentIds": [],
                "invocationId": "invocation-command",
                "botId": "bot-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-command/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert "message-command" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_busy_text_pending_turn_is_not_completed_by_ack():
    adapter = _make_adapter()
    context = _context("invocation-followup")
    adapter._contexts[CONVERSATION_ID] = context
    adapter._event_contexts["message-followup"] = context

    async def handle(_event):
        return "unused"

    async def busy_handler(event, session_key):
        adapter._pending_messages[session_key] = event
        await adapter._send_with_retry(
            chat_id=event.source.chat_id,
            content="Interrupting current task.",
            reply_to=event.message_id,
            metadata={"message_id": event.message_id},
        )
        return True

    adapter.set_message_handler(handle)
    adapter.set_busy_session_handler(busy_handler)
    event = _event(adapter, message_id="message-followup")
    event.text = "please adjust this"
    event.message_type = MessageType.TEXT
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    assert [post["path"] for post in adapter._client.posts] == [
        "/hermes-platform/messages",
    ]
    assert adapter._event_contexts["message-followup"] is context


@pytest.mark.asyncio
async def test_send_invocation_progress_posts_structured_event():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts[CONVERSATION_ID] = context

    result = await adapter.send_invocation_progress(
        CONVERSATION_ID,
        {
            "type": "tool.started",
            "status": "running",
            "toolCallId": "call-1",
            "toolName": "shell",
            "label": "Shell: pytest",
            "payload": {"args": {"command": "pytest"}},
        },
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "tool.started",
                "status": "running",
                "toolCallId": "call-1",
                "toolName": "shell",
                "label": "Shell: pytest",
                "payload": {"args": {"command": "pytest"}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_send_invocation_progress_uses_thread_metadata_context():
    adapter = _make_adapter()
    first = _context("invocation-first")
    first["thread_id"] = "thread-1"
    second = _context("invocation-second")
    second["thread_id"] = "thread-2"
    adapter._contexts[adapter._context_key(CONVERSATION_ID, "thread-1")] = first
    adapter._contexts[adapter._context_key(CONVERSATION_ID, "thread-2")] = second

    result = await adapter.send_invocation_progress(
        CONVERSATION_ID,
        {"type": "tool.started", "status": "running"},
        metadata={"thread_id": "thread-1"},
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "tool.started",
                "status": "running",
                "threadId": "thread-1",
            },
        }
    ]


@pytest.mark.asyncio
async def test_send_invocation_progress_prefers_message_metadata_over_latest_context():
    adapter = _make_adapter()
    original = _context("invocation-original")
    original["thread_id"] = "thread-1"
    command = _context("invocation-command")
    command["thread_id"] = "thread-1"
    adapter._contexts[adapter._context_key(CONVERSATION_ID, "thread-1")] = command
    adapter._contexts[CONVERSATION_ID] = command
    adapter._event_contexts["message-original"] = original
    adapter._event_contexts["message-command"] = command

    result = await adapter.send_invocation_progress(
        CONVERSATION_ID,
        {"type": "tool.started", "status": "running"},
        metadata={"thread_id": "thread-1", "message_id": "message-original"},
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-original/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "tool.started",
                "status": "running",
                "threadId": "thread-1",
            },
        }
    ]


@pytest.mark.asyncio
async def test_send_session_title_update_posts_structured_event_without_session_payload():
    adapter = _make_adapter()
    context = _context("invocation-title")
    context["thread_id"] = "thread-1"
    adapter._contexts[adapter._context_key(CONVERSATION_ID, "thread-1")] = context

    result = await adapter.send_session_title_update(
        CONVERSATION_ID,
        "Generated Task Title",
        metadata={"thread_id": "thread-1"},
    )

    assert result.success is True
    client = cast(_FakeClient, adapter._client)
    assert client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-title/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "session.title",
                "status": "completed",
                "label": "Generated Task Title",
                "preview": "Generated Task Title",
                "payload": {
                    "title": "Generated Task Title",
                },
                "threadId": "thread-1",
            },
        }
    ]


@pytest.mark.asyncio
async def test_send_exec_approval_posts_structured_approval_request():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts[CONVERSATION_ID] = context

    result = await adapter.send_exec_approval(
        CONVERSATION_ID,
        "rm -rf /important",
        session_key=SESSION_KEY,
        description="recursive delete",
        request_id="approval-request-1",
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "approval.request",
                "status": "waiting",
                "label": "Command approval required",
                "preview": "rm -rf /important",
                "payload": {
                    "requestId": "approval-request-1",
                    "command": "rm -rf /important",
                    "description": "recursive delete",
                    "sessionKey": SESSION_KEY,
                    "choices": ["once", "session", "always", "deny"],
                },
            },
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("approval_kwargs", "expected_choices"),
    [
        (
            {
                "allow_permanent": False,
                "allow_session": True,
                "smart_denied": False,
            },
            ["once", "session", "deny"],
        ),
        (
            {
                "allow_permanent": True,
                "allow_session": False,
                "smart_denied": False,
            },
            ["once", "deny"],
        ),
        (
            {
                "allow_permanent": True,
                "allow_session": True,
                "smart_denied": True,
            },
            ["once", "deny"],
        ),
    ],
)
async def test_send_exec_approval_honors_gateway_choice_policy(
    approval_kwargs,
    expected_choices,
):
    adapter = _make_adapter()
    adapter._contexts[CONVERSATION_ID] = _context("invocation-policy")

    result = await adapter.send_exec_approval(
        CONVERSATION_ID,
        "rm -rf /important",
        session_key="agent:main:thechat:dm:chat-1",
        **approval_kwargs,
    )

    assert result.success is True
    client = cast(_FakeClient, adapter._client)
    assert client.posts[0]["json"]["payload"]["choices"] == expected_choices


@pytest.mark.asyncio
async def test_interaction_request_retries_ambiguous_progress_with_same_id():
    class _AmbiguousProgressClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def post(self, path, json):
            self.posts.append({"path": path, "json": json})
            if json.get("type") == "approval.request":
                self.attempts += 1
                if self.attempts == 1:
                    raise httpx.ReadTimeout("ambiguous progress response")
            return _FakeResponse()

    adapter = _make_adapter()
    client = _AmbiguousProgressClient()
    adapter._client = cast(httpx.AsyncClient, client)
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)

    result = await adapter.send_exec_approval(
        CONVERSATION_ID,
        "pwd",
        session_key=SESSION_KEY,
        request_id="stable-progress-request",
    )

    assert result.success is True
    assert len(client.posts) == 2
    assert client.posts[0] == client.posts[1]
    assert (
        client.posts[0]["json"]["payload"]["requestId"]
        == "stable-progress-request"
    )
    assert "stable-progress-request" in adapter._interaction_requests


@pytest.mark.asyncio
async def test_send_clarify_posts_structured_request_and_keeps_typed_fallback():
    from tools import clarify_gateway

    adapter = _make_adapter()
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    clarify_gateway.register(
        clarify_id="clarify-request-1",
        session_key=SESSION_KEY,
        question="Choose targets",
        choices=["alpha", "beta"],
        multi_select=True,
    )
    try:
        result = await adapter.send_clarify(
            CONVERSATION_ID,
            "Choose targets",
            ["alpha", "beta"],
            "clarify-request-1",
            SESSION_KEY,
        )

        assert result.success is True
        client = cast(_FakeClient, adapter._client)
        assert client.posts == [
            {
                "path": f"/hermes-platform/invocations/{INVOCATION_ID}/progress",
                "json": {
                    "botId": "bot-1",
                    "conversationId": CONVERSATION_ID,
                    "type": "clarify.request",
                    "status": "waiting",
                    "label": "Input required",
                    "preview": "Choose targets",
                    "payload": {
                        "requestId": "clarify-request-1",
                        "sessionKey": SESSION_KEY,
                        "question": "Choose targets",
                        "choices": ["alpha", "beta"],
                        "multiSelect": True,
                        "allowOther": True,
                    },
                },
            }
        ]
        pending = clarify_gateway.get_pending_for_session(
            SESSION_KEY, include_choice_prompts=True
        )
        assert pending is not None
        assert pending.awaiting_text is True
        assert clarify_gateway.resolve_text_response_for_session(
            SESSION_KEY, "1, 2"
        ) is True
        assert clarify_gateway.wait_for_response(
            "clarify-request-1", timeout=0.1
        ) == '["alpha", "beta"]'
    finally:
        clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_send_open_ended_clarify_uses_null_choices():
    from tools import clarify_gateway

    adapter = _make_adapter()
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    clarify_gateway.register(
        "clarify-open",
        SESSION_KEY,
        "What should happen next?",
        None,
    )
    try:
        result = await adapter.send_clarify(
            CONVERSATION_ID,
            "What should happen next?",
            None,
            "clarify-open",
            SESSION_KEY,
        )

        assert result.success is True
        payload = cast(_FakeClient, adapter._client).posts[-1]["json"]["payload"]
        assert payload["choices"] is None
        assert payload["multiSelect"] is False
        assert payload["allowOther"] is True
    finally:
        clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_send_approval_resolution_targets_requesting_invocation():
    """approval.resolved must land on the invocation that sent approval.request.

    The /approve reply arrives as its own TheChat invocation and replaces the
    per-chat context, so the adapter has to remember the requesting context
    per session key.
    """
    adapter = _make_adapter()
    original = _context("invocation-original")
    adapter._contexts[CONVERSATION_ID] = original

    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "rm -rf /important",
        session_key=SESSION_KEY,
        request_id="approval-request-original",
    )

    # The /approve message overwrites the chat context with its own invocation.
    adapter._contexts[CONVERSATION_ID] = _context("invocation-approve-command")

    result = await adapter.send_approval_resolution(
        CONVERSATION_ID,
        session_key=SESSION_KEY,
        choice="session",
        resolved_count=1,
    )

    assert result.success is True
    resolution_post = adapter._client.posts[-1]
    assert resolution_post["path"] == (
        "/hermes-platform/invocations/invocation-original/progress"
    )
    assert resolution_post["json"]["type"] == "approval.resolved"
    assert resolution_post["json"]["payload"] == {
        "requestId": "approval-request-original",
        "choice": "session",
        "sessionKey": SESSION_KEY,
        "resolveAll": False,
        "resolvedCount": 1,
    }


@pytest.mark.asyncio
async def test_send_approval_resolution_uses_exact_typed_request_ids():
    adapter = _make_adapter()
    adapter._contexts[CONVERSATION_ID] = _context("invocation-first")
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "first command",
        session_key=SESSION_KEY,
        request_id="approval-request-first",
    )
    adapter._contexts[CONVERSATION_ID] = _context("invocation-second")
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "second command",
        session_key=SESSION_KEY,
        request_id="approval-request-second",
    )

    result = await adapter.send_approval_resolution(
        CONVERSATION_ID,
        session_key=SESSION_KEY,
        choice="deny",
        resolved_count=1,
        request_ids=["approval-request-second"],
    )

    assert result.success is True
    resolution_post = adapter._client.posts[-1]
    assert resolution_post["path"] == (
        "/hermes-platform/invocations/invocation-second/progress"
    )
    assert resolution_post["json"]["payload"]["requestId"] == (
        "approval-request-second"
    )
    assert "approval-request-first" in adapter._interaction_requests
    assert "approval-request-second" not in adapter._interaction_requests


@pytest.mark.asyncio
async def test_send_approval_resolution_falls_back_to_chat_context():
    adapter = _make_adapter()
    adapter._contexts[CONVERSATION_ID] = _context("invocation-current")

    result = await adapter.send_approval_resolution(
        CONVERSATION_ID,
        session_key="agent:main:thechat:dm:chat-1",
        choice="once",
        resolved_count=2,
        resolve_all=True,
    )

    assert result.success is True
    assert adapter._client.posts[-1]["path"] == (
        "/hermes-platform/invocations/invocation-current/progress"
    )
    assert adapter._client.posts[-1]["json"]["payload"]["resolveAll"] is True
    assert adapter._client.posts[-1]["json"]["payload"]["resolvedCount"] == 2


@pytest.mark.asyncio
async def test_send_approval_resolution_without_any_context_fails():
    adapter = _make_adapter()

    result = await adapter.send_approval_resolution(
        "chat-unknown",
        session_key="agent:main:thechat:dm:chat-unknown",
        choice="once",
    )

    assert result.success is False
    assert adapter._client.posts == []


def test_gateway_thechat_metadata_carries_originating_message_id():
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id=CONVERSATION_ID,
        chat_type="dm",
        user_id="user-1",
    )

    assert runner._thread_metadata_for_source(source, "message-original") == {
        "message_id": "message-original"
    }


@pytest.mark.asyncio
async def test_gateway_thechat_auto_title_callback_sends_after_context_cleanup():
    runner = object.__new__(GatewayRunner)
    adapter = _make_adapter()
    context = _context("invocation-title")
    context["thread_id"] = "thread-1"
    adapter._contexts[adapter._context_key(CONVERSATION_ID, "thread-1")] = context
    runner.adapters = {Platform.THECHAT: adapter}

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id=CONVERSATION_ID,
        chat_type="dm",
        user_id="user-1",
        thread_id="thread-1",
        message_id="message-1",
    )

    callback = runner._make_thechat_session_title_callback(
        source,
        event_message_id="message-1",
    )

    assert callable(callback)
    # The real title generation callback can run after on_processing_complete
    # removes live TheChat contexts, so the runner snapshots the context before
    # handing the callback to maybe_auto_title.
    adapter._contexts.clear()
    adapter._event_contexts.clear()

    callback("Generated Task Title")
    for _ in range(3):
        await asyncio.sleep(0)

    client = cast(_FakeClient, adapter._client)
    assert client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-title/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": CONVERSATION_ID,
                "type": "session.title",
                "status": "completed",
                "label": "Generated Task Title",
                "preview": "Generated Task Title",
                "payload": {
                    "title": "Generated Task Title",
                },
                "threadId": "thread-1",
            },
        }
    ]


@pytest.mark.asyncio
async def test_connect_defaults_to_polling_without_webhook_url(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(thechat.httpx, "AsyncClient", lambda **_kwargs: fake_client)

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={
                "base_url": "http://thechat.test",
            },
        )
    )
    polled = []

    async def poll_loop():
        polled.append(True)

    monkeypatch.setattr(adapter, "_poll_loop", poll_loop)

    assert await adapter.connect() is True
    await thechat.asyncio.sleep(0)
    assert polled == [True]
    assert adapter.webhook_url == ""
    # No webhook registration — only the slash command registration call.
    assert [post["path"] for post in fake_client.posts] == ["/bots/me/commands"]

    await adapter.disconnect()
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_connect_starts_webhook_server_and_registers_generic_bot_webhook(
    monkeypatch,
):
    fake_client = _FakeClient()
    # CI installs [all,dev] without the optional messaging extra. This test
    # stubs the webhook server itself, so bypass only the dependency gate.
    monkeypatch.setattr(thechat, "web", object())
    monkeypatch.setattr(thechat.httpx, "AsyncClient", lambda **_kwargs: fake_client)

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={
                "base_url": "http://thechat.test",
                "webhook_url": "http://gateway.test/thechat/webhook",
            },
        )
    )
    started = []

    async def start_webhook_server():
        started.append(adapter.webhook_url)

    monkeypatch.setattr(adapter, "_start_webhook_server", start_webhook_server)

    assert await adapter.connect() is True
    assert started == ["http://gateway.test/thechat/webhook"]
    assert [post["path"] for post in fake_client.posts] == [
        "/bots/me/webhook",
        "/bots/me/commands",
    ]
    assert fake_client.posts[0]["json"] == {
        "url": "http://gateway.test/thechat/webhook"
    }
    assert adapter._poll_task is None
    assert not hasattr(adapter, "_ws_task")

    await adapter.disconnect()
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_connect_registers_slash_commands(monkeypatch):
    fake_client = _FakeClient()
    monkeypatch.setattr(thechat.httpx, "AsyncClient", lambda **_kwargs: fake_client)

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test"},
        )
    )

    async def poll_loop():
        return None

    monkeypatch.setattr(adapter, "_poll_loop", poll_loop)

    assert await adapter.connect() is True
    registration = next(
        post for post in fake_client.posts if post["path"] == "/bots/me/commands"
    )
    commands = registration["json"]["commands"]
    by_name = {entry["command"]: entry for entry in commands}

    assert by_name["new"]["aliases"] == ["reset"]
    assert by_name["new"]["argsHint"] == "[name]"
    assert "help" in by_name
    assert "stop" in by_name
    # CLI-only and Telegram-specific commands never reach TheChat.
    assert "quit" not in by_name
    assert "start" not in by_name
    assert "topic" not in by_name

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_fails_when_current_command_registration_contract_fails(monkeypatch):
    class _FailingCommandsClient(_FakeClient):
        async def post(self, path, json):
            if path == "/bots/me/commands":
                raise httpx.HTTPError("404 not found")
            return await super().post(path, json=json)

    fake_client = _FailingCommandsClient()
    monkeypatch.setattr(thechat.httpx, "AsyncClient", lambda **_kwargs: fake_client)

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test"},
        )
    )

    async def poll_loop():
        return None

    monkeypatch.setattr(adapter, "_poll_loop", poll_loop)

    assert await adapter.connect() is False
    assert not adapter.is_connected
    assert adapter._client is None


@pytest.mark.asyncio
async def test_webhook_event_dispatches_to_gateway_message_handler():
    adapter = _make_adapter()
    handled = []

    async def handle(event):
        handled.append(event)

    adapter.handle_message = cast(Any, handle)

    await adapter._handle_platform_event_safely(
        _platform_item(
            text="hello from webhook",
        )
    )

    assert len(handled) == 1
    event = handled[0]
    assert event.text == "hello from webhook"
    assert event.message_id == "message-1"
    assert event.channel_prompt is None
    assert event.source.chat_id == CONVERSATION_ID
    assert event.source.chat_type == "dm"
    assert adapter._contexts[CONVERSATION_ID]["invocation_id"] == INVOCATION_ID


@pytest.mark.asyncio
async def test_threaded_platform_event_sets_source_thread_and_context_key():
    adapter = _make_adapter()
    handled = []

    async def handle(event):
        handled.append(event)

    adapter.handle_message = cast(Any, handle)

    await adapter._handle_platform_event_safely(
        _platform_item(
            text="hello from threaded task",
            thread_id="task-thread-1",
        )
    )

    assert len(handled) == 1
    event = handled[0]
    assert event.source.chat_id == CONVERSATION_ID
    assert event.source.thread_id == "task-thread-1"
    context = adapter._contexts[f"{CONVERSATION_ID}:thread:task-thread-1"]
    assert context["invocation_id"] == INVOCATION_ID
    assert context["thread_id"] == "task-thread-1"


@pytest.mark.asyncio
async def test_polling_event_dispatches_to_gateway_message_handler():
    event = _platform_item(text="hello from polling")
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test", "poll_interval": 0.001},
        )
    )
    client = _PollingClient(event)
    adapter._client = cast(httpx.AsyncClient, client)
    adapter._running = True
    handled = []

    async def handle(message):
        handled.append(message)
        adapter._running = False

    adapter.handle_message = cast(Any, handle)
    await adapter._poll_loop()

    assert client.gets[0] == {
        "path": "/hermes-platform/events",
        "params": {"limit": 10},
    }
    assert len(handled) == 1
    assert handled[0].text == "hello from polling"


def test_polling_is_default_without_webhook_url():
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={
                "base_url": "https://thechat.test",
                "webhook_host": "0.0.0.0",
                "webhook_port": 9999,
                "webhook_path": "custom/thechat",
            },
        )
    )

    assert adapter.webhook_url == ""
    assert adapter.poll_interval == 1.0


def test_webhook_url_can_be_configured_explicitly():
    adapter = _make_adapter()

    assert adapter.webhook_url == "http://gateway.test/thechat/webhook"


@pytest.mark.asyncio
async def test_webhook_registration_stores_the_one_time_secret():
    adapter = _make_adapter()

    await adapter._register_webhook()

    assert adapter._webhook_secret == "whsec-test"


def test_webhook_authorization_uses_registered_hmac_secret():
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    body = '{"type":"thechat.hermes_platform.event"}'
    timestamp = 1_700_000_000
    signature = hmac.new(
        b"whsec-test",
        f"{timestamp}.{body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "X-Webhook-Timestamp": str(timestamp),
        "X-Webhook-Signature": signature,
        "Authorization": "Bearer wrong",
    }

    assert adapter._is_authorized_webhook_request(headers, body, now=timestamp) is True
    headers["X-Webhook-Signature"] = "0" * 64
    assert adapter._is_authorized_webhook_request(headers, body, now=timestamp) is False


def test_webhook_authorization_rejects_missing_or_stale_signatures():
    adapter = _make_adapter()
    body = "{}"
    timestamp = 1_700_000_000

    assert adapter._is_authorized_webhook_request({}, body, now=timestamp) is False

    adapter._webhook_secret = "whsec-test"
    signature = hmac.new(
        b"whsec-test",
        f"{timestamp}.{body}".encode(),
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "X-Webhook-Timestamp": str(timestamp),
        "X-Webhook-Signature": signature,
    }
    assert (
        adapter._is_authorized_webhook_request(headers, body, now=timestamp + 301)
        is False
    )


@pytest.mark.asyncio
async def test_direct_approval_resolves_exact_waiter_without_message_dispatch(
    tmp_path, monkeypatch
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-direct"
    entry = _ApprovalEntry({"command": "rm -rf /important"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    dispatched = []

    async def must_not_dispatch(item):
        dispatched.append(item)
        return True

    adapter._handle_platform_event_safely = must_not_dispatch
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "rm -rf /important",
        SESSION_KEY,
        request_id=request_id,
    )
    adapter._typing_paused.add(CONVERSATION_ID)
    payload = _interaction_payload(request_id=request_id, response="once")
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)

        assert response.status == 200
        assert json.loads(response.text) == {"ok": True, "duplicate": False}
        assert entry.event.is_set()
        assert entry.result == "once"
        assert SESSION_KEY not in _gateway_queues
        assert CONVERSATION_ID not in adapter._typing_paused
        assert dispatched == []
        assert request_id not in adapter._interaction_requests
        resolved = cast(_FakeClient, adapter._client).posts[-1]
        assert resolved["path"] == (
            f"/hermes-platform/invocations/{INVOCATION_ID}/progress"
        )
        assert resolved["json"]["type"] == "approval.resolved"
        assert resolved["json"]["payload"] == {
            "requestId": request_id,
            "choice": "once",
            "sessionKey": SESSION_KEY,
            "resolveAll": False,
            "resolvedCount": 1,
        }
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_approval_cannot_jump_fifo_head(tmp_path, monkeypatch):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    first = _ApprovalEntry({"command": "first"}, request_id="approval-first")
    second = _ApprovalEntry({"command": "second"}, request_id="approval-second")
    _gateway_queues[SESSION_KEY] = [first, second]
    await adapter.send_exec_approval(
        CONVERSATION_ID, "first", SESSION_KEY, request_id=first.request_id
    )
    await adapter.send_exec_approval(
        CONVERSATION_ID, "second", SESSION_KEY, request_id=second.request_id
    )
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter,
                _interaction_payload(
                    interaction_id="later-approval-click",
                    request_id=second.request_id,
                ),
            )
        )

        assert response.status == 409
        assert not first.event.is_set()
        assert not second.event.is_set()
        assert _gateway_queues[SESSION_KEY] == [first, second]
        assert second.request_id in adapter._interaction_requests
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_webhook_acks_before_blocked_resolution_publication(
    tmp_path, monkeypatch
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    class _BlockingResolutionClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.resolution_started = asyncio.Event()
            self.release_resolution = asyncio.Event()

        async def post(self, path, json):
            self.posts.append({"path": path, "json": json})
            if json.get("type") == "approval.resolved":
                self.resolution_started.set()
                await self.release_resolution.wait()
            return _FakeResponse()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    client = _BlockingResolutionClient()
    adapter._client = cast(httpx.AsyncClient, client)
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    entry = _ApprovalEntry({"command": "danger"}, request_id="approval-fast-ack")
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=entry.request_id,
    )
    try:
        response = await asyncio.wait_for(
            adapter._handle_webhook(
                _signed_webhook_request(
                    adapter,
                    _interaction_payload(
                        interaction_id="fast-ack",
                        request_id=entry.request_id,
                    ),
                )
            ),
            timeout=1.0,
        )

        assert response.status == 200
        assert entry.event.is_set()
        await asyncio.wait_for(client.resolution_started.wait(), timeout=1.0)
        assert entry.request_id in adapter._interaction_requests

        client.release_resolution.set()
        await _drain_webhook_tasks(adapter)
        assert entry.request_id not in adapter._interaction_requests
        assert [post["json"]["type"] for post in client.posts] == [
            "approval.request",
            "approval.resolved",
        ]
    finally:
        client.release_resolution.set()
        await _drain_webhook_tasks(adapter)
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_webhook_acks_when_ledger_completion_needs_retry(
    tmp_path, monkeypatch
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    entry = _ApprovalEntry({"command": "danger"}, request_id="approval-ledger-retry")
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=entry.request_id,
    )
    real_complete = thechat.complete_inbound_event
    completion_attempts = 0

    def flaky_complete(**kwargs):
        nonlocal completion_attempts
        completion_attempts += 1
        if completion_attempts < 3:
            raise OSError("temporary ledger failure")
        return real_complete(**kwargs)

    monkeypatch.setattr(thechat, "complete_inbound_event", flaky_complete)
    payload = _interaction_payload(
        interaction_id="ledger-retry",
        request_id=entry.request_id,
    )
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )

        assert response.status == 200
        assert entry.event.is_set()
        await _drain_webhook_tasks(adapter)
        assert completion_attempts == 3

        replay = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter,
                payload,
                timestamp=int(time.time()) + 1,
            )
        )
        assert json.loads(replay.text) == {"ok": True, "duplicate": True}
    finally:
        await _drain_webhook_tasks(adapter)
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_response", "multi_select", "expected_waiter_response"),
    [
        ("custom answer", False, "custom answer"),
        (["alpha", "beta"], True, '["alpha", "beta"]'),
    ],
)
async def test_direct_clarify_resolves_string_or_json_encoded_list(
    tmp_path,
    monkeypatch,
    direct_response,
    multi_select,
    expected_waiter_response,
):
    from tools import clarify_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "clarify-request-direct"
    clarify_gateway.register(
        request_id,
        SESSION_KEY,
        "Choose targets",
        ["alpha", "beta"],
        multi_select=multi_select,
    )
    await adapter.send_clarify(
        CONVERSATION_ID,
        "Choose targets",
        ["alpha", "beta"],
        request_id,
        SESSION_KEY,
    )
    adapter._typing_paused.add(CONVERSATION_ID)
    payload = _interaction_payload(
        interaction_id=f"progress-{int(multi_select)}",
        request_type="clarify.request",
        request_id=request_id,
        response=direct_response,
    )
    try:
        webhook_response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)

        assert webhook_response.status == 200
        assert clarify_gateway.wait_for_response(request_id, timeout=0.1) == (
            expected_waiter_response
        )
        assert CONVERSATION_ID not in adapter._typing_paused
        assert request_id not in adapter._interaction_requests
        resolved = cast(_FakeClient, adapter._client).posts[-1]
        assert resolved["path"] == (
            f"/hermes-platform/invocations/{INVOCATION_ID}/progress"
        )
        assert resolved["json"]["type"] == "clarify.resolved"
        assert resolved["json"]["payload"] == {
            "requestId": request_id,
            "sessionKey": SESSION_KEY,
            "response": direct_response,
        }
    finally:
        clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_typed_clarify_winner_rejects_late_direct_callback_and_clears_card(
    tmp_path, monkeypatch
):
    from tools import clarify_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "clarify-typed-winner"
    clarify_gateway.register(
        request_id,
        SESSION_KEY,
        "Pick one",
        ["alpha", "beta"],
    )
    await adapter.send_clarify(
        CONVERSATION_ID,
        "Pick one",
        ["alpha", "beta"],
        request_id,
        SESSION_KEY,
    )
    resolved = {}
    assert clarify_gateway.resolve_text_response_for_session(
        SESSION_KEY,
        "2",
        resolved=resolved,
    ) is True

    losing = await adapter._handle_webhook(
        _signed_webhook_request(
            adapter,
            _interaction_payload(
                interaction_id="late-direct-clarify",
                request_type="clarify.request",
                request_id=request_id,
                response="alpha",
            ),
        )
    )
    assert losing.status == 409
    assert request_id in adapter._interaction_requests

    published = await adapter.send_clarify_resolution(
        chat_id=CONVERSATION_ID,
        session_key=SESSION_KEY,
        request_id=resolved["request_id"],
        response=resolved["response"],
    )
    assert published.success is True
    assert request_id not in adapter._interaction_requests
    assert cast(_FakeClient, adapter._client).posts[-1]["json"] == {
        "botId": "bot-1",
        "conversationId": CONVERSATION_ID,
        "type": "clarify.resolved",
        "status": "completed",
        "label": "Input received",
        "payload": {
            "requestId": request_id,
            "sessionKey": SESSION_KEY,
            "response": "beta",
        },
    }
    assert clarify_gateway.wait_for_response(request_id, timeout=0.1) == "beta"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        {"requestId": "unknown-request"},
        {"sessionKey": "agent:main:thechat:dm:other"},
        {"invocationId": OTHER_INVOCATION_ID},
        {"conversationId": OTHER_CONVERSATION_ID},
        {"threadId": "other-thread"},
        {"requestType": "clarify.request", "response": "answer"},
    ],
)
async def test_direct_interaction_rejects_cross_context_resolution(
    tmp_path, monkeypatch, mutation
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-isolated"
    entry = _ApprovalEntry({"command": "danger"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    payload = _interaction_payload(
        interaction_id="context-" + next(iter(mutation)),
        request_id=request_id,
    )
    payload["interaction"].update(mutation)
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )

        assert response.status == 409
        assert not entry.event.is_set()
        assert _gateway_queues[SESSION_KEY] == [entry]
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_interaction_returns_stale_when_waiter_is_gone(
    tmp_path, monkeypatch
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-expired"
    _gateway_queues[SESSION_KEY] = [
        _ApprovalEntry({"command": "danger"}, request_id=request_id)
    ]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    _gateway_queues.pop(SESSION_KEY)

    response = await adapter._handle_webhook(
        _signed_webhook_request(
            adapter,
            _interaction_payload(
                interaction_id="progress-stale",
                request_id=request_id,
            ),
        )
    )

    assert response.status == 409
    assert request_id not in adapter._interaction_requests


def test_direct_interaction_envelope_validation_is_strict_and_bounded():
    adapter = _make_adapter()
    valid = _interaction_payload()
    assert adapter._extract_webhook_interaction(valid) is valid["interaction"]

    invalid_payloads = []
    extra = json.loads(json.dumps(valid))
    extra["interaction"]["extra"] = True
    invalid_payloads.append(extra)
    invalid_uuid = json.loads(json.dumps(valid))
    invalid_uuid["interaction"]["invocationId"] = "not-a-uuid"
    invalid_payloads.append(invalid_uuid)
    oversized = json.loads(json.dumps(valid))
    oversized["interaction"]["sessionKey"] = "s" * 1025
    invalid_payloads.append(oversized)
    wrong_approval_shape = json.loads(json.dumps(valid))
    wrong_approval_shape["interaction"]["response"] = ["once"]
    invalid_payloads.append(wrong_approval_shape)
    wrong_request_type_shape = json.loads(json.dumps(valid))
    wrong_request_type_shape["interaction"]["requestType"] = ["approval.request"]
    invalid_payloads.append(wrong_request_type_shape)
    empty_clarify = _interaction_payload(
        request_type="clarify.request", response=[]
    )
    invalid_payloads.append(empty_clarify)

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            adapter._extract_webhook_interaction(payload)


@pytest.mark.asyncio
async def test_direct_interaction_auth_and_malformed_statuses(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    payload = _interaction_payload()

    bad_signature = _signed_webhook_request(adapter, payload)
    bad_signature.headers["X-Webhook-Signature"] = "0" * 64
    unauthorized = await adapter._handle_webhook(bad_signature)
    assert unauthorized.status == 401

    malformed_payload = _interaction_payload()
    malformed_payload["interaction"]["response"] = ["once"]
    malformed = await adapter._handle_webhook(
        _signed_webhook_request(adapter, malformed_payload)
    )
    assert malformed.status == 400


@pytest.mark.asyncio
async def test_direct_interaction_duplicate_and_conflicting_reuse(
    tmp_path, monkeypatch
):
    from tools.approval import _ApprovalEntry, _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-dedupe"
    entry = _ApprovalEntry({"command": "danger"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    payload = _interaction_payload(
        interaction_id="progress-dedupe",
        request_id=request_id,
        response="once",
    )
    try:
        first = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)
        duplicate = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=int(time.time()) + 1)
        )
        conflict_payload = json.loads(json.dumps(payload))
        conflict_payload["interaction"]["response"] = "deny"
        conflict = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter, conflict_payload, timestamp=int(time.time()) + 2
            )
        )

        assert first.status == 200
        assert json.loads(duplicate.text) == {"ok": True, "duplicate": True}
        assert conflict.status == 409
        assert entry.result == "once"
        resolution_posts = [
            post
            for post in cast(_FakeClient, adapter._client).posts
            if post["json"].get("type") == "approval.resolved"
        ]
        assert len(resolution_posts) == 1
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_signed_webhook_replays_are_deduplicated_durably(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    handled = []

    async def handle(item):
        handled.append(item["invocationId"])
        return True

    adapter._handle_platform_event_safely = handle
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }
    timestamp = int(time.time())

    first, concurrent_retry = await asyncio.gather(
        adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=timestamp)
        ),
        adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=timestamp)
        ),
    )
    await asyncio.gather(*adapter._webhook_tasks)

    assert first.status == concurrent_retry.status == 200
    assert handled == [INVOCATION_ID]
    assert sorted(
        json.loads(response.text).get("duplicate", False)
        for response in (first, concurrent_retry)
    ) == [False, True]

    fresh_signature_retry = await adapter._handle_webhook(
        _signed_webhook_request(adapter, payload, timestamp=timestamp + 1)
    )
    assert fresh_signature_retry.status == 200
    assert json.loads(fresh_signature_retry.text) == {"ok": True, "duplicate": True}
    assert handled == [INVOCATION_ID]

    restarted_adapter = _make_adapter()
    restarted_adapter._webhook_secret = "whsec-test"
    restarted_adapter._handle_platform_event_safely = handle
    after_restart = await restarted_adapter._handle_webhook(
        _signed_webhook_request(
            restarted_adapter,
            payload,
            timestamp=timestamp + 2,
        )
    )
    assert after_restart.status == 200
    assert json.loads(after_restart.text) == {"ok": True, "duplicate": True}
    assert restarted_adapter._webhook_tasks == set()
    assert handled == [INVOCATION_ID]

    conflicting_payload = json.loads(json.dumps(payload))
    conflicting_payload["event"]["text"] = "different signed body"
    conflict = await restarted_adapter._handle_webhook(
        _signed_webhook_request(
            restarted_adapter,
            conflicting_payload,
            timestamp=timestamp + 3,
        )
    )
    assert conflict.status == 409
    assert handled == [INVOCATION_ID]


@pytest.mark.asyncio
async def test_accepted_webhook_is_recovered_after_worker_cancellation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    started = asyncio.Event()
    never = asyncio.Event()

    async def block_until_cancelled(item):
        assert item["invocationId"] == INVOCATION_ID
        started.set()
        await never.wait()
        return True

    adapter._handle_platform_event_safely = block_until_cancelled
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }

    response = await adapter._handle_webhook(_signed_webhook_request(adapter, payload))
    assert response.status == 200
    await asyncio.wait_for(started.wait(), timeout=1)

    tasks = list(adapter._webhook_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    restarted = _make_adapter()
    restarted._webhook_secret = "whsec-test"
    recovered = []

    async def handle_recovered(item):
        recovered.append(item["invocationId"])
        return True

    restarted._handle_platform_event_safely = handle_recovered
    await restarted._process_durable_webhook_event(INVOCATION_ID)

    assert recovered == [INVOCATION_ID]
    replay = await restarted._handle_webhook(
        _signed_webhook_request(restarted, payload, timestamp=int(time.time()) + 1)
    )
    assert replay.status == 200
    assert json.loads(replay.text) == {"ok": True, "duplicate": True}


@pytest.mark.asyncio
async def test_signed_webhook_fails_closed_when_replay_ledger_is_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"

    def fail_reservation(**_kwargs):
        raise OSError("state database unavailable")

    monkeypatch.setattr(thechat, "accept_inbound_event", fail_reservation)
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }

    response = await adapter._handle_webhook(_signed_webhook_request(adapter, payload))

    assert response.status == 503
    assert adapter._webhook_tasks == set()


def test_webhook_payload_requires_current_envelope_and_event_shape():
    adapter = _make_adapter()
    event = _platform_item()

    assert (
        adapter._extract_webhook_event(
            {"type": "thechat.hermes_platform.event", "event": event}
        )
        is event
    )
    with pytest.raises(ValueError):
        adapter._extract_webhook_event(event)
    with pytest.raises(ValueError):
        adapter._extract_webhook_event({"type": "unknown"})
    with pytest.raises(ValueError):
        adapter._extract_webhook_event(
            {
                "type": "thechat.hermes_platform.event",
                "event": {"invocationId": "invocation-1", "chatId": CONVERSATION_ID},
            }
        )

    branch_event = _platform_item()
    branch_event["sessionIntent"] = {
        "type": "branch",
        "fromThreadId": "source-thread",
        "title": "Alternative",
    }
    assert adapter._validate_platform_event(branch_event) is branch_event

    branch_event["sessionIntent"] = {"type": "resume", "sessionId": "old-session"}
    with pytest.raises(ValueError):
        adapter._validate_platform_event(branch_event)

    for obsolete_chat_id in (
        "direct:user-1",
        "thechat:workspace:workspace-1:conversation:conversation-1:bot:bot-1",
    ):
        obsolete_event = _platform_item()
        obsolete_event["chatId"] = obsolete_chat_id
        obsolete_event["conversation"]["id"] = obsolete_chat_id
        with pytest.raises(ValueError):
            adapter._validate_platform_event(obsolete_event)

    extra_intent_event = _platform_item()
    extra_intent_event["sessionIntent"] = {
        "type": "branch",
        "fromThreadId": "source-thread",
        "title": "Alternative",
        "sessionId": "obsolete-session",
    }
    with pytest.raises(ValueError):
        adapter._validate_platform_event(extra_intent_event)
