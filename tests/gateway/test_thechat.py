import asyncio
from typing import Any, cast

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.platforms import thechat
from gateway.platforms.thechat import TheChatAdapter


class _FakeResponse:
    def __init__(self, data=None):
        self._data = data or {"messageId": "thechat-message-1"}

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


def _context(invocation_id):
    return {
        "invocation_id": invocation_id,
        "bot_id": "bot-1",
        "bot_name": "Hermes",
        "conversation_id": "conversation-1",
        "conversation_name": "Hermes DM",
        "chat_type": "dm",
        "thread_id": None,
    }


def _event(adapter, message_id="message-1", chat_id="chat-1"):
    source = adapter.build_source(
        chat_id=chat_id,
        chat_type="dm",
        user_id="user-1",
        user_name="User",
        message_id=message_id,
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_send_uses_reply_to_context_instead_of_latest_chat_context():
    adapter = _make_adapter()
    first = _context("invocation-first")
    second = _context("invocation-second")
    adapter._contexts["chat-1"] = second
    adapter._event_contexts["message-first"] = first
    adapter._event_contexts["message-second"] = second

    result = await adapter.send("chat-1", "first response", reply_to="message-first")

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"]["invocationId"] == "invocation-first"
    assert adapter._client.posts[0]["json"]["complete"] is False
    assert first["delivered"] is True
    assert "delivered" not in second


@pytest.mark.asyncio
async def test_send_does_not_use_notify_metadata_as_completion_signal():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts["chat-1"] = context

    result = await adapter.send("chat-1", "final response", metadata={"notify": True})

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"]["complete"] is False


@pytest.mark.asyncio
async def test_send_can_post_without_invocation_context():
    adapter = _make_adapter()
    chat_id = "thechat:workspace:workspace-1:conversation:conversation-1:bot:bot-1"

    result = await adapter.send(chat_id, "cron says hello")

    assert result.success is True
    assert adapter._client.posts[0]["path"] == "/hermes-platform/messages"
    assert adapter._client.posts[0]["json"] == {
        "chatId": chat_id,
        "content": "cron says hello",
        "platformMessageId": adapter._client.posts[0]["json"]["platformMessageId"],
        "complete": False,
        "botId": "bot-1",
        "conversationId": "conversation-1",
    }


@pytest.mark.asyncio
async def test_processing_cancel_reports_cancelled_without_chat_message():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts["chat-1"] = context
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
    assert "chat-1" not in adapter._contexts


@pytest.mark.asyncio
async def test_processing_success_without_delivery_marks_invocation_completed():
    adapter = _make_adapter()
    context = _context("invocation-first")
    adapter._contexts["chat-1"] = context
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
    adapter._contexts["chat-1"] = context
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
    adapter._contexts["chat-1"] = stop_context
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
                "chatId": "chat-1",
                "content": "Stopped.",
                "platformMessageId": adapter._client.posts[0]["json"]["platformMessageId"],
                "complete": False,
                "invocationId": "invocation-stop",
                "botId": "bot-1",
                "conversationId": "conversation-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-stop/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert "chat-1" not in adapter._contexts


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
    adapter._contexts["chat-1"] = command_context
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
                "chatId": "chat-1",
                "content": response_text,
                "platformMessageId": adapter._client.posts[0]["json"]["platformMessageId"],
                "complete": False,
                "invocationId": "invocation-command",
                "botId": "bot-1",
                "conversationId": "conversation-1",
            },
        },
        {
            "path": "/hermes-platform/invocations/invocation-command/completed",
            "json": {"reason": "Hermes gateway completed"},
        },
    ]
    assert "chat-1" not in adapter._contexts
    assert "message-command" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_active_queue_command_completes_when_queued_turn_completes():
    adapter = _make_adapter()
    command_context = _context("invocation-queue")
    adapter._contexts["chat-1"] = command_context
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
                "chatId": "chat-1",
                "content": "Queued for the next turn.",
                "platformMessageId": adapter._client.posts[0]["json"]["platformMessageId"],
                "complete": False,
                "invocationId": "invocation-queue",
                "botId": "bot-1",
                "conversationId": "conversation-1",
            },
        }
    ]
    assert adapter._event_contexts["message-queue"] is command_context

    queued_event = adapter._pending_messages.pop(session_key)
    await adapter.on_processing_complete(queued_event, ProcessingOutcome.SUCCESS)

    assert adapter._client.posts[-1] == {
        "path": "/hermes-platform/invocations/invocation-queue/completed",
        "json": {"reason": "Hermes gateway completed"},
    }
    assert "message-queue" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_busy_command_ack_marks_thechat_invocation_completed():
    adapter = _make_adapter()
    command_context = _context("invocation-command")
    adapter._contexts["chat-1"] = command_context
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
                "chatId": "chat-1",
                "content": "Interrupting current task.",
                "platformMessageId": adapter._client.posts[0]["json"]["platformMessageId"],
                "complete": False,
                "invocationId": "invocation-command",
                "botId": "bot-1",
                "conversationId": "conversation-1",
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
    adapter._contexts["chat-1"] = context
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
    adapter._contexts["chat-1"] = context

    result = await adapter.send_invocation_progress(
        "chat-1",
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
                "conversationId": "conversation-1",
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
    adapter._contexts[adapter._context_key("chat-1", "thread-1")] = first
    adapter._contexts[adapter._context_key("chat-1", "thread-2")] = second

    result = await adapter.send_invocation_progress(
        "chat-1",
        {"type": "tool.started", "status": "running"},
        metadata={"thread_id": "thread-1"},
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-first/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": "conversation-1",
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
    adapter._contexts[adapter._context_key("chat-1", "thread-1")] = command
    adapter._contexts["chat-1"] = command
    adapter._event_contexts["message-original"] = original
    adapter._event_contexts["message-command"] = command

    result = await adapter.send_invocation_progress(
        "chat-1",
        {"type": "tool.started", "status": "running"},
        metadata={"thread_id": "thread-1", "message_id": "message-original"},
    )

    assert result.success is True
    assert adapter._client.posts == [
        {
            "path": "/hermes-platform/invocations/invocation-original/progress",
            "json": {
                "botId": "bot-1",
                "conversationId": "conversation-1",
                "type": "tool.started",
                "status": "running",
                "threadId": "thread-1",
            },
        }
    ]


def test_gateway_thechat_metadata_carries_originating_message_id():
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )

    assert runner._thread_metadata_for_source(source, "message-original") == {
        "message_id": "message-original"
    }


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
    assert fake_client.posts == []

    await adapter.disconnect()
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_connect_starts_webhook_server_and_registers_generic_bot_webhook(
    monkeypatch,
):
    fake_client = _FakeClient()
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
    assert fake_client.posts == [
        {
            "path": "/bots/me/webhook",
            "json": {"url": "http://gateway.test/thechat/webhook"},
        }
    ]
    assert adapter._poll_task is None
    assert not hasattr(adapter, "_ws_task")

    await adapter.disconnect()
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_webhook_event_dispatches_to_gateway_message_handler():
    adapter = _make_adapter()
    handled = []

    async def handle(event):
        handled.append(event)

    adapter.handle_message = cast(Any, handle)

    await adapter._handle_platform_event_safely(
        {
            "chatId": "direct:user-1",
            "chatType": "dm",
            "invocationId": "invocation-1",
            "messageId": "message-1",
            "text": "hello from webhook",
            "instructions": "reply concisely",
            "sender": {"id": "user-1", "name": "User"},
            "bot": {"id": "bot-1", "name": "Hermes"},
            "conversation": {
                "id": "conversation-1",
                "name": "Hermes DM",
                "workspaceId": "workspace-1",
            },
        }
    )

    assert len(handled) == 1
    event = handled[0]
    assert event.text == "hello from webhook"
    assert event.message_id == "message-1"
    assert event.channel_prompt == "reply concisely"
    assert event.source.chat_id == "direct:user-1"
    assert event.source.chat_type == "dm"
    assert adapter._contexts["direct:user-1"]["invocation_id"] == "invocation-1"


@pytest.mark.asyncio
async def test_threaded_platform_event_sets_source_thread_and_context_key():
    adapter = _make_adapter()
    handled = []

    async def handle(event):
        handled.append(event)

    adapter.handle_message = cast(Any, handle)

    await adapter._handle_platform_event_safely(
        {
            "chatId": "direct:user-1",
            "chatType": "dm",
            "threadId": "task-thread-1",
            "invocationId": "invocation-1",
            "messageId": "message-1",
            "text": "hello from threaded task",
            "sender": {"id": "user-1", "name": "User"},
            "bot": {"id": "bot-1", "name": "Hermes"},
            "conversation": {
                "id": "conversation-1",
                "name": "Hermes DM",
                "workspaceId": "workspace-1",
            },
        }
    )

    assert len(handled) == 1
    event = handled[0]
    assert event.source.chat_id == "direct:user-1"
    assert event.source.thread_id == "task-thread-1"
    context = adapter._contexts["direct:user-1:thread:task-thread-1"]
    assert context["invocation_id"] == "invocation-1"
    assert context["thread_id"] == "task-thread-1"


@pytest.mark.asyncio
async def test_polling_event_dispatches_to_gateway_message_handler():
    event = {
        "chatId": "direct:user-1",
        "chatType": "dm",
        "invocationId": "invocation-1",
        "messageId": "message-1",
        "text": "hello from polling",
        "sender": {"id": "user-1", "name": "User"},
        "bot": {"id": "bot-1", "name": "Hermes"},
        "conversation": {"id": "conversation-1", "name": "Hermes DM"},
    }
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


def test_webhook_authorization_uses_bot_token():
    adapter = _make_adapter()

    assert (
        adapter._is_authorized_webhook_request({"Authorization": "Bearer bot-token"})
        is True
    )
    assert (
        adapter._is_authorized_webhook_request({"Authorization": "Bearer wrong"})
        is False
    )


def test_webhook_payload_extracts_wrapped_and_direct_events():
    adapter = _make_adapter()
    event = {"invocationId": "invocation-1", "chatId": "chat-1"}

    assert (
        adapter._extract_webhook_event(
            {"type": "thechat.hermes_platform.event", "event": event}
        )
        is event
    )
    assert adapter._extract_webhook_event(event) is event
    with pytest.raises(ValueError):
        adapter._extract_webhook_event({"type": "unknown"})
