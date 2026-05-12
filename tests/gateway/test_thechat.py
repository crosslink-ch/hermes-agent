from typing import Any, cast

import httpx
import pytest

from gateway.config import PlatformConfig
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
    assert first["delivered"] is True
    assert "delivered" not in second


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
async def test_processing_success_after_delivery_does_not_complete_twice():
    adapter = _make_adapter()
    context = _context("invocation-first")
    context["delivered"] = True
    adapter._contexts["chat-1"] = context
    adapter._event_contexts["message-first"] = context

    await adapter.on_processing_complete(
        _event(adapter, message_id="message-first"),
        ProcessingOutcome.SUCCESS,
    )

    assert adapter._client.posts == []


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
