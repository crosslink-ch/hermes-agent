import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
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

    async def post(self, path, json):
        self.posts.append({"path": path, "json": json})
        return _FakeResponse()


def _make_adapter():
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test"},
        )
    )
    adapter._client = _FakeClient()
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
