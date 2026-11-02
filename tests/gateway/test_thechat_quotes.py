"""TheChat quoted replies reach the agent as the gateway's reply-to context."""

from typing import Any, cast

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource

CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
INVOCATION_ID = "22222222-2222-4222-8222-222222222222"
QUOTED_MESSAGE_ID = "55555555-5555-4555-8555-555555555555"
BOT_USER_ID = "bot-user-1"


class _Client:
    async def post(self, path, json):
        raise AssertionError(f"unexpected POST {path}")


def _adapter() -> tuple[TheChatAdapter, list]:
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test"},
        )
    )
    adapter._client = cast(httpx.AsyncClient, _Client())
    handled: list = []

    async def handle(event):
        handled.append(event)

    adapter.handle_message = cast(Any, handle)
    return adapter, handled


def _item(**overrides):
    item = {
        "id": INVOCATION_ID,
        "invocationId": INVOCATION_ID,
        "chatId": CONVERSATION_ID,
        "chatType": "dm",
        "threadId": None,
        "messageId": "message-1",
        "text": "How does the OS know a packet is return traffic?",
        "sender": {"id": "user-1", "name": "Bruno"},
        "bot": {"id": "bot-1", "userId": BOT_USER_ID, "name": "Koda"},
        "conversation": {
            "id": CONVERSATION_ID,
            "type": "direct",
            "name": None,
            "workspaceId": None,
        },
    }
    item.update(overrides)
    return item


def _quote(**overrides):
    quote = {
        "text": "Return traffic for those connections is allowed automatically (stateful).",
        "messageId": QUOTED_MESSAGE_ID,
        "senderId": BOT_USER_ID,
        "senderName": "Koda",
    }
    quote.update(overrides)
    return quote


def _agent_text(event) -> str:
    """The text the agent receives for this event's turn."""
    return GatewayRunner._prepend_inbound_reply_context(event, event.source, event.text)


@pytest.mark.asyncio
async def test_quote_of_the_bots_message_reaches_the_agent_as_its_own_reply_context():
    adapter, handled = _adapter()

    await adapter._handle_platform_event(_item(quote=_quote()))

    [event] = handled
    assert event.text == "How does the OS know a packet is return traffic?"
    assert event.reply_to_message_id == QUOTED_MESSAGE_ID
    assert event.reply_to_is_own_message is True
    assert event.reply_to_author_name == "Koda"
    assert _agent_text(event) == (
        '[Replying to your previous message: "Return traffic for those connections is'
        ' allowed automatically (stateful)."]\n\n'
        "How does the OS know a packet is return traffic?"
    )


@pytest.mark.asyncio
async def test_quote_of_another_person_keeps_its_author_and_reaches_the_agent():
    adapter, handled = _adapter()

    await adapter._handle_platform_event(
        _item(quote=_quote(text="Deploy window is Thursday.", senderId="user-2", senderName="Alina"))
    )

    [event] = handled
    assert event.reply_to_is_own_message is False
    assert event.reply_to_author_id == "user-2"
    assert _agent_text(event).startswith('[Replying to: "Deploy window is Thursday."]\n\n')


@pytest.mark.asyncio
async def test_quote_whose_source_was_deleted_still_reaches_the_agent():
    adapter, handled = _adapter()

    await adapter._handle_platform_event(
        _item(quote=_quote(messageId=None, senderId=None, senderName=None))
    )

    [event] = handled
    assert event.reply_to_message_id is not None
    assert event.reply_to_is_own_message is False
    assert _agent_text(event).startswith('[Replying to: "Return traffic')


@pytest.mark.asyncio
async def test_messages_without_a_quote_carry_no_reply_context():
    adapter, handled = _adapter()

    await adapter._handle_platform_event(_item())

    [event] = handled
    assert event.reply_to_message_id is None
    assert event.reply_to_text is None
    assert _agent_text(event) == event.text


@pytest.mark.parametrize(
    "quote",
    [
        "Return traffic",
        {"text": "x"},
        _quote(text="   "),
        _quote(text="x" * 4_001),
        _quote(messageId=7),
    ],
)
@pytest.mark.asyncio
async def test_a_malformed_quote_is_dropped_and_the_message_still_reaches_the_agent(quote):
    adapter, handled = _adapter()

    await adapter._handle_platform_event(_item(quote=quote))

    [event] = handled
    assert event.text == "How does the OS know a packet is return traffic?"
    assert event.reply_to_text is None
    assert event.reply_to_message_id is None


@pytest.mark.asyncio
async def test_null_quotes_and_fields_added_by_newer_servers_are_accepted():
    adapter, handled = _adapter()

    await adapter._handle_platform_event(_item(quote=None))
    await adapter._handle_platform_event(
        _item(messageId="message-2", quote={**_quote(), "addedLater": True})
    )

    plain, quoted = handled
    assert plain.reply_to_text is None
    assert quoted.reply_to_message_id == QUOTED_MESSAGE_ID


def _text_event(text: str, *, quote: str | None = None, message_id: str = "m1") -> MessageEvent:
    source = SessionSource(platform=Platform.THECHAT, chat_id=CONVERSATION_ID, chat_type="dm", user_id="user-1")
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=message_id)
    if quote is not None:
        event.reply_to_message_id = f"source-of-{message_id}"
        event.reply_to_text = quote
    return event


def test_a_merged_followup_keeps_its_own_quote():
    pending = {"session": _text_event("First question")}

    merge_pending_message_event(
        pending, "session", _text_event("Why stateful?", quote="Return traffic is allowed.", message_id="m2"),
        merge_text=True,
    )

    merged = pending["session"]
    assert merged.text == 'First question\n[Replying to: "Return traffic is allowed."]\n\nWhy stateful?'
    assert merged.reply_to_text is None


def test_a_merged_followup_with_the_turns_quote_is_not_repeated():
    pending = {"session": _text_event("First", quote="Direction matters.")}

    merge_pending_message_event(
        pending, "session", _text_event("Second", quote="Direction matters.", message_id="m1"), merge_text=True,
    )

    assert pending["session"].text == "First\nSecond"
    assert pending["session"].reply_to_text == "Direction matters."
