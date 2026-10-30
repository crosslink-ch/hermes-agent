"""Optional fields of a TheChat ``HermesPlatformEvent``: validation and mapping.

The adapter facade (``thechat.py``) validates the required envelope; the optional
fields that carry per-message context live here so each has one validator and one
mapping onto ``MessageEvent``.
"""

from __future__ import annotations

import logging
from typing import Any

# TheChat stores at most 4,000 characters of quoted text; anything longer is not
# from a current TheChat server.
QUOTE_TEXT_MAX_LENGTH = 4_000
_QUOTE_FIELDS = {"text", "messageId", "senderId", "senderName"}

logger = logging.getLogger(__name__)


def validate_session_intent(item: dict[str, Any]) -> None:
    session_intent = item.get("sessionIntent")
    if session_intent is None:
        return
    if (
        not isinstance(session_intent, dict)
        or session_intent.get("type") != "branch"
        or set(session_intent) != {"type", "fromThreadId", "title"}
        or not _is_optional_str(session_intent["fromThreadId"])
        or not _is_optional_str(session_intent["title"])
    ):
        raise ValueError("TheChat event has an invalid sessionIntent")


def quote_of(item: dict[str, Any]) -> dict[str, Any] | None:
    """The validated quote of an event: the snippet of an earlier message it replies to.

    ``messageId`` is null once the quoted message was deleted; the text is a
    snapshot and stays. Author fields are null when TheChat could not attribute
    the text to the quoted message's author. A quote is optional context, so a
    malformed one is ignored and the message itself still reaches the agent;
    fields TheChat may add later are ignored too.
    """
    quote = item.get("quote")
    if quote is None:
        return None
    if (
        isinstance(quote, dict)
        and _QUOTE_FIELDS <= quote.keys()
        and isinstance(quote["text"], str)
        and quote["text"].strip()
        and len(quote["text"]) <= QUOTE_TEXT_MAX_LENGTH
        and all(_is_optional_str(quote[key]) for key in ("messageId", "senderId", "senderName"))
    ):
        return quote
    logger.warning("TheChat event %s has an invalid quote; delivering it without one", item.get("id"))
    return None


def reply_fields(item: dict[str, Any]) -> dict[str, Any]:
    """``MessageEvent`` reply-context keyword arguments for a validated event.

    The gateway injects ``reply_to_text`` only when ``reply_to_message_id`` is
    set, so a quote whose source was deleted is anchored to the quoting message
    instead of being dropped.
    """
    quote = quote_of(item)
    if quote is None:
        return {}
    return {
        "reply_to_message_id": quote["messageId"] or f"quote:{item['messageId']}",
        "reply_to_text": quote["text"],
        "reply_to_author_id": quote["senderId"],
        "reply_to_author_name": quote["senderName"],
        "reply_to_is_own_message": (
            quote["senderId"] is not None and quote["senderId"] == item["bot"]["userId"]
        ),
    }


def _is_optional_str(value: Any) -> bool:
    return value is None or isinstance(value, str)
