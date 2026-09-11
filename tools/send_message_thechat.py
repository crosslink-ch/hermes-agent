"""Standalone proactive TheChat text and attachment delivery."""
from __future__ import annotations

import os
import uuid
from typing import cast
from tools.send_message_senders import _media_caption_split, _DEFAULT_CAPTION_LIMIT, _IMAGE_EXTS, _VIDEO_EXTS, _AUDIO_EXTS
from tools.send_message_targets import _THECHAT_CONVERSATION_RE
import logging
logger = logging.getLogger(__name__)


async def _send_thechat_media(
    pconfig,
    chat_id: str,
    message: str,
    chunks,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Deliver proactive TheChat media via a live or outbound-only adapter."""
    from gateway.config import Platform
    from gateway.platforms.thechat import TheChatAdapter

    from tools.send_message_senders import _live_adapter

    # The upstream resolver is profile-scoped and fails closed on a missing
    # secondary transport; never borrow the default profile's bot identity.
    _, adapter = _live_adapter(Platform.THECHAT)
    owned_adapter = False

    if adapter is None:
        adapter = TheChatAdapter(pconfig)
        if not await adapter.connect_outbound():
            return {"error": "TheChat outbound adapter could not connect"}
        owned_adapter = True

    adapter = cast(TheChatAdapter, adapter)

    metadata = {"thread_id": thread_id} if thread_id else None
    media_files = media_files or []
    caption, body_text = _media_caption_split(
        message,
        media_files,
        max_caption_len=_DEFAULT_CAPTION_LIMIT,
    )
    results = []
    try:
        if body_text.strip():
            for chunk in chunks:
                if not str(chunk or "").strip():
                    continue
                result = await adapter.send(
                    chat_id=chat_id,
                    content=chunk,
                    metadata=metadata,
                )
                if not result.success:
                    return {"error": f"TheChat text delivery failed: {result.error}"}
                results.append(result)

        for media_path, is_voice in media_files:
            ext = os.path.splitext(media_path)[1].lower()
            media_caption = caption if len(media_files) == 1 else None
            if force_document:
                result = await adapter.send_document(
                    chat_id=chat_id,
                    file_path=media_path,
                    caption=media_caption,
                    metadata=metadata,
                )
            elif ext in _IMAGE_EXTS:
                result = await adapter.send_image_file(
                    chat_id=chat_id,
                    image_path=media_path,
                    caption=media_caption,
                    metadata=metadata,
                )
            elif ext in _VIDEO_EXTS:
                result = await adapter.send_video(
                    chat_id=chat_id,
                    video_path=media_path,
                    caption=media_caption,
                    metadata=metadata,
                )
            elif is_voice or ext in _AUDIO_EXTS:
                result = await adapter.send_voice(
                    chat_id=chat_id,
                    audio_path=media_path,
                    caption=media_caption,
                    metadata=metadata,
                )
            else:
                result = await adapter.send_document(
                    chat_id=chat_id,
                    file_path=media_path,
                    caption=media_caption,
                    metadata=metadata,
                )
            if not result.success:
                return {"error": f"TheChat media delivery failed: {result.error}"}
            results.append(result)

        if not results:
            return {"error": "TheChat media delivery produced no messages"}
        return {
            "success": True,
            "message_id": results[-1].message_id,
        }
    finally:
        if owned_adapter:
            await adapter.disconnect()



async def _send_thechat(pconfig, chat_id: str, content: str, *, thread_id=None):
    """Post a proactive TheChat bot message through the Hermes platform API."""
    from gateway.otel import start_span

    base_url = str(getattr(pconfig, "extra", {}).get("base_url") or "").rstrip("/")
    token = str(getattr(pconfig, "token", "") or "").strip()
    if not base_url:
        return {"error": "TheChat base URL is not configured"}
    if not token:
        return {"error": "TheChat bot token is not configured"}

    if not _THECHAT_CONVERSATION_RE.fullmatch(str(chat_id or "").strip()):
        return {
            "success": False,
            "error": "TheChat chat_id must be the current conversation UUID",
            "retryable": False,
        }
    payload = {
        "conversationId": chat_id,
        "content": content,
        "attachmentIds": [],
        "platformMessageId": f"send-message-tool:{uuid.uuid4()}",
        "complete": False,
    }
    if thread_id:
        payload["threadId"] = thread_id
    with start_span(
        "thechat.proactive_message.send",
        {
            "messaging.system": "thechat",
            "messaging.operation": "send",
            "thechat.chat_id": chat_id,
            "thechat.thread_id": str(thread_id or ""),
            "thechat.message.length": len(content or ""),
        },
    ) as span:
        try:
            import httpx

            async with httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(20.0, connect=5.0),
            ) as client:
                response = await client.post("/hermes-platform/messages", json=payload)
                span.set_attribute("http.status_code", response.status_code)
                if response.status_code >= 400:
                    return {
                        "error": (
                            f"TheChat send failed with HTTP {response.status_code}: "
                            f"{response.text[:300]}"
                        )
                    }
                data = response.json()
                span.set_attribute("thechat.message_id", str(data.get("messageId") or ""))
                return {
                    "success": True,
                    "message_id": str(data.get("messageId") or ""),
                }
        except Exception as exc:
            return {"error": f"TheChat send failed: {exc}"}

