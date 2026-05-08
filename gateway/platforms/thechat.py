"""TheChat platform adapter.

TheChat exposes queued Hermes bot invocations as platform events. This adapter
polls those events, feeds them into the normal Hermes gateway message pipeline,
and posts the gateway response back to TheChat as the configured bot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)


def check_thechat_requirements() -> bool:
    return True


class TheChatAdapter(BasePlatformAdapter):
    """Poll TheChat for pending Hermes bot messages."""

    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.THECHAT)
        self.config.extra.setdefault("group_sessions_per_user", False)
        self.base_url = str(config.extra.get("base_url") or os.getenv("THECHAT_BASE_URL", "")).rstrip("/")
        self.token = str(
            config.token
            or config.extra.get("token")
            or os.getenv("THECHAT_BOT_TOKEN")
            or os.getenv("THECHAT_HERMES_BOT_TOKEN")
            or ""
        )
        self.poll_interval = float(config.extra.get("poll_interval") or os.getenv("THECHAT_POLL_INTERVAL", "1.0"))
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._event_contexts: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> bool:
        if not self.base_url:
            logger.error("TheChat: THECHAT_BASE_URL is required")
            return False
        if not self.token:
            logger.error("TheChat: THECHAT_BOT_TOKEN is required")
            return False

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
        try:
            response = await self._client.get("/hermes-platform/health")
            response.raise_for_status()
        except Exception as exc:
            logger.error("TheChat: health check failed: %s", exc)
            await self.disconnect()
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("TheChat adapter connected to %s", self.base_url)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        context = self._contexts.get(chat_id)
        if not context:
            return SendResult(success=False, error=f"No TheChat context for chat {chat_id}")
        if not self._client:
            return SendResult(success=False, error="TheChat client is not connected", retryable=True)
        if self._is_gateway_operational_notice(content):
            logger.debug("TheChat: suppressing gateway operational notice for %s", chat_id)
            return SendResult(success=True, message_id=None, raw_response={"suppressed": True})

        payload = {
            "invocationId": context["invocation_id"],
            "botId": context["bot_id"],
            "conversationId": context["conversation_id"],
            "content": content,
            "platformMessageId": f"thechat-adapter:{uuid.uuid4()}",
        }
        try:
            response = await self._client.post("/hermes-platform/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            return SendResult(success=True, message_id=str(data.get("messageId") or ""), raw_response=data)
        except Exception as exc:
            logger.error("TheChat: failed to send response: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    def _is_gateway_operational_notice(self, content: str) -> bool:
        text = content.strip()
        return (
            text.startswith("📬 No home channel is set for ")
            or text.startswith("⚠️ Non-retryable error")
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        context = self._contexts.get(chat_id)
        if not context or not self._client:
            return
        try:
            await self._client.post(
                "/hermes-platform/typing",
                json={
                    "invocationId": context["invocation_id"],
                    "botId": context["bot_id"],
                    "conversationId": context["conversation_id"],
                },
            )
        except Exception:
            logger.debug("TheChat: typing update failed", exc_info=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        context = self._contexts.get(chat_id) or {}
        return {
            "name": context.get("conversation_name") or context.get("bot_name") or chat_id,
            "type": context.get("chat_type") or "group",
        }

    async def on_processing_start(self, event: MessageEvent) -> None:
        context = self._event_contexts.get(str(event.message_id or ""))
        if context:
            self._contexts[event.source.chat_id] = context

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        context = self._event_contexts.pop(str(event.message_id or ""), None)
        if not context:
            return
        if outcome == ProcessingOutcome.FAILURE and self._client:
            try:
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/failed",
                    json={"error": "Hermes gateway failed to process the message"},
                )
            except Exception:
                logger.debug("TheChat: failed to report processing failure", exc_info=True)

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while self._running:
            try:
                response = await self._client.get("/hermes-platform/events", params={"limit": 10})
                response.raise_for_status()
                payload = response.json()
                for item in payload.get("events", []):
                    await self._handle_platform_event(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TheChat: polling failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _handle_platform_event(self, item: Dict[str, Any]) -> None:
        chat_id = str(item["chatId"])
        invocation_id = str(item["invocationId"])
        bot = item.get("bot") or {}
        conversation = item.get("conversation") or {}
        sender = item.get("sender") or {}
        message_id = str(item.get("messageId") or invocation_id)
        context = {
            "invocation_id": invocation_id,
            "bot_id": str(bot.get("id") or ""),
            "bot_name": str(bot.get("name") or "Hermes"),
            "conversation_id": str(conversation.get("id") or ""),
            "conversation_name": conversation.get("name"),
            "chat_type": item.get("chatType") or "group",
        }
        self._contexts[chat_id] = context
        self._event_contexts[message_id] = context

        text = str(item.get("text") or "").strip()
        source = self.build_source(
            chat_id=chat_id,
            chat_name=conversation.get("name") or context["bot_name"],
            chat_type="dm" if item.get("chatType") == "dm" else "group",
            user_id=str(sender.get("id") or ""),
            user_name=str(sender.get("name") or "TheChat User"),
            guild_id=str(conversation.get("workspaceId") or "") or None,
            message_id=message_id,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.COMMAND if text.startswith("/") else MessageType.TEXT,
            source=source,
            raw_message=item,
            message_id=message_id,
            channel_prompt=item.get("instructions"),
        )
        await self.handle_message(event)
