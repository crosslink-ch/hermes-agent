"""TheChat platform adapter.

TheChat exposes queued Hermes bot invocations either through REST polling or
by pushing them to a webhook URL configured on the bot record. This adapter
feeds those events into the normal Hermes gateway message pipeline and posts
the gateway response back to TheChat as the configured bot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, Optional

import httpx

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised by check_thechat_requirements
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.http_routes import loopback_route
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_DEFAULT_WEBHOOK_PORT = 8765
_DEFAULT_WEBHOOK_PATH = "/thechat/webhook"


def check_thechat_requirements() -> bool:
    if os.getenv("THECHAT_WEBHOOK_URL", "").strip() and web is None:
        logger.warning("TheChat: aiohttp not installed")
        return False
    return True


class TheChatAdapter(BasePlatformAdapter):
    """Receive pending TheChat Hermes bot messages by polling or webhook."""

    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.THECHAT)
        self.config.extra.setdefault("group_sessions_per_user", False)
        self.base_url = str(
            config.extra.get("base_url") or os.getenv("THECHAT_BASE_URL", "")
        ).rstrip("/")
        self.token = str(
            config.token
            or config.extra.get("token")
            or os.getenv("THECHAT_BOT_TOKEN")
            or os.getenv("THECHAT_HERMES_BOT_TOKEN")
            or ""
        )
        self.poll_interval = float(
            config.extra.get("poll_interval")
            or os.getenv("THECHAT_POLL_INTERVAL", "1.0")
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self.webhook_host = str(
            config.extra.get("webhook_host")
            or os.getenv("THECHAT_WEBHOOK_HOST")
            or _DEFAULT_WEBHOOK_HOST
        )
        self.webhook_port = int(
            config.extra.get("webhook_port")
            or os.getenv("THECHAT_WEBHOOK_PORT")
            or _DEFAULT_WEBHOOK_PORT
        )
        self.webhook_path = self._normalize_webhook_path(
            str(
                config.extra.get("webhook_path")
                or os.getenv("THECHAT_WEBHOOK_PATH")
                or _DEFAULT_WEBHOOK_PATH
            )
        )
        configured_webhook_url = str(
            config.extra.get("webhook_url") or os.getenv("THECHAT_WEBHOOK_URL") or ""
        ).strip()
        self.webhook_url = configured_webhook_url
        self._web_app: Optional[Any] = None
        self._web_runner: Optional[Any] = None
        self._web_site: Optional[Any] = None
        self._webhook_tasks: set[asyncio.Task] = set()
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._event_contexts: Dict[str, Dict[str, Any]] = {}

    def public_http_routes(self) -> list[dict]:
        if not self.webhook_url:
            return []
        return [
            loopback_route(
                "thechat-webhook",
                path=self.webhook_path,
                port=self.webhook_port,
            ),
        ]

    async def connect(self) -> bool:
        if not self.base_url:
            logger.error("TheChat: THECHAT_BASE_URL is required")
            return False
        if not self.token:
            logger.error("TheChat: THECHAT_BOT_TOKEN is required")
            return False
        if self.webhook_url and web is None:
            logger.error("TheChat: aiohttp is required for webhook mode")
            return False

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
        try:
            response = await self._client.get("/hermes-platform/health")
            response.raise_for_status()
            if self.webhook_url:
                await self._start_webhook_server()
                await self._register_webhook()
        except Exception as exc:
            logger.error("TheChat: connection setup failed: %s", exc)
            await self.disconnect()
            return False

        self._mark_connected()
        if self.webhook_url:
            logger.info(
                "TheChat adapter connected to %s in webhook mode at %s",
                self.base_url,
                self.webhook_url,
            )
        else:
            self._poll_task = asyncio.create_task(self._poll_loop())
            logger.info(
                "TheChat adapter connected to %s in polling mode",
                self.base_url,
            )
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
        if self._webhook_tasks:
            for task in list(self._webhook_tasks):
                task.cancel()
            await asyncio.gather(*self._webhook_tasks, return_exceptions=True)
            self._webhook_tasks.clear()
        if self._web_runner:
            await self._web_runner.cleanup()
            self._web_runner = None
            self._web_site = None
            self._web_app = None
        if self._client:
            await self._client.aclose()
            self._client = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        context = self._context_for_send(chat_id, reply_to=reply_to, metadata=metadata)
        if not context:
            return SendResult(
                success=False, error=f"No TheChat context for chat {chat_id}"
            )
        if not self._client:
            return SendResult(
                success=False, error="TheChat client is not connected", retryable=True
            )
        if self._is_gateway_operational_notice(content):
            logger.debug(
                "TheChat: suppressing gateway operational notice for %s", chat_id
            )
            return SendResult(
                success=True, message_id=None, raw_response={"suppressed": True}
            )

        payload = {
            "invocationId": context["invocation_id"],
            "botId": context["bot_id"],
            "conversationId": context["conversation_id"],
            "content": content,
            "platformMessageId": f"thechat-adapter:{uuid.uuid4()}",
        }
        try:
            response = await self._client.post(
                "/hermes-platform/messages", json=payload
            )
            response.raise_for_status()
            data = response.json()
            context["delivered"] = True
            return SendResult(
                success=True,
                message_id=str(data.get("messageId") or ""),
                raw_response=data,
            )
        except Exception as exc:
            logger.error("TheChat: failed to send response: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    def _context_for_send(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidate_ids = [reply_to]
        if metadata:
            candidate_ids.extend(
                [
                    metadata.get("message_id"),
                    metadata.get("reply_to_message_id"),
                ]
            )
        for candidate in candidate_ids:
            if candidate is None:
                continue
            context = self._event_contexts.get(str(candidate))
            if context:
                return context
        return self._contexts.get(chat_id)

    def _is_gateway_operational_notice(self, content: str) -> bool:
        text = content.strip()
        return text.startswith("📬 No home channel is set for ") or text.startswith(
            "⚠️ Non-retryable error"
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
            "name": context.get("conversation_name")
            or context.get("bot_name")
            or chat_id,
            "type": context.get("chat_type") or "group",
        }

    async def on_processing_start(self, event: MessageEvent) -> None:
        context = self._event_contexts.get(str(event.message_id or ""))
        if context:
            self._contexts[event.source.chat_id] = context

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        context = self._event_contexts.pop(str(event.message_id or ""), None)
        if not context:
            return
        active_context = self._contexts.get(event.source.chat_id)
        if active_context is context:
            self._contexts.pop(event.source.chat_id, None)
        if not self._client:
            return
        if outcome == ProcessingOutcome.FAILURE:
            try:
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/failed",
                    json={"error": "Hermes gateway failed to process the message"},
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report processing failure", exc_info=True
                )
        elif outcome == ProcessingOutcome.CANCELLED:
            try:
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/cancelled",
                    json={"reason": "Hermes gateway cancelled the message"},
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report processing cancellation", exc_info=True
                )
        elif outcome == ProcessingOutcome.SUCCESS and not context.get("delivered"):
            try:
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/completed",
                    json={"reason": "Hermes gateway completed without a chat response"},
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report silent completion", exc_info=True
                )

    def _normalize_webhook_path(self, path: str) -> str:
        normalized = path.strip() or _DEFAULT_WEBHOOK_PATH
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def _default_webhook_url(self, port: int) -> str:
        host = self.webhook_host
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}{self.webhook_path}"

    async def _start_webhook_server(self) -> None:
        if web is None:
            raise RuntimeError("aiohttp is required for TheChat webhook mode")
        app = web.Application()
        app.router.add_get("/health", self._handle_webhook_health)
        app.router.add_post(self.webhook_path, self._handle_webhook)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.webhook_host, self.webhook_port)
        await site.start()

        self._web_app = app
        self._web_runner = runner
        self._web_site = site
        if self.webhook_port == 0 and not (
            self.config.extra.get("webhook_url") or os.getenv("THECHAT_WEBHOOK_URL")
        ):
            sockets = getattr(getattr(site, "_server", None), "sockets", None) or []
            if sockets:
                self.webhook_port = int(sockets[0].getsockname()[1])
                self.webhook_url = self._default_webhook_url(self.webhook_port)

    async def _handle_webhook_health(self, _request):
        return web.json_response({"ok": True, "platform": "thechat"})

    async def _register_webhook(self) -> None:
        if not self._client:
            raise RuntimeError("TheChat client is not connected")
        response = await self._client.post(
            "/bots/me/webhook", json={"url": self.webhook_url}
        )
        response.raise_for_status()

    def _is_authorized_webhook_request(self, headers: Any) -> bool:
        return headers.get("Authorization", "") == f"Bearer {self.token}"

    async def _handle_webhook(self, request):
        if not self._is_authorized_webhook_request(request.headers):
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            payload = await request.json()
            event = self._extract_webhook_event(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        task = asyncio.create_task(
            self._handle_platform_event_safely(event),
            name=f"thechat-webhook-{event.get('invocationId') or uuid.uuid4()}",
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)
        return web.json_response({"ok": True})

    def _extract_webhook_event(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object")
        if isinstance(payload.get("event"), dict):
            return payload["event"]
        if "invocationId" in payload and "chatId" in payload:
            return payload
        raise ValueError("Webhook payload does not contain a TheChat event")

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while self._running:
            try:
                response = await self._client.get(
                    "/hermes-platform/events", params={"limit": 10}
                )
                response.raise_for_status()
                payload = response.json()
                for item in payload.get("events", []):
                    await self._handle_platform_event(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TheChat: polling failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _handle_platform_event_safely(self, item: Dict[str, Any]) -> None:
        try:
            await self._handle_platform_event(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("TheChat: failed to process platform event")
            invocation_id = item.get("invocationId")
            if self._client and invocation_id:
                try:
                    await self._client.post(
                        f"/hermes-platform/invocations/{invocation_id}/failed",
                        json={
                            "error": f"Hermes gateway failed to process platform event: {exc}"
                        },
                    )
                except Exception:
                    logger.debug(
                        "TheChat: failed to report platform event processing failure",
                        exc_info=True,
                    )

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
            message_type=MessageType.COMMAND
            if text.startswith("/")
            else MessageType.TEXT,
            source=source,
            raw_message=item,
            message_id=message_id,
            channel_prompt=item.get("instructions"),
        )
        await self.handle_message(event)
