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
from gateway.otel import start_span

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

        await self._register_commands()

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

        context = self._context_for_send(
            chat_id, reply_to=reply_to, metadata=metadata
        )
        target = context or self._target_from_chat_id(chat_id)
        payload = {
            "chatId": chat_id,
            "content": content,
            "platformMessageId": f"thechat-adapter:{uuid.uuid4()}",
            "complete": False,
        }
        session_payload = self._session_payload_for_context(
            context, metadata=metadata, reason="message.delivered"
        )
        if session_payload:
            payload["session"] = session_payload
        if target.get("invocation_id"):
            payload["invocationId"] = target["invocation_id"]
        if target.get("bot_id"):
            payload["botId"] = target["bot_id"]
        if target.get("conversation_id"):
            payload["conversationId"] = target["conversation_id"]
        if target.get("thread_id"):
            payload["threadId"] = target["thread_id"]
        with start_span(
            "thechat.message.send",
            {
                "messaging.system": "thechat",
                "messaging.operation": "send",
                "thechat.chat_id": chat_id,
                "thechat.invocation_id": target.get("invocation_id") or "",
                "thechat.bot_id": target.get("bot_id") or "",
                "thechat.conversation_id": target.get("conversation_id") or "",
                "thechat.has_invocation_context": context is not None,
            },
        ) as span:
            try:
                response = await self._client.post(
                    "/hermes-platform/messages", json=payload
                )
                status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    span.set_attribute("http.status_code", status_code)
                response.raise_for_status()
                data = response.json()
                if context is not None:
                    context["delivered"] = True
                span.set_attribute("thechat.message_id", str(data.get("messageId") or ""))
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
        if metadata:
            thread_id = metadata.get("thread_id") or metadata.get("message_thread_id")
            if thread_id:
                context = self._contexts.get(
                    self._context_key(chat_id, str(thread_id))
                )
                if context:
                    return context
        return self._contexts.get(chat_id)

    def _context_key(self, chat_id: str, thread_id: Optional[str] = None) -> str:
        if thread_id:
            return f"{chat_id}:thread:{thread_id}"
        return chat_id

    def _target_from_chat_id(self, chat_id: str) -> Dict[str, Any]:
        target: Dict[str, Any] = {"chat_id": chat_id}
        parts = str(chat_id or "").split(":")
        for index, part in enumerate(parts[:-1]):
            if part == "conversation" and parts[index + 1]:
                target["conversation_id"] = parts[index + 1]
            elif part == "bot" and parts[index + 1]:
                target["bot_id"] = parts[index + 1]
        return target

    def _session_payload_for_context(
        self,
        context: Optional[Dict[str, Any]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        for value in (
            (metadata or {}).get("session"),
            (context or {}).get("session"),
            getattr((context or {}).get("event"), "hermes_session", None),
            (context or {}).get("continuity"),
        ):
            payload = self._normalize_session_payload(value, reason=reason)
            if payload:
                if context is not None:
                    context["session"] = payload
                return payload
        return None

    def _normalize_session_payload(
        self,
        value: Any,
        *,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(value, dict):
            return None

        def _field(camel: str, snake: str) -> Optional[str]:
            raw = value.get(camel, value.get(snake))
            if raw is None:
                return None
            text = str(raw).strip()
            return text or None

        payload: Dict[str, Any] = {}
        for camel, snake in (
            ("sessionId", "session_id"),
            ("sessionKey", "session_key"),
            ("lineageRootId", "lineage_root_id"),
            ("branchFromSessionId", "branch_from_session_id"),
            ("branchFromThreadId", "branch_from_thread_id"),
            ("branchFromLineageRootId", "branch_from_lineage_root_id"),
            ("branchTitle", "branch_title"),
        ):
            field_value = _field(camel, snake)
            if field_value:
                payload[camel] = field_value

        if not any(
            payload.get(key)
            for key in ("sessionId", "sessionKey", "branchFromSessionId")
        ):
            return None

        payload["reason"] = _field("reason", "reason") or reason
        payload["source"] = _field("source", "source") or "hermes"
        updated_at = _field("updatedAt", "updated_at")
        if updated_at:
            payload["updatedAt"] = updated_at
        return payload

    def _is_gateway_operational_notice(self, content: str) -> bool:
        text = content.strip()
        return text.startswith("📬 No home channel is set for ") or text.startswith(
            "⚠️ Non-retryable error"
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        context = self._context_for_send(chat_id, metadata=metadata)
        if not context or not self._client:
            return
        try:
            payload = {
                "invocationId": context["invocation_id"],
                "botId": context["bot_id"],
                "conversationId": context["conversation_id"],
            }
            if context.get("thread_id"):
                payload["threadId"] = context["thread_id"]
            await self._client.post("/hermes-platform/typing", json=payload)
        except Exception:
            logger.debug("TheChat: typing update failed", exc_info=True)

    async def send_invocation_progress(
        self,
        chat_id: str,
        event: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        context = self._context_for_send(chat_id, metadata=metadata)
        if not context:
            return SendResult(
                success=False, error=f"No TheChat context for chat {chat_id}"
            )
        if not self._client:
            return SendResult(
                success=False, error="TheChat client is not connected", retryable=True
            )

        payload = {
            "botId": context["bot_id"],
            "conversationId": context["conversation_id"],
            **event,
        }
        session_payload = self._session_payload_for_context(
            context, metadata=metadata, reason="progress"
        )
        if session_payload:
            payload["session"] = session_payload
        if context.get("thread_id"):
            payload["threadId"] = context["thread_id"]
        with start_span(
            "thechat.invocation.progress.send",
            {
                "messaging.system": "thechat",
                "messaging.operation": "progress",
                "thechat.chat_id": chat_id,
                "thechat.invocation_id": context["invocation_id"],
                "thechat.bot_id": context["bot_id"],
                "thechat.conversation_id": context["conversation_id"],
                "thechat.progress.type": event.get("type") or "",
                "thechat.progress.tool": event.get("toolName") or "",
            },
        ) as span:
            try:
                response = await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/progress",
                    json=payload,
                )
                status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    span.set_attribute("http.status_code", status_code)
                response.raise_for_status()
                return SendResult(success=True, raw_response=response.json())
            except Exception as exc:
                logger.debug("TheChat: failed to send invocation progress", exc_info=True)
                return SendResult(success=False, error=str(exc), retryable=True)

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
            context["event"] = event
            self._contexts[
                self._context_key(event.source.chat_id, event.source.thread_id)
            ] = context

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        context = self._event_contexts.pop(str(event.message_id or ""), None)
        if not context:
            return
        context_key = self._context_key(event.source.chat_id, event.source.thread_id)
        active_context = self._contexts.get(context_key)
        if active_context is context:
            self._contexts.pop(context_key, None)
        if not self._client:
            return
        if outcome == ProcessingOutcome.FAILURE:
            try:
                payload: Dict[str, Any] = {
                    "error": "Hermes gateway failed to process the message"
                }
                session_payload = self._session_payload_for_context(
                    context, reason="invocation.failed"
                )
                if session_payload:
                    payload["session"] = session_payload
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/failed",
                    json=payload,
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report processing failure", exc_info=True
                )
        elif outcome == ProcessingOutcome.CANCELLED:
            try:
                payload = {"reason": "Hermes gateway cancelled the message"}
                session_payload = self._session_payload_for_context(
                    context, reason="invocation.cancelled"
                )
                if session_payload:
                    payload["session"] = session_payload
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/cancelled",
                    json=payload,
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report processing cancellation", exc_info=True
                )
        elif outcome == ProcessingOutcome.SUCCESS:
            try:
                payload = {
                    "reason": "Hermes gateway completed"
                    if context.get("delivered")
                    else "Hermes gateway completed without a chat response"
                }
                session_payload = self._session_payload_for_context(
                    context, reason="invocation.completed"
                )
                if session_payload:
                    payload["session"] = session_payload
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/completed",
                    json=payload,
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report completion", exc_info=True
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

    async def _register_commands(self) -> None:
        """Register the gateway's slash commands with TheChat.

        Telegram setMyCommands-style: TheChat stores the list on the bot
        record and surfaces it as a command menu in its clients.  Best-effort
        — older TheChat servers without the endpoint must not break connect.
        """
        if not self._client:
            return
        try:
            from hermes_cli.commands import thechat_menu_commands

            commands, hidden_count = thechat_menu_commands()
            response = await self._client.post(
                "/bots/me/commands", json={"commands": commands}
            )
            if getattr(response, "status_code", None) == 404:
                logger.info(
                    "TheChat: server does not support command registration; skipping"
                )
                return
            response.raise_for_status()
            logger.info(
                "TheChat: registered %d slash commands (%d hidden by cap)",
                len(commands),
                hidden_count,
            )
        except Exception as exc:
            logger.warning("TheChat: failed to register slash commands: %s", exc)

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
        thread_id = str(item.get("threadId") or item.get("thread_id") or "") or None
        bot = item.get("bot") or {}
        conversation = item.get("conversation") or {}
        sender = item.get("sender") or {}
        message_id = str(item.get("messageId") or invocation_id)
        with start_span(
            "thechat.event.handle",
            {
                "messaging.system": "thechat",
                "messaging.operation": "receive",
                "thechat.chat_id": chat_id,
                "thechat.invocation_id": invocation_id,
                "thechat.bot_id": str(bot.get("id") or ""),
                "thechat.conversation_id": str(conversation.get("id") or ""),
                "thechat.chat_type": item.get("chatType") or "group",
                "thechat.thread_id": thread_id or "",
                "thechat.continuity_id": str((item.get("continuity") or {}).get("id") or ""),
            },
        ) as span:
            context = {
                "invocation_id": invocation_id,
                "bot_id": str(bot.get("id") or ""),
                "bot_name": str(bot.get("name") or "Hermes"),
                "conversation_id": str(conversation.get("id") or ""),
                "conversation_name": conversation.get("name"),
                "chat_type": item.get("chatType") or "group",
                "thread_id": thread_id,
            }
            continuity = item.get("continuity")
            if isinstance(continuity, dict):
                context["continuity"] = continuity
            self._contexts[self._context_key(chat_id, thread_id)] = context
            if thread_id is None:
                self._contexts[chat_id] = context
            self._event_contexts[message_id] = context

            text = str(item.get("text") or "").strip()
            span.set_attribute("thechat.message_id", message_id)
            span.set_attribute("thechat.message.length", len(text))
            source = self.build_source(
                chat_id=chat_id,
                chat_name=conversation.get("name") or context["bot_name"],
                chat_type="dm" if item.get("chatType") == "dm" else "group",
                user_id=str(sender.get("id") or ""),
                user_name=str(sender.get("name") or "TheChat User"),
                guild_id=str(conversation.get("workspaceId") or "") or None,
                thread_id=thread_id,
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
