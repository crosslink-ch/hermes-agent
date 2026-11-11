"""TheChat platform adapter.

TheChat exposes queued Hermes bot invocations either through REST polling or
by pushing them to a webhook URL configured on the bot record. Webhook mode
uses a timestamped HMAC secret returned by authenticated registration, keeping
the Better Auth bot API key outbound-only. This adapter feeds those events into
the normal Hermes gateway message pipeline and posts the gateway response back
to TheChat as the configured bot.

Inbound attachment settings in config.yaml under platforms.thechat:
attachment_transfer_timeout_seconds (default 600) limits each file; and
attachment_batch_timeout_seconds (default 3600) limits the whole message's
attachment batch. Both accept positive finite seconds up to 86400. For example,
`hermes config set --force platforms.thechat.attachment_transfer_timeout_seconds 900`
(or `hermes config set --force platforms.thechat.attachment_batch_timeout_seconds 5400`).
The gateway must be restarted for a changed adapter setting to take effect.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Optional

import httpx

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised by check_thechat_requirements
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.http_routes import loopback_route
from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.thechat_attachments import TheChatAttachmentsMixin
from gateway.platforms.thechat_interactions import TheChatInteractionsMixin
from gateway.platforms.thechat_webhook import (
    _DEFAULT_WEBHOOK_HOST,
    _DEFAULT_WEBHOOK_PATH,
    _DEFAULT_WEBHOOK_PORT,
    TheChatWebhookMixin,
)
from gateway.otel import start_span

logger = logging.getLogger(__name__)


def check_thechat_requirements() -> bool:
    if os.getenv("THECHAT_WEBHOOK_URL", "").strip() and web is None:
        logger.warning("TheChat: aiohttp not installed")
        return False
    return True


class TheChatAdapter(
    TheChatAttachmentsMixin,
    TheChatInteractionsMixin,
    TheChatWebhookMixin,
    BasePlatformAdapter,
):
    """Receive pending TheChat Hermes bot messages by polling or webhook.

    Topical behaviour lives in the mixins: attachments (``thechat_attachments``), approval and
    clarify interactions (``thechat_interactions``) and the webhook receiver (``thechat_webhook``).
    """

    SUPPORTS_MESSAGE_EDITING = False

    @classmethod
    def supports_exec_approval_buttons(cls) -> bool:
        """TheChat renders approvals natively (inbox request, or a legacy progress card on older
        servers) through its own ``send_exec_approval``, not the base prompt hook the default
        probe looks for. Without this the runner sends the plain-text ``/approve`` prompt."""
        return True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.THECHAT)
        self.config.extra.setdefault("group_sessions_per_user", False)
        self.base_url = str(config.extra.get("base_url") or "").rstrip("/")
        self.token = str(config.token or "")
        self.poll_interval = float(config.extra.get("poll_interval") or 1.0)
        self._init_attachment_timeouts(config.extra)
        self._client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self.webhook_host = str(
            config.extra.get("webhook_host") or _DEFAULT_WEBHOOK_HOST
        )
        self.webhook_port = int(
            config.extra.get("webhook_port", _DEFAULT_WEBHOOK_PORT)
        )
        self.webhook_path = self._normalize_webhook_path(
            str(
                config.extra.get("webhook_path") or _DEFAULT_WEBHOOK_PATH
            )
        )
        configured_webhook_url = str(config.extra.get("webhook_url") or "").strip()
        self.webhook_url = configured_webhook_url
        self._web_app: Optional[Any] = None
        self._web_runner: Optional[Any] = None
        self._web_site: Optional[Any] = None
        self._webhook_tasks: set[asyncio.Task] = set()
        self._webhook_recovery_task: Optional[asyncio.Task] = None
        self._webhook_lease_owner = f"{os.getpid()}:{uuid.uuid4().hex}"
        self._webhook_secret = ""
        self._contexts: dict[str, dict[str, Any]] = {}
        self._event_contexts: dict[str, dict[str, Any]] = {}
        # session_key -> context of the invocation that requested an exec
        # approval; lets approval.resolved reach the original invocation.
        self._approval_contexts: dict[str, dict[str, Any]] = {}
        # Opaque request id -> immutable snapshot of the invocation/request
        # context that produced a blocking approval or clarify prompt.
        self._interaction_requests: dict[str, dict[str, Any]] = {}
        # Typed /approve remains FIFO and may resolve all pending requests.
        self._approval_request_ids: dict[str, list[str]] = {}
        from gateway.platforms.thechat_vault import VaultUnlockBroker
        self._vault_unlock = VaultUnlockBroker()
        from gateway.platforms.thechat_secure_input import SecureInputBroker
        self._secure_input = SecureInputBroker()
        self._secure_input_supported = False
        self._owner_user_id = ""
        from gateway.platforms.thechat_requests import BotRequestBridge
        # Approvals/clarify as durable TheChat inbox requests when the server supports them.
        self._bot_requests = BotRequestBridge(self, redact=self._redact_interaction_value)

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

    async def connect_outbound(self) -> bool:
        """Open only the authenticated API client for proactive delivery."""
        if self._client is not None:
            return True
        if not self.base_url or not self.token:
            return False
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(20.0, connect=5.0),
        )
        try:
            response = await self._client.get("/hermes-platform/health")
            response.raise_for_status()
            health = response.json()
            self._capture_vault_owner(health)
            self._bot_requests.capture_capabilities(health)
            self._capture_secure_input_capability(health)
        except Exception:
            await self._client.aclose()
            self._client = None
            return False
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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
            health = response.json()
            self._capture_vault_owner(health)
            self._bot_requests.capture_capabilities(health)
            self._capture_secure_input_capability(health)
            from gateway.platforms.thechat_vault import VaultUnlockBroker
            if self._vault_unlock.closed:
                self._vault_unlock = VaultUnlockBroker()
            from gateway.platforms.thechat_secure_input import SecureInputBroker
            if self._secure_input.closed:
                self._secure_input = SecureInputBroker()
            if self.webhook_url:
                await self._start_webhook_server()
                await self._register_webhook()
            await self._register_commands()
        except Exception as exc:
            logger.error("TheChat: connection setup failed: %s", exc)
            await self.disconnect()
            return False

        self._mark_connected()
        # Close requests a previous process left open; in webhook mode also sweep for answers
        # whose push failed (polling mode receives them with every poll).
        self._bot_requests.start(sweep_answers=bool(self.webhook_url))
        if self.webhook_url:
            self._webhook_recovery_task = asyncio.create_task(
                self._webhook_recovery_loop()
            )
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

    def _capture_vault_owner(self, health: Any) -> None:
        # No fallback to the bot's user id, sender, config or model-controlled input.
        from gateway.platforms.thechat_vault import VaultUnlockError, _token
        self._owner_user_id = ""
        try:
            if isinstance(health, dict):
                self._owner_user_id = _token(health.get("ownerUserId"), 255)
        except VaultUnlockError:
            pass  # Older workers may omit this field; TheChat 1Password access stays denied.

    def _capture_secure_input_capability(self, health) -> None:
        capabilities = health.get("capabilities") if isinstance(health, dict) else None
        version = capabilities.get("secureInput") if isinstance(capabilities, dict) else None
        self._secure_input_supported = type(version) is int and version == 3

    async def disconnect(self) -> None:
        self._running = False
        self._vault_unlock.close()
        self._secure_input.close()
        self._secure_input_supported = False
        await self._bot_requests.stop()
        self._owner_user_id = ""
        if self._webhook_recovery_task:
            self._webhook_recovery_task.cancel()
            try:
                await self._webhook_recovery_task
            except asyncio.CancelledError:
                pass
            self._webhook_recovery_task = None
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
        self._webhook_secret = ""
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
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

        context, target, error = self._resolve_send_target(
            chat_id,
            reply_to=reply_to,
            metadata=metadata,
        )
        if error:
            return error
        # Only progress ("⏳ Working", busy acks) is flagged: TheChat then notifies only for
        # finished work. Other interim sends (shutdown notices, approval prompts) notify.
        interim = (metadata or {}).get("_interim_send") == "progress"
        payload = self._platform_message_payload(target, text=content, interim=interim)
        with start_span(
            "thechat.message.send",
            {
                "messaging.system": "thechat",
                "messaging.operation": "send",
                "thechat.chat_id": chat_id,
                "thechat.invocation_id": target.get("invocation_id") or "",
                "thechat.bot_id": target.get("bot_id") or "",
                "thechat.conversation_id": target.get("conversation_id") or "",
                "thechat.thread_id": target.get("thread_id") or "",
                "thechat.has_invocation_context": context is not None,
                "thechat.message.interim": interim,
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

    def _resolve_send_target(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[
        Optional[dict[str, Any]],
        dict[str, Any],
        Optional[SendResult],
    ]:
        context = self._context_for_send(
            chat_id, reply_to=reply_to, metadata=metadata
        )
        if context is not None and self._is_current_chat_id(chat_id):
            context_conversation_id = str(context.get("conversation_id") or "")
            if context_conversation_id != str(uuid.UUID(str(chat_id))):
                return (
                    context,
                    {},
                    SendResult(
                        success=False,
                        error=(
                            "TheChat delivery context does not match the "
                            "destination conversation"
                        ),
                        retryable=False,
                    ),
                )
        if context is None and not self._is_current_chat_id(chat_id):
            return (
                None,
                {},
                SendResult(
                    success=False,
                    error="TheChat chat_id must be the current conversation UUID",
                    retryable=False,
                ),
            )
        target = dict(context or {"conversation_id": chat_id})
        target.setdefault("conversation_id", chat_id)
        metadata_thread_id = (metadata or {}).get("thread_id") or (
            metadata or {}
        ).get("message_thread_id")
        if metadata_thread_id and not target.get("thread_id"):
            target["thread_id"] = str(metadata_thread_id)
        return context, target, None

    @staticmethod
    def _platform_message_payload(
        target: dict[str, Any],
        *,
        text: Optional[str] = None,
        attachment_ids: Optional[list[str]] = None,
        platform_message_id: Optional[str] = None,
        interim: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "conversationId": str(target["conversation_id"]),
            "attachmentIds": list(attachment_ids or []),
        }
        if text is not None:
            payload["content"] = text
        if interim:
            payload["interim"] = True
        if target.get("invocation_id"):
            payload["invocationId"] = str(target["invocation_id"])
        if target.get("bot_id"):
            payload["botId"] = str(target["bot_id"])
        if platform_message_id:
            payload["platformMessageId"] = platform_message_id
        if target.get("thread_id"):
            payload["threadId"] = str(target["thread_id"])
        return payload

    def _context_for_send(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
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

    @staticmethod
    def _is_current_chat_id(chat_id: Any) -> bool:
        value = str(chat_id or "").strip()
        try:
            return str(uuid.UUID(value)) == value.lower()
        except (ValueError, AttributeError):
            return False

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
        event: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        context = context or self._context_for_send(chat_id, metadata=metadata)
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
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                return SendResult(
                    success=False,
                    error=str(exc),
                    retryable=retryable,
                )

    async def send_session_title_update(
        self,
        chat_id: str,
        title: str,
        metadata: Optional[dict[str, Any]] = None,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Publish an auto-generated Hermes session title to TheChat.

        TheChat only needs the human-readable title for thread renaming.  Do
        not include Hermes session ids in this normal progress event; Hermes
        remains the source of truth for resolving session continuity.
        """
        normalized_title = str(title or "").strip()
        if not normalized_title:
            return SendResult(
                success=False,
                error="TheChat session title update requires title",
            )

        event_payload: dict[str, Any] = {"title": normalized_title}

        return await self.send_invocation_progress(
            chat_id,
            {
                "type": "session.title",
                "status": "completed",
                "label": normalized_title,
                "preview": normalized_title,
                "payload": event_payload,
            },
            metadata=metadata,
            context=context,
        )

    def _context_from_platform_event(
        self,
        item: Any,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        invocation_id = item.get("invocationId")
        if not invocation_id:
            return None
        bot = item["bot"]
        conversation = item["conversation"]
        thread_id = str(item.get("threadId") or "") or None
        session_intent = item.get("sessionIntent")
        context: dict[str, Any] = {
            "invocation_id": str(invocation_id),
            "bot_id": str(bot["id"]),
            "bot_name": str(bot["name"]),
            "conversation_id": str(conversation["id"]),
            "conversation_name": conversation.get("name"),
            "chat_type": item["chatType"],
            "thread_id": thread_id,
        }
        sender = item.get("sender")
        context["requester_user_id"] = sender.get("id") if isinstance(sender, dict) else None
        if isinstance(session_intent, dict):
            context["sessionIntent"] = session_intent
        return context

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        context = self._contexts.get(chat_id) or {}
        return {
            "name": context.get("conversation_name")
            or context.get("bot_name")
            or chat_id,
            "type": context.get("chat_type") or "group",
        }

    async def on_processing_start(self, event: MessageEvent) -> None:
        event_key = str(event.message_id or "")
        context = self._event_contexts.get(event_key)
        if context is None:
            # `/queue <prompt>` is acknowledged and completed immediately while
            # the queued turn is replayed later using the same platform event.
            # Rehydrate the context from raw_message so the deferred response and
            # tool-progress still route to the original TheChat task/thread.
            context = self._context_from_platform_event(event.raw_message)
            if context and event_key:
                self._event_contexts[event_key] = context
        if context:
            context["event"] = event
            self._contexts[
                self._context_key(event.source.chat_id, event.source.thread_id)
            ] = context

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        if (
            outcome == ProcessingOutcome.SUCCESS
            and getattr(event, "_hermes_startup_restore_queued", False)
            and not getattr(event, "_hermes_startup_restore_replay", False)
        ):
            # Startup only admitted the message. Keep its invocation/prompt routing until
            # the replayed turn finishes; a silent completion here rejects later unlock cards.
            return
        context = self._event_contexts.pop(str(event.message_id or ""), None)
        if not context:
            return
        self._vault_unlock.cancel_invocation(context["invocation_id"])
        self._secure_input.cancel_invocation(context["invocation_id"])
        context_key = self._context_key(event.source.chat_id, event.source.thread_id)
        active_context = self._contexts.get(context_key)
        if active_context is context:
            self._contexts.pop(context_key, None)
        if not self._client:
            return
        if outcome == ProcessingOutcome.FAILURE:
            try:
                payload: dict[str, Any] = {
                    "error": "Hermes gateway failed to process the message"
                }
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
                await self._client.post(
                    f"/hermes-platform/invocations/{context['invocation_id']}/completed",
                    json=payload,
                )
            except Exception:
                logger.debug(
                    "TheChat: failed to report completion", exc_info=True
                )

    async def _register_commands(self) -> None:
        """Register the gateway's slash commands with TheChat.

        Telegram setMyCommands-style: TheChat stores the list on the bot
        record and surfaces it as a command menu in its clients.
        """
        if not self._client:
            raise RuntimeError("TheChat client is not connected")
        from hermes_cli.commands_platforms import thechat_menu_commands

        commands, hidden_count = thechat_menu_commands()
        response = await self._client.post(
            "/bots/me/commands", json={"commands": commands}
        )
        response.raise_for_status()
        logger.info(
            "TheChat: registered %d slash commands (%d hidden by cap)",
            len(commands),
            hidden_count,
        )

    def _validate_platform_event(self, item: Any) -> dict[str, Any]:
        """Validate the current TheChat ``HermesPlatformEvent`` contract."""
        if not isinstance(item, dict):
            raise ValueError("TheChat event must be a JSON object")
        required = {
            "id",
            "invocationId",
            "chatId",
            "chatType",
            "threadId",
            "text",
            "messageId",
            "sender",
            "bot",
            "conversation",
        }
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(f"TheChat event is missing required fields: {', '.join(missing)}")
        for key in ("id", "invocationId", "chatId", "messageId"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"TheChat event has an invalid {key}")
        if not isinstance(item["text"], str):
            raise ValueError("TheChat event has an invalid text")
        if item["id"] != item["invocationId"]:
            raise ValueError("TheChat event id must match invocationId")
        if not self._is_current_chat_id(item["chatId"]):
            raise ValueError("TheChat event chatId must be a conversation UUID")
        if item["chatType"] not in {"dm", "group"}:
            raise ValueError("TheChat event has an invalid chatType")
        if item["threadId"] is not None and not isinstance(item["threadId"], str):
            raise ValueError("TheChat event has an invalid threadId")
        nested_fields = {
            "sender": {"id", "name"},
            "bot": {"id", "userId", "name"},
            "conversation": {"id", "type", "name", "workspaceId"},
        }
        for key, fields in nested_fields.items():
            value = item[key]
            if not isinstance(value, dict) or not fields.issubset(value):
                raise ValueError(f"TheChat event has an invalid {key} object")
        conversation = item["conversation"]
        if conversation["id"] != item["chatId"]:
            raise ValueError("TheChat event chatId must match conversation.id")
        expected_chat_type = "dm" if conversation["type"] == "direct" else "group"
        if conversation["type"] not in {"direct", "group"} or item["chatType"] != expected_chat_type:
            raise ValueError("TheChat event chatType does not match conversation.type")
        session_intent = item.get("sessionIntent")
        if session_intent is not None:
            if (
                not isinstance(session_intent, dict)
                or session_intent.get("type") != "branch"
                or set(session_intent) != {"type", "fromThreadId", "title"}
                or (
                    session_intent["fromThreadId"] is not None
                    and not isinstance(session_intent["fromThreadId"], str)
                )
                or (
                    session_intent["title"] is not None
                    and not isinstance(session_intent["title"], str)
                )
            ):
                raise ValueError("TheChat event has an invalid sessionIntent")
        if not item["text"] and not item.get("attachments"):
            raise ValueError("TheChat event must contain text or attachments")
        return item

    async def _poll_loop(self) -> None:
        assert self._client is not None
        while self._running:
            try:
                response = await self._client.get(
                    "/hermes-platform/events", params={"limit": 10}
                )
                response.raise_for_status()
                payload = response.json()
                try:
                    # Answers first: they unblock runs that are already waiting.
                    await self._bot_requests.handle_polled_answers(
                        payload.get("requestAnswers")
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("TheChat: applying bot request answers failed", exc_info=True)
                for item in payload.get("events", []):
                    await self._handle_platform_event(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TheChat: polling failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _handle_platform_event_safely(self, item: dict[str, Any]) -> bool:
        try:
            await self._handle_platform_event(item)
            return True
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
            return False

    async def _handle_platform_event(self, item: dict[str, Any]) -> None:
        item = self._validate_platform_event(item)
        chat_id = str(item["chatId"])
        invocation_id = str(item["invocationId"])
        thread_id = str(item.get("threadId") or "") or None
        bot = item["bot"]
        conversation = item["conversation"]
        sender = item["sender"]
        message_id = str(item["messageId"])
        session_intent = item.get("sessionIntent")
        session_intent_type = (
            str(session_intent.get("type") or "")
            if isinstance(session_intent, dict)
            else ""
        )
        with start_span(
            "thechat.event.handle",
            {
                "messaging.system": "thechat",
                "messaging.operation": "receive",
                "thechat.chat_id": chat_id,
                "thechat.invocation_id": invocation_id,
                "thechat.bot_id": str(bot["id"]),
                "thechat.conversation_id": str(conversation["id"]),
                "thechat.chat_type": item["chatType"],
                "thechat.thread_id": thread_id or "",
                "thechat.session_intent.type": session_intent_type,
            },
        ) as span:
            context = self._context_from_platform_event(item)
            if context is None:
                raise ValueError("TheChat event missing invocation context")
            self._contexts[self._context_key(chat_id, thread_id)] = context
            if thread_id is None:
                self._contexts[chat_id] = context
            self._event_contexts[message_id] = context

            text = str(item["text"]).strip()
            media_urls, media_types, media_kinds = (
                await self._download_inbound_attachments(item.get("attachments"))
            )
            raw_attachments = item.get("attachments")
            attachment_count = (
                len(raw_attachments) if isinstance(raw_attachments, list) else 0
            )
            failed_attachment_count = max(0, attachment_count - len(media_urls))
            if raw_attachments is not None and not isinstance(raw_attachments, list):
                failed_attachment_count += 1
            if failed_attachment_count:
                failure_note = self._attachment_failure_note()
                text = f"{text}\n\n{failure_note}".strip() if text else failure_note

            span.set_attribute("thechat.message_id", message_id)
            span.set_attribute("thechat.message.length", len(text))
            span.set_attribute("thechat.attachment.count", attachment_count)
            span.set_attribute(
                "thechat.attachment.cached_count",
                len(media_urls),
            )
            source = self.build_source(
                chat_id=chat_id,
                chat_name=conversation.get("name") or context["bot_name"],
                chat_type="dm" if item["chatType"] == "dm" else "group",
                user_id=str(sender["id"]),
                user_name=str(sender["name"]),
                guild_id=str(conversation["workspaceId"] or "") or None,
                thread_id=thread_id,
                message_id=message_id,
            )
            if text.startswith("/"):
                message_type = MessageType.COMMAND
            elif "image" in media_kinds:
                message_type = MessageType.PHOTO
            elif "video" in media_kinds:
                message_type = MessageType.VIDEO
            elif "audio" in media_kinds:
                message_type = MessageType.AUDIO
            elif media_urls:
                message_type = MessageType.DOCUMENT
            else:
                message_type = MessageType.TEXT
            event = MessageEvent(
                text=text,
                message_type=message_type,
                source=source,
                raw_message=item,
                message_id=message_id,
                media_urls=media_urls,
                media_types=media_types,
            )
            await self.handle_message(event)
