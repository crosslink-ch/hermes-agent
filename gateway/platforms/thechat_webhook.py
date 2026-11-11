"""TheChat webhook receiver.

The local aiohttp server, webhook registration and HMAC verification, the secret-bearing
in-memory lanes (secure input, vault unlock), bot-request answers, direct interactions and the
durable inbox that deduplicates, leases and recovers pushed platform events.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised by check_thechat_requirements
    web = None  # type: ignore[assignment]

from gateway.inbound_event_ledger import (
    InboundEventCapacityError,
    InboundEventConflictError,
    accept_inbound_event,
    claim_inbound_event,
    complete_inbound_event,
    fail_inbound_event,
    list_recoverable_inbound_events,
    renew_inbound_event_lease,
)
from gateway.platforms.thechat_interactions import (
    _INTERACTION_PROGRESS_MAX_ATTEMPTS,
    _INTERACTION_PROGRESS_RETRY_BASE_SECONDS,
    _DirectInteractionError,
)

logger = logging.getLogger("gateway.platforms.thechat")

_DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_DEFAULT_WEBHOOK_PORT = 8765
_DEFAULT_WEBHOOK_PATH = "/thechat/webhook"
_WEBHOOK_MAX_AGE_SECONDS = 300
_WEBHOOK_INBOX_LEASE_SECONDS = 60.0
_WEBHOOK_INBOX_RECOVERY_INTERVAL_SECONDS = 1.0
_WEBHOOK_INBOX_MAX_ATTEMPTS = 5


class TheChatWebhookMixin:
    """Webhook server, verification, dispatch and durable inbox for ``TheChatAdapter``."""

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
        if self.webhook_port == 0 and not self.webhook_url:
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
        data = response.json()
        secret = data.get("webhookSecret") if isinstance(data, dict) else None
        if not isinstance(secret, str) or not secret:
            raise RuntimeError("TheChat webhook registration did not return a secret")
        self._webhook_secret = secret

    def _is_authorized_webhook_request(
        self,
        headers: Any,
        body: str,
        *,
        now: Optional[float] = None,
    ) -> bool:
        if not self._webhook_secret:
            return False

        timestamp_header = headers.get("X-Webhook-Timestamp", "")
        signature = headers.get("X-Webhook-Signature", "")
        try:
            timestamp = int(timestamp_header)
        except (TypeError, ValueError):
            return False

        current_time = time.time() if now is None else now
        if abs(current_time - timestamp) > _WEBHOOK_MAX_AGE_SECONDS:
            return False

        signed_content = f"{timestamp_header}.{body}".encode()
        expected = hmac.new(
            self._webhook_secret.encode(),
            signed_content,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _inbound_event_source(self) -> str:
        return f"thechat:{self.base_url}"

    def _inbound_interaction_source(self) -> str:
        # Keep synchronous direct interactions out of the durable asynchronous
        # message-event recovery scan. They still use the same ledger schema
        # and replay/conflict semantics under a distinct source namespace.
        return f"{self._inbound_event_source()}:interaction"

    async def _complete_direct_interaction_ledger_with_retry(
        self,
        *,
        source: str,
        event_id: str,
    ) -> None:
        """Finish durable replay bookkeeping after a resolved callback."""
        for attempt in range(_INTERACTION_PROGRESS_MAX_ATTEMPTS):
            try:
                completed = await asyncio.to_thread(
                    complete_inbound_event,
                    source=source,
                    event_id=event_id,
                    lease_owner=self._webhook_lease_owner,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Any ledger failure is retried: the interaction itself is already resolved.
                logger.debug(
                    "TheChat: direct interaction ledger completion attempt failed",
                    exc_info=True,
                )
                completed = False
            if completed:
                return
            if attempt + 1 < _INTERACTION_PROGRESS_MAX_ATTEMPTS:
                await asyncio.sleep(
                    _INTERACTION_PROGRESS_RETRY_BASE_SECONDS * (2**attempt)
                )
        logger.error("TheChat: failed to complete direct interaction replay ledger")

    def _schedule_direct_interaction_ledger_completion(
        self,
        *,
        source: str,
        event_id: str,
    ) -> None:
        task = asyncio.create_task(
            self._complete_direct_interaction_ledger_with_retry(
                source=source,
                event_id=event_id,
            ),
            name=f"thechat-interaction-ledger-{event_id}",
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    async def _handle_direct_interaction_webhook(
        self,
        interaction: dict[str, Any],
        body_bytes: bytes,
    ):
        source = self._inbound_interaction_source()
        event_id = interaction["id"]
        try:
            acceptance = await asyncio.to_thread(
                accept_inbound_event,
                source=source,
                event_id=event_id,
                payload=body_bytes,
            )
        except InboundEventConflictError:
            return web.json_response(
                {"error": "Conflicting webhook interaction"}, status=409
            )
        except InboundEventCapacityError:
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"}, status=503
            )
        except Exception:
            logger.exception("TheChat: failed to accept signed direct interaction")
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"}, status=503
            )

        if acceptance.status == "completed":
            return web.json_response({"ok": True, "duplicate": True})

        try:
            claim = await asyncio.to_thread(
                claim_inbound_event,
                source=source,
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                lease_seconds=_WEBHOOK_INBOX_LEASE_SECONDS,
                max_attempts=_WEBHOOK_INBOX_MAX_ATTEMPTS,
            )
        except Exception:
            logger.exception("TheChat: failed to claim signed direct interaction")
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"}, status=503
            )
        if claim is None:
            return web.json_response(
                {"error": "Webhook interaction is already being processed"},
                status=409,
            )

        try:
            claimed_payload = json.loads(claim.payload)
            claimed_interaction = self._extract_webhook_interaction(claimed_payload)
            resolved_record = await self._resolve_direct_interaction(
                claimed_interaction
            )
        except _DirectInteractionError as exc:
            await asyncio.to_thread(
                fail_inbound_event,
                source=source,
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                error="direct interaction rejected",
                retry_delay_seconds=0.0,
            )
            return web.json_response({"error": str(exc)}, status=exc.status)
        except ValueError:
            await asyncio.to_thread(
                fail_inbound_event,
                source=source,
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                error="malformed direct interaction",
                retry_delay_seconds=0.0,
            )
            return web.json_response({"error": "Invalid interaction"}, status=400)
        except Exception:
            logger.exception("TheChat: direct interaction processing failed")
            await asyncio.to_thread(
                fail_inbound_event,
                source=source,
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                error="direct interaction processing failed",
                retry_delay_seconds=0.0,
            )
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"}, status=503
            )

        try:
            completed = await asyncio.to_thread(
                complete_inbound_event,
                source=source,
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
            )
        except Exception:
            logger.debug(
                "TheChat: direct interaction ledger completion failed; retrying off the response path",
                exc_info=True,
            )
            completed = False
        if not completed:
            # Resolution is already final. A transient ledger failure must not
            # turn a winning click into an error at TheChat, so finish the
            # replay bookkeeping off the response path.
            self._schedule_direct_interaction_ledger_completion(
                source=source,
                event_id=event_id,
            )
        self._schedule_direct_interaction_resolution(
            claimed_interaction,
            resolved_record,
        )
        return web.json_response({"ok": True, "duplicate": False})

    def _schedule_durable_webhook_event(self, event_id: str) -> None:
        task = asyncio.create_task(
            self._process_durable_webhook_event(event_id),
            name=f"thechat-webhook-{event_id}",
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)

    async def _webhook_recovery_loop(self) -> None:
        while self._running:
            try:
                event_ids = await asyncio.to_thread(
                    list_recoverable_inbound_events,
                    source=self._inbound_event_source(),
                    max_attempts=_WEBHOOK_INBOX_MAX_ATTEMPTS,
                )
                for event_id in event_ids:
                    self._schedule_durable_webhook_event(event_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TheChat: failed to scan durable webhook inbox")
            await asyncio.sleep(_WEBHOOK_INBOX_RECOVERY_INTERVAL_SECONDS)

    async def _renew_durable_webhook_lease(self, event_id: str) -> None:
        interval = _WEBHOOK_INBOX_LEASE_SECONDS / 3
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(
                renew_inbound_event_lease,
                source=self._inbound_event_source(),
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                lease_seconds=_WEBHOOK_INBOX_LEASE_SECONDS,
            )
            if not renewed:
                return

    async def _process_durable_webhook_event(self, event_id: str) -> None:
        try:
            claim = await asyncio.to_thread(
                claim_inbound_event,
                source=self._inbound_event_source(),
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                lease_seconds=_WEBHOOK_INBOX_LEASE_SECONDS,
                max_attempts=_WEBHOOK_INBOX_MAX_ATTEMPTS,
            )
        except Exception:
            logger.exception("TheChat: failed to claim durable webhook event %s", event_id)
            return
        if claim is None:
            return

        lease_task = asyncio.create_task(
            self._renew_durable_webhook_lease(event_id),
            name=f"thechat-webhook-lease-{event_id}",
        )
        try:
            payload = json.loads(claim.payload)
            event = self._extract_webhook_event(payload)
            succeeded = await self._handle_platform_event_safely(event)
            if succeeded:
                await asyncio.to_thread(
                    complete_inbound_event,
                    source=self._inbound_event_source(),
                    event_id=event_id,
                    lease_owner=self._webhook_lease_owner,
                )
            else:
                await asyncio.to_thread(
                    fail_inbound_event,
                    source=self._inbound_event_source(),
                    event_id=event_id,
                    lease_owner=self._webhook_lease_owner,
                    error="platform event handler failed",
                    retry_delay_seconds=min(60.0, float(2 ** (claim.attempt - 1))),
                )
        except asyncio.CancelledError:
            await asyncio.to_thread(
                fail_inbound_event,
                source=self._inbound_event_source(),
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                error="webhook processing interrupted",
                retry_delay_seconds=0.0,
            )
            raise
        except Exception as exc:
            logger.exception("TheChat: durable webhook processing failed for %s", event_id)
            await asyncio.to_thread(
                fail_inbound_event,
                source=self._inbound_event_source(),
                event_id=event_id,
                lease_owner=self._webhook_lease_owner,
                error=str(exc),
                retry_delay_seconds=min(60.0, float(2 ** (claim.attempt - 1))),
            )
        finally:
            lease_task.cancel()
            await asyncio.gather(lease_task, return_exceptions=True)

    async def _handle_webhook(self, request):
        try:
            response = await self._handle_webhook_body(request)
        except web.HTTPException as exc:
            exc.headers["Cache-Control"] = "no-store"
            raise
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handle_webhook_body(self, request):
        body_bytes = await request.read()
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not self._is_authorized_webhook_request(request.headers, body):
            return web.json_response({"error": "Unauthorized"}, status=401)
        # Dedicated secret-bearing interaction lane: verify signature FIRST, then
        # dispatch in memory BEFORE the durable message/approval inbox can see it.
        from gateway.platforms.thechat_vault import VaultUnlockError
        from gateway.platforms.thechat_secure_input import SecureInputError, strict_object
        try:
            payload = json.loads(body, object_pairs_hook=strict_object)
            if isinstance(payload, dict) and payload.get("type") == "thechat.hermes_platform.secure_input":
                try:
                    if not self._secure_input_supported or not self.webhook_url:
                        raise SecureInputError()
                    duplicate = self._secure_input.resolve(payload)
                    return web.json_response({"ok": True, "duplicate": duplicate})
                except SecureInputError as exc:
                    return web.json_response({"error": "Invalid or stale secure input interaction"}, status=exc.status)
                except Exception:  # health: allow BLE001 -- secret-bearing lane: resolve() runs the turn's authority callback, so no type list is knowable; it fails closed with a fixed answer and logs neither text nor traceback
                    return web.json_response({"error": "Secure input interaction failed"}, status=409)
            if isinstance(payload, dict) and payload.get("type") == "thechat.hermes_platform.vault_unlock":
                try:
                    duplicate = self._vault_unlock.resolve(payload)
                    return web.json_response({"ok": True, "duplicate": duplicate})
                except VaultUnlockError as exc:
                    return web.json_response({"error": "Invalid or stale vault unlock interaction"}, status=exc.status)
                except Exception:  # health: allow BLE001 -- secret-bearing lane: any other broker failure fails closed with a fixed answer and logs neither text nor traceback
                    return web.json_response({"error": "Vault unlock interaction failed"}, status=409)
            if isinstance(payload, dict) and payload.get("type") == "thechat.bot_request.answered":
                # Answers are durable in TheChat; a lost reply is simply pushed again, and the
                # adapter applies each answer once. ``applied`` in the reply is the ack.
                return web.json_response(self._bot_requests.handle_webhook(payload))
            if (
                isinstance(payload, dict)
                and payload.get("type")
                == "thechat.hermes_platform.interaction"
            ):
                interaction = self._extract_webhook_interaction(payload)
                return await self._handle_direct_interaction_webhook(
                    interaction, body_bytes
                )
            event = self._extract_webhook_event(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception:
            logger.exception("TheChat: signed webhook payload could not be handled")
            return web.json_response({"error": "Invalid JSON"}, status=400)

        try:
            acceptance = await asyncio.to_thread(
                accept_inbound_event,
                source=self._inbound_event_source(),
                event_id=event["invocationId"],
                payload=body_bytes,
            )
        except InboundEventConflictError:
            logger.warning(
                "TheChat: rejected conflicting payload for invocation %s",
                event["invocationId"],
            )
            return web.json_response(
                {"error": "Conflicting webhook invocation"},
                status=409,
            )
        except InboundEventCapacityError:
            logger.error("TheChat: durable webhook inbox is at capacity")
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"},
                status=503,
            )
        except Exception:
            logger.exception("TheChat: failed to durably accept signed webhook event")
            return web.json_response(
                {"error": "Webhook receiver temporarily unavailable"},
                status=503,
            )
        if acceptance.status == "completed":
            return web.json_response({"ok": True, "duplicate": True})

        # New and pending duplicates both schedule a claimant. SQLite leases
        # ensure only one task processes the payload, while a duplicate request
        # repairs the scheduling side of an ambiguous earlier acknowledgement.
        self._schedule_durable_webhook_event(event["invocationId"])
        return web.json_response(
            {"ok": True, "duplicate": acceptance.status == "pending"}
        )

    def _extract_webhook_event(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object")
        if payload.get("type") != "thechat.hermes_platform.event":
            raise ValueError("Webhook payload has an invalid TheChat event type")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("Webhook payload does not contain a TheChat event")
        return self._validate_platform_event(event)
