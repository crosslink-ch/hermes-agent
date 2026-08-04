"""TheChat platform adapter.

TheChat exposes queued Hermes bot invocations either through REST polling or
by pushing them to a webhook URL configured on the bot record. Webhook mode
uses a timestamped HMAC secret returned by authenticated registration, keeping
the Better Auth bot API key outbound-only. This adapter feeds those events into
the normal Hermes gateway message pipeline and posts the gateway response back
to TheChat as the configured bot.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import parse_qs, quote, urljoin, urlsplit, urlunsplit

import httpx

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised by check_thechat_requirements
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.http_routes import loopback_route
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
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    cache_media_bytes,
    get_inbound_media_max_bytes,
)
from gateway.otel import start_span
from tools.url_safety import async_is_safe_url, create_ssrf_safe_async_client

logger = logging.getLogger(__name__)

_DEFAULT_WEBHOOK_HOST = "127.0.0.1"
_DEFAULT_WEBHOOK_PORT = 8765
_DEFAULT_WEBHOOK_PATH = "/thechat/webhook"
_WEBHOOK_MAX_AGE_SECONDS = 300
_WEBHOOK_INBOX_LEASE_SECONDS = 60.0
_WEBHOOK_INBOX_RECOVERY_INTERVAL_SECONDS = 1.0
_WEBHOOK_INBOX_MAX_ATTEMPTS = 5
_ATTACHMENT_TRANSFER_TIMEOUT_SECONDS = 20.0
_ATTACHMENT_CONNECT_TIMEOUT_SECONDS = 5.0
_ATTACHMENT_POLL_TIMEOUT_SECONDS = 120.0
_ATTACHMENT_POLL_INTERVAL_SECONDS = 0.25
_ATTACHMENT_SEND_MAX_ATTEMPTS = 3
_ATTACHMENT_SEND_RETRY_BASE_SECONDS = 0.25
_ATTACHMENT_MAX_REDIRECTS = 3
_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024
_ATTACHMENT_OUTBOUND_MAX_BYTES = 10 * 1024 * 1024
_ATTACHMENT_INBOUND_MAX_COUNT = 10
_ATTACHMENT_INBOUND_TOTAL_MAX_BYTES = 50 * 1024 * 1024
_ATTACHMENT_INBOUND_TOTAL_TIMEOUT_SECONDS = 60.0
_ATTACHMENT_CHUNK_BYTES = 1024 * 1024
_ATTACHMENT_FILENAME_MAX_CHARS = 180
_ATTACHMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_ATTACHMENT_KINDS = frozenset(
    {"image", "file", "document", "audio", "video"}
)
_ATTACHMENT_UPLOAD_HEADERS = frozenset(
    {
        "content-length",
        "content-type",
        "if-none-match",
        "x-amz-checksum-sha256",
    }
)
_ATTACHMENT_MEDIA_TYPE_ALIASES = {
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "image/jpg": "image/jpeg",
}


def _normalize_attachment_media_type(value: Any) -> str:
    normalized = str(value or "").split(";", 1)[0].strip().lower()
    return _ATTACHMENT_MEDIA_TYPE_ALIASES.get(normalized, normalized)


def _s3_error_code(response: httpx.Response) -> Optional[str]:
    """Extract only S3's bounded symbolic error code, never signed details."""
    match = re.search(r"<Code>([A-Za-z0-9_.-]{1,64})</Code>", response.text[:8192])
    return match.group(1) if match else None


class _AttachmentError(RuntimeError):
    """Safe attachment error whose text never contains an object-store URL."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        preserve_attachment: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.preserve_attachment = preserve_attachment


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
        self.base_url = str(config.extra.get("base_url") or "").rstrip("/")
        self.token = str(config.token or "")
        self.poll_interval = float(config.extra.get("poll_interval") or 1.0)
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
        self._contexts: Dict[str, Dict[str, Any]] = {}
        self._event_contexts: Dict[str, Dict[str, Any]] = {}
        # session_key -> context of the invocation that requested an exec
        # approval; lets approval.resolved reach the original invocation.
        self._approval_contexts: Dict[str, Dict[str, Any]] = {}

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
            if self.webhook_url:
                await self._start_webhook_server()
                await self._register_webhook()
            await self._register_commands()
        except Exception as exc:
            logger.error("TheChat: connection setup failed: %s", exc)
            await self.disconnect()
            return False

        self._mark_connected()
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

    async def disconnect(self) -> None:
        self._running = False
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

        context, target, error = self._resolve_send_target(
            chat_id,
            reply_to=reply_to,
            metadata=metadata,
        )
        if error:
            return error
        payload = self._platform_message_payload(target, text=content)
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[
        Optional[Dict[str, Any]],
        Dict[str, Any],
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
        target: Dict[str, Any],
        *,
        text: Optional[str] = None,
        attachment_ids: Optional[list[str]] = None,
        platform_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "conversationId": str(target["conversation_id"]),
            "attachmentIds": list(attachment_ids or []),
        }
        if text is not None:
            payload["content"] = text
        if target.get("invocation_id"):
            payload["invocationId"] = str(target["invocation_id"])
        if target.get("bot_id"):
            payload["botId"] = str(target["bot_id"])
        if platform_message_id:
            payload["platformMessageId"] = platform_message_id
        if target.get("thread_id"):
            payload["threadId"] = str(target["thread_id"])
        return payload

    @staticmethod
    def _inbound_attachment_size_limit() -> int:
        configured = get_inbound_media_max_bytes()
        if configured > 0:
            return min(configured, _ATTACHMENT_MAX_BYTES)
        return _ATTACHMENT_MAX_BYTES

    def _outbound_attachment_size_limit(self) -> int:
        configured = self.config.extra.get("attachment_outbound_max_bytes")
        if configured is None:
            return _ATTACHMENT_OUTBOUND_MAX_BYTES
        try:
            value = int(configured)
        except (TypeError, ValueError):
            return _ATTACHMENT_OUTBOUND_MAX_BYTES
        return max(1, min(value, _ATTACHMENT_OUTBOUND_MAX_BYTES))

    @staticmethod
    def _sanitize_attachment_filename(value: Any, fallback: str) -> str:
        raw = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
        raw = "".join(char for char in raw if char.isprintable() and char != "\x00")
        safe = re.sub(r"[^\w.\- ]", "_", raw, flags=re.UNICODE)
        safe = re.sub(r"\s+", " ", safe).strip(" .")
        if not safe or safe in {".", ".."}:
            safe = fallback
        return safe[:_ATTACHMENT_FILENAME_MAX_CHARS]

    @staticmethod
    def _attachment_failure_note() -> str:
        return (
            "[The user attempted to send an attachment, but it could not be "
            "downloaded safely. Ask them to retry the attachment if it is needed.]"
        )

    def _parse_attachment_descriptor(
        self,
        raw: Any,
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise _AttachmentError("Attachment descriptor must be an object")

        attachment_id = str(raw.get("id") or "")
        if not _ATTACHMENT_ID_RE.fullmatch(attachment_id):
            raise _AttachmentError("Attachment descriptor has an invalid id")

        download_path = raw.get("contentPath")
        if not isinstance(download_path, str):
            raise _AttachmentError("Attachment descriptor has no download path")
        parsed_path = urlsplit(download_path)
        expected_path = f"/attachments/{quote(attachment_id, safe='')}/content"
        if (
            parsed_path.scheme
            or parsed_path.netloc
            or parsed_path.query
            or parsed_path.fragment
            or parsed_path.path != expected_path
        ):
            raise _AttachmentError("Attachment descriptor has an unsafe download path")

        raw_size = raw.get("sizeBytes")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise _AttachmentError("Attachment descriptor has an invalid size")
        size_bytes = raw_size
        if size_bytes <= 0:
            raise _AttachmentError("Attachment descriptor has an invalid size")

        limit = self._inbound_attachment_size_limit()
        if size_bytes > limit:
            raise _AttachmentError(
                f"Attachment exceeds the {limit}-byte inbound limit"
            )

        media_type = _normalize_attachment_media_type(
            raw.get("mediaType") or "application/octet-stream"
        )
        if (
            len(media_type) > 255
            or "/" not in media_type
            or " " in media_type
            or any(char in media_type for char in "\r\n\x00")
        ):
            media_type = "application/octet-stream"

        kind = str(raw.get("kind") or "file").lower()
        if kind not in _ATTACHMENT_KINDS:
            kind = "image" if media_type.startswith("image/") else "file"
        fallback_name = f"attachment-{attachment_id}"
        filename = self._sanitize_attachment_filename(
            raw.get("fileName"),
            fallback_name,
        )
        return {
            "id": attachment_id,
            "download_path": download_path,
            "filename": filename,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "kind": kind,
        }

    async def _download_inbound_attachments(
        self,
        raw_attachments: Any,
    ) -> tuple[list[str], list[str], list[str]]:
        media_urls: list[str] = []
        media_types: list[str] = []
        media_kinds: list[str] = []

        if raw_attachments is None:
            return media_urls, media_types, media_kinds
        if not isinstance(raw_attachments, list):
            logger.warning("TheChat: ignored malformed attachment list")
            return media_urls, media_types, media_kinds

        if len(raw_attachments) > _ATTACHMENT_INBOUND_MAX_COUNT:
            logger.warning(
                "TheChat: ignored %s attachments beyond the inbound count limit",
                len(raw_attachments) - _ATTACHMENT_INBOUND_MAX_COUNT,
            )
        deadline = (
            asyncio.get_running_loop().time()
            + _ATTACHMENT_INBOUND_TOTAL_TIMEOUT_SECONDS
        )
        total_bytes = 0
        for raw_descriptor in raw_attachments[:_ATTACHMENT_INBOUND_MAX_COUNT]:
            try:
                descriptor = self._parse_attachment_descriptor(raw_descriptor)
                if (
                    total_bytes + descriptor["size_bytes"]
                    > _ATTACHMENT_INBOUND_TOTAL_MAX_BYTES
                ):
                    raise _AttachmentError(
                        "Inbound attachment batch exceeds the cumulative byte limit"
                    )
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise _AttachmentError(
                        "Inbound attachment batch exceeded the transfer deadline",
                        retryable=True,
                    )
                async with asyncio.timeout(remaining):
                    data = await self._download_attachment_bytes(descriptor)
                total_bytes += len(data)
                default_kind = descriptor["kind"]
                if default_kind in {"file", "document"}:
                    default_kind = "document"
                cached = cache_media_bytes(
                    data,
                    filename=descriptor["filename"],
                    mime_type=descriptor["media_type"],
                    default_kind=default_kind,
                )
                if cached is None:
                    raise _AttachmentError("Attachment content failed validation")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "TheChat: inbound attachment was rejected (%s)",
                    exc.__class__.__name__,
                )
                continue

            media_urls.append(cached.path)
            media_types.append(cached.media_type)
            media_kinds.append(cached.kind)

        return media_urls, media_types, media_kinds

    async def _download_attachment_bytes(
        self,
        descriptor: Dict[str, Any],
    ) -> bytes:
        try:
            async with asyncio.timeout(_ATTACHMENT_TRANSFER_TIMEOUT_SECONDS):
                return await self._download_attachment_bytes_impl(descriptor)
        except TimeoutError as exc:
            raise _AttachmentError(
                "Attachment download timed out",
                retryable=True,
            ) from exc

    async def _download_attachment_bytes_impl(
        self,
        descriptor: Dict[str, Any],
    ) -> bytes:
        if not self._client:
            raise _AttachmentError("TheChat client is not connected")

        expected_size = descriptor["size_bytes"]
        limit = self._inbound_attachment_size_limit()
        read_limit = min(limit, expected_size)
        try:
            async with self._client.stream(
                "GET",
                descriptor["download_path"],
                follow_redirects=False,
                timeout=httpx.Timeout(
                    _ATTACHMENT_TRANSFER_TIMEOUT_SECONDS,
                    connect=_ATTACHMENT_CONNECT_TIMEOUT_SECONDS,
                ),
            ) as response:
                status = int(getattr(response, "status_code", 0))
                if 300 <= status < 400:
                    location = str(response.headers.get("location") or "")
                    if not location:
                        raise _AttachmentError(
                            "Attachment download redirect had no location"
                        )
                    object_url = urljoin(
                        f"{self.base_url}{descriptor['download_path']}",
                        location,
                    )
                elif 200 <= status < 300:
                    data = await self._read_bounded_body(
                        response,
                        limit=read_limit,
                    )
                    self._verify_attachment_size(data, expected_size)
                    return data
                else:
                    raise _AttachmentError(
                        f"TheChat attachment download failed with HTTP {status}"
                    )
        except _AttachmentError:
            raise
        except httpx.TimeoutException as exc:
            raise _AttachmentError(
                "TheChat attachment download timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise _AttachmentError(
                "TheChat attachment download failed",
                retryable=True,
            ) from exc

        await self._validate_object_store_url(object_url)
        data = await self._download_object_store_bytes(
            object_url,
            limit=read_limit,
        )
        self._verify_attachment_size(data, expected_size)
        return data

    @staticmethod
    def _verify_attachment_size(
        data: bytes,
        expected_size: Optional[int],
    ) -> None:
        if expected_size is not None and len(data) != expected_size:
            raise _AttachmentError(
                "Attachment content length did not match its descriptor"
            )

    @staticmethod
    async def _read_bounded_body(response: Any, *, limit: int) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except (TypeError, ValueError):
                declared_size = None
            if declared_size is not None and declared_size > limit:
                raise _AttachmentError("Attachment response exceeded the size limit")

        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes(_ATTACHMENT_CHUNK_BYTES):
            total += len(chunk)
            if total > limit:
                raise _AttachmentError("Attachment response exceeded the size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    async def _validate_object_store_url(self, url: str) -> None:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise _AttachmentError("Object-store URL was malformed") from exc
        if (
            any(char in url for char in "\r\n\x00")
            or len(url) > 8192
            or parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise _AttachmentError("Object-store URL was not an approved HTTPS URL")

        netloc = parsed.hostname
        if ":" in netloc and not netloc.startswith("["):
            netloc = f"[{netloc}]"
        if port is not None:
            netloc = f"{netloc}:{port}"
        # Check only the origin. Signed query strings never enter URL-safety
        # logs, even when DNS resolution or parsing fails.
        safe_probe = urlunsplit(("https", netloc, "/", "", ""))
        if not await async_is_safe_url(safe_probe):
            raise _AttachmentError("Object-store URL target was blocked")

    def _new_object_store_client(self) -> httpx.AsyncClient:
        return create_ssrf_safe_async_client(
            timeout=httpx.Timeout(
                _ATTACHMENT_TRANSFER_TIMEOUT_SECONDS,
                connect=_ATTACHMENT_CONNECT_TIMEOUT_SECONDS,
            ),
            follow_redirects=False,
            headers={"User-Agent": "HermesAgent/TheChat"},
        )

    async def _download_object_store_bytes(
        self,
        initial_url: str,
        *,
        limit: int,
    ) -> bytes:
        current_url = initial_url
        try:
            async with self._new_object_store_client() as client:
                for redirect_count in range(_ATTACHMENT_MAX_REDIRECTS + 1):
                    async with client.stream("GET", current_url) as response:
                        status = int(getattr(response, "status_code", 0))
                        if 200 <= status < 300:
                            return await self._read_bounded_body(
                                response,
                                limit=limit,
                            )
                        if 300 <= status < 400:
                            if redirect_count >= _ATTACHMENT_MAX_REDIRECTS:
                                raise _AttachmentError(
                                    "Object-store download redirected too many times"
                                )
                            location = str(response.headers.get("location") or "")
                            if not location:
                                raise _AttachmentError(
                                    "Object-store redirect had no location"
                                )
                            next_url = urljoin(current_url, location)
                            await self._validate_object_store_url(next_url)
                            current_url = next_url
                            continue
                        raise _AttachmentError(
                            f"Object-store download failed with HTTP {status}"
                        )
        except _AttachmentError:
            raise
        except httpx.TimeoutException as exc:
            raise _AttachmentError(
                "Object-store download timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise _AttachmentError(
                "Object-store download failed",
                retryable=True,
            ) from exc

        raise _AttachmentError("Object-store download failed")

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> SendResult:
        media_type = mimetypes.guess_type(image_path)[0] or "image/png"
        return await self._send_attachment_file(
            chat_id,
            image_path,
            file_name=Path(image_path).name,
            media_type=media_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> SendResult:
        display_name = file_name or Path(file_path).name
        media_type = (
            mimetypes.guess_type(display_name)[0]
            or mimetypes.guess_type(file_path)[0]
            or "application/octet-stream"
        )
        return await self._send_attachment_file(
            chat_id,
            file_path,
            file_name=display_name,
            media_type=media_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> SendResult:
        media_type = mimetypes.guess_type(audio_path)[0] or "audio/mpeg"
        return await self._send_attachment_file(
            chat_id,
            audio_path,
            file_name=Path(audio_path).name,
            media_type=media_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> SendResult:
        media_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
        return await self._send_attachment_file(
            chat_id,
            video_path,
            file_name=Path(video_path).name,
            media_type=media_type,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
        )

    async def _send_attachment_file(
        self,
        chat_id: str,
        file_path: str,
        *,
        file_name: str,
        media_type: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        result: SendResult | None = None
        for attempt in range(_ATTACHMENT_SEND_MAX_ATTEMPTS):
            result = await self._send_attachment_file_once(
                chat_id,
                file_path,
                file_name=file_name,
                media_type=media_type,
                caption=caption,
                reply_to=reply_to,
                metadata=metadata,
            )
            raw = result.raw_response if isinstance(result.raw_response, dict) else {}
            stage = raw.get("attachment_stage")
            if (
                result.success
                or not result.retryable
                or stage not in {"reserve", "upload", "complete"}
                or attempt + 1 >= _ATTACHMENT_SEND_MAX_ATTEMPTS
            ):
                return result
            await asyncio.sleep(
                _ATTACHMENT_SEND_RETRY_BASE_SECONDS * (attempt + 1)
            )
        return result or SendResult(
            success=False,
            error="Attachment delivery failed",
            retryable=True,
        )

    async def _send_attachment_file_once(
        self,
        chat_id: str,
        file_path: str,
        *,
        file_name: str,
        media_type: str,
        caption: Optional[str],
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> SendResult:
        if not self._client:
            return SendResult(
                success=False,
                error="TheChat client is not connected",
                retryable=True,
            )
        media_type = _normalize_attachment_media_type(media_type)

        context, target, error = self._resolve_send_target(
            chat_id,
            reply_to=reply_to,
            metadata=metadata,
        )
        if error:
            return error

        safe_path = self.validate_media_delivery_path(file_path)
        if safe_path is None:
            return SendResult(
                success=False,
                error="Attachment file path was rejected",
                retryable=False,
            )
        path = Path(safe_path)
        try:
            declared_size = path.stat().st_size
        except OSError:
            return SendResult(
                success=False,
                error="Attachment file was not found",
                retryable=False,
            )
        outbound_limit = self._outbound_attachment_size_limit()
        if declared_size < 1 or declared_size > outbound_limit:
            return SendResult(
                success=False,
                error="Attachment file exceeds the configured media size limit",
                retryable=False,
            )
        safe_name = self._sanitize_attachment_filename(
            file_name,
            "attachment",
        )

        attachment_id: Optional[str] = None
        stage = "prepare"
        try:
            size_bytes, checksum = self._hash_attachment_file(
                path,
                max_bytes=outbound_limit,
            )
            stage = "reserve"
            attachment_id, upload = await self._reserve_attachment(
                target,
                file_name=safe_name,
                media_type=media_type,
                size_bytes=size_bytes,
                checksum=checksum,
            )
            stage = "upload"
            await self._upload_attachment(
                path,
                size_bytes=size_bytes,
                upload=upload,
            )
            stage = "complete"
            await self._complete_and_wait_for_attachment(attachment_id)
            platform_message_id = self._attachment_platform_message_id(
                context,
                target,
                checksum=checksum,
                file_name=safe_name,
                caption=caption,
            )
            payload = self._platform_message_payload(
                target,
                text=caption if caption else None,
                attachment_ids=[attachment_id],
                platform_message_id=platform_message_id,
            )
            stage = "message"
            response = await self._post_attachment_message(payload)
            data = response.json()
            if context is not None:
                context["delivered"] = True
            return SendResult(
                success=True,
                message_id=str(data.get("messageId") or ""),
                raw_response=data,
            )
        except asyncio.CancelledError:
            await self._discard_attachment(attachment_id)
            raise
        except Exception as exc:
            if not (
                isinstance(exc, _AttachmentError)
                and exc.preserve_attachment
            ):
                await self._discard_attachment(attachment_id)
            if isinstance(exc, _AttachmentError):
                safe_error = str(exc)
            else:
                safe_error = (
                    f"Attachment delivery failed ({exc.__class__.__name__})"
                )
            # Do not use exc_info here: transport exceptions may embed a
            # presigned URL (including its credential-bearing query string).
            logger.warning("TheChat: %s", safe_error)
            return SendResult(
                success=False,
                error=safe_error,
                raw_response={
                    "attachment_stage": (
                        "processing"
                        if isinstance(exc, _AttachmentError)
                        and exc.preserve_attachment
                        else stage
                    )
                },
                retryable=(
                    getattr(exc, "retryable", False)
                    or isinstance(
                        exc,
                        (httpx.TimeoutException, httpx.TransportError),
                    )
                    or (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code >= 500
                    )
                ),
            )

    @staticmethod
    def _attachment_platform_message_id(
        context: Optional[Dict[str, Any]],
        target: Dict[str, Any],
        *,
        checksum: str,
        file_name: str,
        caption: Optional[str],
    ) -> str:
        if context is None or not target.get("invocation_id"):
            return f"hermes-attachment-{uuid.uuid4()}"
        sequence = int(context.get("_attachment_delivery_sequence") or 0)
        context["_attachment_delivery_sequence"] = sequence + 1
        fingerprint = "\0".join(
            [
                str(target["invocation_id"]),
                str(target.get("thread_id") or ""),
                str(sequence),
                checksum,
                file_name,
                caption or "",
            ]
        )
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return f"hermes-attachment-{digest}"

    async def _post_attachment_message(
        self,
        payload: Dict[str, Any],
    ) -> httpx.Response:
        assert self._client is not None
        last_error: Optional[Exception] = None
        for attempt in range(_ATTACHMENT_SEND_MAX_ATTEMPTS):
            try:
                response = await self._client.post(
                    "/hermes-platform/messages",
                    json=payload,
                )
                status = int(getattr(response, "status_code", 0))
                retryable_status = status in {408, 429} or status >= 500
                if retryable_status and attempt + 1 < _ATTACHMENT_SEND_MAX_ATTEMPTS:
                    await asyncio.sleep(
                        _ATTACHMENT_SEND_RETRY_BASE_SECONDS * (attempt + 1)
                    )
                    continue
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt + 1 >= _ATTACHMENT_SEND_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(
                    _ATTACHMENT_SEND_RETRY_BASE_SECONDS * (attempt + 1)
                )
        if last_error is not None:
            raise last_error
        raise _AttachmentError("TheChat attachment message delivery failed")

    @staticmethod
    def _hash_attachment_file(
        path: Path,
        *,
        max_bytes: int,
    ) -> tuple[int, str]:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_ATTACHMENT_CHUNK_BYTES)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise _AttachmentError(
                        "Attachment file exceeds the configured media size limit"
                    )
                digest.update(chunk)
        if size_bytes < 1:
            raise _AttachmentError("Attachment file is empty")
        checksum = base64.b64encode(digest.digest()).decode("ascii")
        return size_bytes, checksum

    async def _reserve_attachment(
        self,
        target: Dict[str, Any],
        *,
        file_name: str,
        media_type: str,
        size_bytes: int,
        checksum: str,
    ) -> tuple[str, Dict[str, Any]]:
        assert self._client is not None
        response = await self._client.post(
            "/attachments",
            json={
                "conversationId": str(target["conversation_id"]),
                "fileName": file_name,
                "mediaType": media_type,
                "sizeBytes": size_bytes,
                "checksumSha256": checksum,
            },
        )
        response.raise_for_status()
        data = response.json()
        attachment = data.get("attachment")
        upload = data.get("upload")
        if not isinstance(attachment, dict) or not isinstance(upload, dict):
            raise _AttachmentError("TheChat returned an invalid upload reservation")
        attachment_id = str(attachment.get("id") or "")
        if not _ATTACHMENT_ID_RE.fullmatch(attachment_id):
            raise _AttachmentError("TheChat returned an invalid attachment id")
        try:
            if upload.get("method") != "PUT":
                raise _AttachmentError("TheChat returned an unsupported upload method")
            upload_url = upload.get("url")
            headers = upload.get("headers")
            if (
                not isinstance(upload_url, str)
                or not upload_url
                or not isinstance(headers, dict)
                or not isinstance(upload.get("expiresAt"), str)
            ):
                raise _AttachmentError("TheChat returned invalid upload instructions")
            await self._validate_object_store_url(upload_url)

            normalized_headers: Dict[str, str] = {}
            for key, value in headers.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not key
                    or any(char in key or char in value for char in "\r\n\x00")
                ):
                    raise _AttachmentError("TheChat returned invalid upload headers")
                normalized_key = key.lower()
                if (
                    normalized_key not in _ATTACHMENT_UPLOAD_HEADERS
                    or normalized_key in normalized_headers
                ):
                    raise _AttachmentError(
                        "TheChat returned unsupported upload headers"
                    )
                normalized_headers[normalized_key] = value
            upload_media_type = _normalize_attachment_media_type(
                normalized_headers.get("content-type")
            )
            if upload_media_type != media_type:
                raise _AttachmentError(
                    "TheChat returned a mismatched upload media type"
                )
            header_checksum = normalized_headers.get("x-amz-checksum-sha256")
            query_checksums = [
                value
                for key, values in parse_qs(
                    urlsplit(upload_url).query,
                    keep_blank_values=True,
                ).items()
                if key.lower() == "x-amz-checksum-sha256"
                for value in values
            ]
            if header_checksum is not None and header_checksum != checksum:
                raise _AttachmentError(
                    "TheChat returned a mismatched upload checksum"
                )
            if query_checksums and query_checksums != [checksum]:
                raise _AttachmentError(
                    "TheChat returned a mismatched upload checksum"
                )
            if header_checksum is None and not query_checksums:
                raise _AttachmentError(
                    "TheChat returned upload instructions without a checksum"
                )
            content_length = normalized_headers.get("content-length")
            if content_length is not None and content_length != str(size_bytes):
                raise _AttachmentError("TheChat returned the wrong upload length")
            if normalized_headers.get("if-none-match") not in {None, "*"}:
                raise _AttachmentError("TheChat returned an unsafe upload precondition")
            return attachment_id, {
                "url": upload_url,
                "headers": normalized_headers,
            }
        except asyncio.CancelledError:
            await self._discard_attachment(attachment_id)
            raise
        except Exception:
            await self._discard_attachment(attachment_id)
            raise

    async def _discard_attachment(self, attachment_id: Optional[str]) -> None:
        if not attachment_id or not self._client:
            return
        try:
            response = await self._client.delete(
                f"/attachments/{quote(attachment_id, safe='')}"
            )
            status = int(getattr(response, "status_code", 0))
            if status not in {200, 202, 204, 404, 409}:
                logger.debug(
                    "TheChat: attachment cleanup returned HTTP %s",
                    status,
                )
        except Exception as exc:
            logger.debug(
                "TheChat: attachment cleanup failed (%s)",
                exc.__class__.__name__,
            )

    @staticmethod
    async def _file_chunks(path: Path) -> AsyncIterator[bytes]:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(_ATTACHMENT_CHUNK_BYTES)
                if not chunk:
                    return
                yield chunk

    async def _upload_attachment(
        self,
        path: Path,
        *,
        size_bytes: int,
        upload: Dict[str, Any],
    ) -> None:
        headers = dict(upload["headers"])
        if not any(key.lower() == "content-length" for key in headers):
            headers["Content-Length"] = str(size_bytes)
        try:
            async with self._new_object_store_client() as client:
                response = await client.put(
                    upload["url"],
                    headers=headers,
                    content=self._file_chunks(path),
                )
        except httpx.TimeoutException as exc:
            raise _AttachmentError(
                "Object-store upload timed out",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise _AttachmentError(
                "Object-store upload failed",
                retryable=True,
            ) from exc
        status = int(getattr(response, "status_code", 0))
        if not 200 <= status < 300:
            s3_code = _s3_error_code(response)
            detail = f" ({s3_code})" if s3_code else ""
            raise _AttachmentError(
                f"Object-store upload failed with HTTP {status}{detail}",
                retryable=status >= 500,
            )

    async def _complete_and_wait_for_attachment(
        self,
        attachment_id: str,
    ) -> Dict[str, Any]:
        assert self._client is not None
        complete_response = await self._client.post(
            f"/attachments/{quote(attachment_id, safe='')}/complete"
        )
        complete_response.raise_for_status()

        try:
            async with asyncio.timeout(_ATTACHMENT_POLL_TIMEOUT_SECONDS):
                while True:
                    response = await self._client.get(
                        f"/attachments/{quote(attachment_id, safe='')}"
                    )
                    response.raise_for_status()
                    data = response.json()
                    attachment = data.get("attachment", data)
                    if not isinstance(attachment, dict):
                        raise _AttachmentError(
                            "TheChat returned an invalid attachment status"
                        )
                    status = str(attachment.get("status") or "").lower()
                    if status == "ready":
                        return attachment
                    if status == "rejected":
                        raise _AttachmentError(
                            "TheChat rejected the attachment upload"
                        )
                    if status not in {
                        "pending",
                        "pending_upload",
                        "reserved",
                        "uploading",
                        "processing",
                        "uploaded_quarantined",
                        "scanning",
                    }:
                        raise _AttachmentError(
                            "TheChat returned an unknown attachment status"
                        )
                    await asyncio.sleep(_ATTACHMENT_POLL_INTERVAL_SECONDS)
        except TimeoutError as exc:
            raise _AttachmentError(
                "Timed out waiting for TheChat attachment processing",
                retryable=True,
                preserve_attachment=True,
            ) from exc

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
        event: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
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
                return SendResult(success=False, error=str(exc), retryable=True)

    async def send_session_title_update(
        self,
        chat_id: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        context: Optional[Dict[str, Any]] = None,
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

        event_payload: Dict[str, Any] = {"title": normalized_title}

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

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send command approval as structured TheChat invocation progress."""
        command_preview = command[:4000] + "..." if len(command) > 4000 else command
        if smart_denied or not allow_session:
            choices = ["once", "deny"]
        else:
            choices = ["once", "session"]
            if allow_permanent:
                choices.append("always")
            choices.append("deny")
        # Remember which invocation asked, keyed by session. The /approve or
        # /deny reply arrives as its own TheChat invocation and replaces the
        # per-chat context, so the later approval.resolved event must be
        # routed through this snapshot to land on the same invocation as the
        # approval.request. Overwritten by the next request for the session;
        # never used unless the gateway actually resolved a blocked approval.
        context = self._context_for_send(chat_id, metadata=metadata)
        if context:
            self._approval_contexts[session_key] = context
        return await self.send_invocation_progress(
            chat_id,
            {
                "type": "approval.request",
                "status": "waiting",
                "label": "Command approval required",
                "preview": command_preview,
                "payload": {
                    "command": command,
                    "description": description,
                    "sessionKey": session_key,
                    "choices": choices,
                },
            },
            metadata=metadata,
            context=context,
        )

    async def send_approval_resolution(
        self,
        chat_id: str,
        session_key: str,
        choice: str,
        resolved_count: int = 1,
        resolve_all: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Publish approval.resolved progress so clients dismiss the card.

        TheChat resolves approval cards oldest-first per ``sessionKey``,
        mirroring ``resolve_gateway_approval``; ``resolveAll`` collapses every
        pending card for the session at once.
        """
        context = self._approval_contexts.get(session_key) or self._context_for_send(
            chat_id, metadata=metadata
        )
        if not context:
            return SendResult(
                success=False,
                error=f"No TheChat approval context for session {session_key}",
            )
        return await self.send_invocation_progress(
            chat_id,
            {
                "type": "approval.resolved",
                "status": "completed",
                "label": "Approval resolved",
                "payload": {
                    "choice": choice,
                    "sessionKey": session_key,
                    "resolveAll": resolve_all,
                    "resolvedCount": resolved_count,
                },
            },
            metadata=metadata,
            context=context,
        )

    def _context_from_platform_event(
        self,
        item: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        invocation_id = item.get("invocationId")
        if not invocation_id:
            return None
        bot = item["bot"]
        conversation = item["conversation"]
        thread_id = str(item.get("threadId") or "") or None
        session_intent = item.get("sessionIntent")
        context: Dict[str, Any] = {
            "invocation_id": str(invocation_id),
            "bot_id": str(bot["id"]),
            "bot_name": str(bot["name"]),
            "conversation_id": str(conversation["id"]),
            "conversation_name": conversation.get("name"),
            "chat_type": item["chatType"],
            "thread_id": thread_id,
        }
        if isinstance(session_intent, dict):
            context["sessionIntent"] = session_intent
        return context

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
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

    async def _register_commands(self) -> None:
        """Register the gateway's slash commands with TheChat.

        Telegram setMyCommands-style: TheChat stores the list on the bot
        record and surfaces it as a command menu in its clients.
        """
        if not self._client:
            raise RuntimeError("TheChat client is not connected")
        from hermes_cli.commands import thechat_menu_commands

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
        body_bytes = await request.read()
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not self._is_authorized_webhook_request(request.headers, body):
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            payload = json.loads(body)
            event = self._extract_webhook_event(payload)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception:
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

    def _extract_webhook_event(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object")
        if payload.get("type") != "thechat.hermes_platform.event":
            raise ValueError("Webhook payload has an invalid TheChat event type")
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("Webhook payload does not contain a TheChat event")
        return self._validate_platform_event(event)

    def _validate_platform_event(self, item: Any) -> Dict[str, Any]:
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
                for item in payload.get("events", []):
                    await self._handle_platform_event(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("TheChat: polling failed: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _handle_platform_event_safely(self, item: Dict[str, Any]) -> bool:
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

    async def _handle_platform_event(self, item: Dict[str, Any]) -> None:
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
