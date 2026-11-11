"""TheChat attachments: bounded inbound downloads and checksummed outbound uploads.

Inbound files are fetched through TheChat's content route (which may redirect to a presigned
object-store URL) under per-file and per-batch deadlines; outbound files are reserved, uploaded
to the object store, completed and then attached to a platform message. Errors surfaced from
here never contain an object-store URL.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import parse_qs, quote, urljoin, urlsplit, urlunsplit

import httpx

from gateway.platforms.base import SendResult, cache_media_bytes, get_inbound_media_max_bytes
from tools.url_safety import async_is_safe_url, create_ssrf_safe_async_client

logger = logging.getLogger("gateway.platforms.thechat")

_ATTACHMENT_TRANSFER_TIMEOUT_SECONDS = 600.0
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
_ATTACHMENT_INBOUND_TOTAL_TIMEOUT_SECONDS = 3600.0
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
_ATTACHMENT_TIMEOUT_MAX_SECONDS = 24 * 60 * 60
_ATTACHMENT_MEDIA_TYPE_ALIASES = {
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "image/jpg": "image/jpeg",
}


def _normalize_attachment_media_type(value: Any) -> str:
    normalized = str(value or "").split(";", 1)[0].strip().lower()
    return _ATTACHMENT_MEDIA_TYPE_ALIASES.get(normalized, normalized)


def _attachment_timeout_seconds(extra: dict, key: str, default: float) -> float:
    """Read a bounded platform timeout without letting bad config disable downloads."""
    if key not in extra:
        return default
    value = extra[key]
    try:
        seconds = float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError, OverflowError):
        seconds = float("nan")
    if math.isfinite(seconds) and 0 < seconds <= _ATTACHMENT_TIMEOUT_MAX_SECONDS:
        return seconds
    logger.warning("TheChat: invalid %s; using default timeout", key)
    return default


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


class TheChatAttachmentsMixin:
    """Inbound attachment download and outbound attachment delivery for ``TheChatAdapter``."""

    def _init_attachment_timeouts(self, extra: dict) -> None:
        self.attachment_transfer_timeout_seconds = _attachment_timeout_seconds(
            extra, "attachment_transfer_timeout_seconds", _ATTACHMENT_TRANSFER_TIMEOUT_SECONDS
        )
        self.attachment_batch_timeout_seconds = _attachment_timeout_seconds(
            extra, "attachment_batch_timeout_seconds", _ATTACHMENT_INBOUND_TOTAL_TIMEOUT_SECONDS
        )

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
    ) -> dict[str, Any]:
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
            + self.attachment_batch_timeout_seconds
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
        descriptor: dict[str, Any],
    ) -> bytes:
        try:
            async with asyncio.timeout(self.attachment_transfer_timeout_seconds):
                return await self._download_attachment_bytes_impl(descriptor)
        except TimeoutError as exc:
            raise _AttachmentError(
                "Attachment download timed out",
                retryable=True,
            ) from exc

    async def _download_attachment_bytes_impl(
        self,
        descriptor: dict[str, Any],
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
                    self.attachment_transfer_timeout_seconds,
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
                self.attachment_transfer_timeout_seconds,
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
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]] = None,
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
        metadata: Optional[dict[str, Any]],
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
        metadata: Optional[dict[str, Any]],
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
        context: Optional[dict[str, Any]],
        target: dict[str, Any],
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
        payload: dict[str, Any],
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
        target: dict[str, Any],
        *,
        file_name: str,
        media_type: str,
        size_bytes: int,
        checksum: str,
    ) -> tuple[str, dict[str, Any]]:
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

            normalized_headers: dict[str, str] = {}
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
        upload: dict[str, Any],
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
    ) -> dict[str, Any]:
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
