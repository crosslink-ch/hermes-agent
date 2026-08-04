from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms import base as platform_base
from gateway.platforms import thechat
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.thechat import TheChatAdapter


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
INVOCATION_ID = "22222222-2222-4222-8222-222222222222"
SCRATCH_ROOT = Path(
    os.environ.get(
        "HERMES_TEST_SCRATCH_DIR",
        str(Path.home() / ".cache" / "hermes-tests" / "thechat-attachments"),
    )
)


class _StreamResponse:
    def __init__(
        self,
        *,
        body: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = headers or {}

    async def __aenter__(self) -> "_StreamResponse":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int | None = None):
        size = chunk_size or len(self.body) or 1
        for offset in range(0, len(self.body), size):
            yield self.body[offset : offset + size]


class _ObjectStoreClient:
    def __init__(
        self,
        *,
        download_bodies: dict[str, bytes] | None = None,
        upload_status: int = 200,
        upload_error: Exception | None = None,
    ) -> None:
        self.download_bodies = download_bodies or {}
        self.upload_status = upload_status
        self.upload_error = upload_error
        self.stream_calls: list[str] = []
        self.put_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_ObjectStoreClient":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    def stream(self, method: str, url: str) -> _StreamResponse:
        assert method == "GET"
        self.stream_calls.append(url)
        return _StreamResponse(
            body=self.download_bodies[url],
            headers={"content-length": str(len(self.download_bodies[url]))},
        )

    async def put(
        self,
        url: str,
        *,
        headers: dict[str, str],
        content: Any,
    ) -> _StreamResponse:
        if self.upload_error is not None:
            raise self.upload_error
        body = bytearray()
        async for chunk in content:
            body.extend(chunk)
        self.put_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "body": bytes(body),
            }
        )
        return _StreamResponse(status_code=self.upload_status)


@pytest.fixture
def attachment_scratch(monkeypatch):
    root = SCRATCH_ROOT / f"pytest-{uuid.uuid4().hex}"
    image_cache = root / "cache" / "images"
    document_cache = root / "cache" / "documents"
    root.mkdir(parents=True)
    monkeypatch.setattr(platform_base, "IMAGE_CACHE_DIR", image_cache)
    monkeypatch.setattr(platform_base, "DOCUMENT_CACHE_DIR", document_cache)
    yield root
    shutil.rmtree(root)


@pytest.fixture
def allow_object_store_urls(monkeypatch):
    checked: list[str] = []

    async def allow(url: str) -> bool:
        checked.append(url)
        return True

    monkeypatch.setattr(thechat, "async_is_safe_url", allow)
    return checked


def _adapter() -> TheChatAdapter:
    return TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "https://thechat.example"},
        )
    )


def _platform_event(
    *,
    attachments: Any,
    text: str = "",
    thread_id: str | None = "thread-1",
) -> dict[str, Any]:
    return {
        "id": INVOCATION_ID,
        "invocationId": INVOCATION_ID,
        "chatId": CONVERSATION_ID,
        "chatType": "dm",
        "threadId": thread_id,
        "messageId": "message-1",
        "text": text,
        "attachments": attachments,
        "sender": {"id": "user-1", "name": "User"},
        "bot": {"id": "bot-1", "userId": "bot-user-1", "name": "Hermes"},
        "conversation": {
            "id": CONVERSATION_ID,
            "type": "direct",
            "name": "Hermes DM",
            "workspaceId": "workspace-1",
        },
    }


def _descriptor(
    attachment_id: str,
    *,
    file_name: str,
    media_type: str,
    body: bytes,
    kind: str,
) -> dict[str, Any]:
    return {
        "id": attachment_id,
        "fileName": file_name,
        "mediaType": media_type,
        "sizeBytes": len(body),
        "kind": kind,
        "contentPath": f"/attachments/{attachment_id}/content",
    }


def _json_response(
    status_code: int,
    payload: dict[str, Any],
    request: httpx.Request,
) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=request)


@pytest.mark.asyncio
async def test_inbound_image_and_file_follow_authenticated_redirects_into_media_cache(
    attachment_scratch,
    allow_object_store_urls,
):
    png = b"\x89PNG\r\n\x1a\nimage"
    document = b"hello from a text attachment"
    descriptors = [
        _descriptor(
            "attachment-image",
            file_name="../../photo.png",
            media_type="image/png",
            body=png,
            kind="image",
        ),
        _descriptor(
            "attachment-file",
            file_name=r"..\..\report?.txt",
            media_type="text/plain",
            body=document,
            kind="file",
        ),
    ]
    object_urls = {
        "attachment-image": (
            "https://objects.example/image?X-Amz-Signature=image-secret"
        ),
        "attachment-file": (
            "https://objects.example/file?X-Amz-Signature=file-secret"
        ),
    }
    api_requests: list[httpx.Request] = []

    async def api_handler(request: httpx.Request) -> httpx.Response:
        api_requests.append(request)
        attachment_id = request.url.path.split("/")[2]
        return httpx.Response(
            307,
            headers={"location": object_urls[attachment_id]},
            request=request,
        )

    object_client = _ObjectStoreClient(
        download_bodies={
            object_urls["attachment-image"]: png,
            object_urls["attachment-file"]: document,
        }
    )
    adapter = _adapter()
    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url,
        headers={"Authorization": "Bearer bot-token"},
        transport=httpx.MockTransport(api_handler),
    )
    cast(Any, adapter)._new_object_store_client = lambda: object_client
    handled: list[MessageEvent] = []

    async def handle(event: MessageEvent) -> None:
        handled.append(event)

    adapter.handle_message = cast(Any, handle)
    try:
        await adapter._handle_platform_event(
            _platform_event(attachments=descriptors)
        )
    finally:
        await adapter._client.aclose()

    assert len(handled) == 1
    event = handled[0]
    assert event.text == ""
    assert event.message_type is MessageType.PHOTO
    assert event.media_types == ["image/png", "text/plain"]
    assert len(event.media_urls) == 2
    assert Path(event.media_urls[0]).is_relative_to(
        attachment_scratch / "cache" / "images"
    )
    document_path = Path(event.media_urls[1])
    assert document_path.is_relative_to(
        attachment_scratch / "cache" / "documents"
    )
    assert document_path.name.endswith("_report_.txt")
    assert document_path.read_bytes() == document
    assert [request.url.path for request in api_requests] == [
        "/attachments/attachment-image/content",
        "/attachments/attachment-file/content",
    ]
    assert all(
        request.headers["authorization"] == "Bearer bot-token"
        for request in api_requests
    )
    assert object_client.stream_calls == list(object_urls.values())
    assert allow_object_store_urls == [
        "https://objects.example/",
        "https://objects.example/",
    ]


@pytest.mark.asyncio
async def test_inbound_batch_limits_count_and_cumulative_bytes(
    attachment_scratch,
    monkeypatch,
):
    adapter = _adapter()
    descriptors = [
        _descriptor(
            f"attachment-{index}",
            file_name=f"file-{index}.txt",
            media_type="text/plain",
            body=b"x",
            kind="file",
        )
        for index in range(4)
    ]
    downloaded: list[str] = []

    async def download(descriptor: dict[str, Any]) -> bytes:
        downloaded.append(descriptor["id"])
        return b"x"

    adapter._download_attachment_bytes = cast(Any, download)
    monkeypatch.setattr(thechat, "_ATTACHMENT_INBOUND_MAX_COUNT", 3)
    monkeypatch.setattr(thechat, "_ATTACHMENT_INBOUND_TOTAL_MAX_BYTES", 2)

    media_urls, media_types, media_kinds = (
        await adapter._download_inbound_attachments(descriptors)
    )

    assert downloaded == ["attachment-0", "attachment-1"]
    assert len(media_urls) == 2
    assert media_types == ["text/plain", "text/plain"]
    assert media_kinds == ["document", "document"]


@pytest.mark.asyncio
async def test_inbound_unsafe_and_oversize_descriptors_are_safe_warnings(
    monkeypatch,
):
    monkeypatch.setattr(thechat, "_ATTACHMENT_MAX_BYTES", 8)
    adapter = _adapter()
    network_requests: list[httpx.Request] = []

    async def api_handler(request: httpx.Request) -> httpx.Response:
        network_requests.append(request)
        raise AssertionError("invalid descriptors must be rejected before download")

    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url,
        headers={"Authorization": "Bearer bot-token"},
        transport=httpx.MockTransport(api_handler),
    )
    handled: list[MessageEvent] = []

    async def handle(event: MessageEvent) -> None:
        handled.append(event)

    adapter.handle_message = cast(Any, handle)
    unsafe = _descriptor(
        "unsafe-path",
        file_name="../../etc/passwd",
        media_type="text/plain",
        body=b"x",
        kind="file",
    )
    unsafe["contentPath"] = "file:///etc/passwd"
    oversize = _descriptor(
        "too-large",
        file_name="large.bin",
        media_type="application/octet-stream",
        body=b"x" * 9,
        kind="file",
    )
    try:
        await adapter._handle_platform_event(
            _platform_event(
                attachments=[unsafe, oversize],
                text="../../etc/passwd",
            )
        )
    finally:
        await adapter._client.aclose()

    assert network_requests == []
    assert len(handled) == 1
    event = handled[0]
    assert event.media_urls == []
    assert event.media_types == []
    assert event.message_type is MessageType.TEXT
    assert event.text.startswith("../../etc/passwd\n\n[The user attempted")
    assert "file:///etc/passwd" not in event.text


@pytest.mark.asyncio
async def test_inbound_rejects_unapproved_redirect_without_leaking_query(
    caplog,
):
    secret = "never-log-this-signature"
    adapter = _adapter()

    async def api_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            307,
            headers={
                "location": (
                    "http://169.254.169.254/latest/meta-data"
                    f"?X-Amz-Signature={secret}"
                )
            },
            request=request,
        )

    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url,
        headers={"Authorization": "Bearer bot-token"},
        transport=httpx.MockTransport(api_handler),
    )
    handled: list[MessageEvent] = []
    adapter.handle_message = cast(
        Any,
        lambda event: handled.append(event),
    )

    async def handle(event: MessageEvent) -> None:
        handled.append(event)

    adapter.handle_message = cast(Any, handle)
    descriptor = _descriptor(
        "unsafe-redirect",
        file_name="note.txt",
        media_type="text/plain",
        body=b"x",
        kind="file",
    )
    with caplog.at_level(logging.WARNING, logger=thechat.__name__):
        try:
            await adapter._handle_platform_event(
                _platform_event(attachments=[descriptor])
            )
        finally:
            await adapter._client.aclose()

    assert handled[0].media_urls == []
    assert "could not be downloaded safely" in handled[0].text
    assert secret not in caplog.text


def _outbound_adapter(
    *,
    statuses: list[str],
    object_client: _ObjectStoreClient,
    api_calls: list[dict[str, Any]],
    message_status: int | list[int] = 200,
    upload_headers: dict[str, str] | None = None,
    checksum_in_query: bool = False,
) -> TheChatAdapter:
    status_queue = list(statuses)
    message_status_queue = (
        list(message_status)
        if isinstance(message_status, list)
        else [message_status]
    )
    upload_url = (
        "https://objects.example/upload"
        "?X-Amz-Signature=outbound-secret"
    )

    async def api_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        api_calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "json": body,
                "authorization": request.headers.get("authorization"),
            }
        )
        if request.method == "POST" and request.url.path == "/attachments":
            assert isinstance(body, dict)
            return _json_response(
                200,
                {
                    "attachment": {
                        "id": "attachment-outbound",
                        "status": "reserved",
                    },
                    "upload": {
                        "method": "PUT",
                        "url": (
                            f"{upload_url}&x-amz-checksum-sha256="
                            f"{quote(str(body['checksumSha256']), safe='')}"
                            if checksum_in_query
                            else upload_url
                        ),
                        "headers": (
                            upload_headers
                            if upload_headers is not None
                            else {
                                "Content-Type": body["mediaType"],
                                **(
                                    {}
                                    if checksum_in_query
                                    else {
                                        "x-amz-checksum-sha256": body[
                                            "checksumSha256"
                                        ]
                                    }
                                ),
                            }
                        ),
                        "expiresAt": "2026-07-23T12:00:00Z",
                    },
                },
                request,
            )
        if request.method == "POST" and request.url.path.endswith("/complete"):
            return _json_response(200, {"ok": True}, request)
        if request.method == "GET" and request.url.path == (
            "/attachments/attachment-outbound"
        ):
            status = status_queue.pop(0) if status_queue else statuses[-1]
            return _json_response(
                200,
                {
                    "attachment": {
                        "id": "attachment-outbound",
                        "status": status,
                    }
                },
                request,
            )
        if (
            request.method == "POST"
            and request.url.path == "/hermes-platform/messages"
        ):
            current_message_status = (
                message_status_queue.pop(0)
                if len(message_status_queue) > 1
                else message_status_queue[0]
            )
            return _json_response(
                current_message_status,
                {"messageId": "message-outbound"},
                request,
            )
        if request.method == "DELETE" and request.url.path == (
            "/attachments/attachment-outbound"
        ):
            return httpx.Response(204, request=request)
        raise AssertionError(f"Unexpected TheChat request: {request.method} {request.url}")

    adapter = _adapter()
    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url,
        headers={"Authorization": "Bearer bot-token"},
        transport=httpx.MockTransport(api_handler),
    )
    cast(Any, adapter)._new_object_store_client = lambda: object_client
    context = {
        "invocation_id": INVOCATION_ID,
        "bot_id": "bot-1",
        "conversation_id": CONVERSATION_ID,
        "thread_id": "thread-1",
    }
    adapter._contexts[
        adapter._context_key(CONVERSATION_ID, "thread-1")
    ] = context
    return adapter


def test_object_store_client_uses_connect_time_ssrf_guard(monkeypatch):
    captured: dict[str, Any] = {}
    guarded_client = object()

    def create_guarded_client(**kwargs):
        captured.update(kwargs)
        return guarded_client

    monkeypatch.setattr(
        thechat,
        "create_ssrf_safe_async_client",
        create_guarded_client,
    )

    assert _adapter()._new_object_store_client() is guarded_client
    assert captured["follow_redirects"] is False
    assert captured["headers"] == {"User-Agent": "HermesAgent/TheChat"}
    assert isinstance(captured["timeout"], httpx.Timeout)


@pytest.mark.asyncio
async def test_attachment_pipeline_retries_transient_pre_message_stage(
    monkeypatch,
):
    adapter = _adapter()
    attempts = 0

    async def send_once(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return SendResult(
                success=False,
                error="temporary upload failure",
                raw_response={"attachment_stage": "upload"},
                retryable=True,
            )
        return SendResult(success=True, message_id="message-1")

    monkeypatch.setattr(adapter, "_send_attachment_file_once", send_once)
    monkeypatch.setattr(thechat, "_ATTACHMENT_SEND_RETRY_BASE_SECONDS", 0)

    result = await adapter._send_attachment_file(
        CONVERSATION_ID,
        "/unused/file.txt",
        file_name="file.txt",
        media_type="text/plain",
        caption=None,
        reply_to=None,
        metadata=None,
    )

    assert result.success is True
    assert attempts == 2


def test_explicit_destination_rejects_foreign_cached_context(monkeypatch):
    adapter = _adapter()
    foreign_conversation = "22222222-2222-4222-8222-222222222222"
    monkeypatch.setattr(
        adapter,
        "_context_for_send",
        lambda *_args, **_kwargs: {"conversation_id": foreign_conversation},
    )

    _context, target, error = adapter._resolve_send_target(CONVERSATION_ID)

    assert target == {}
    assert error is not None
    assert error.success is False
    assert "does not match" in (error.error or "")


@pytest.mark.asyncio
async def test_outbound_image_runs_upload_lifecycle_headers_poll_and_thread(
    attachment_scratch,
    allow_object_store_urls,
):
    image_path = attachment_scratch / "generated.png"
    image_bytes = b"\x89PNG\r\n\x1a\ngenerated"
    image_path.write_bytes(image_bytes)
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["processing", "ready"],
        object_client=object_client,
        api_calls=api_calls,
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_image_file(
            CONVERSATION_ID,
            str(image_path),
            caption="caption must not be duplicated",
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is True
    reserve = api_calls[0]
    assert reserve == {
        "method": "POST",
        "path": "/attachments",
        "json": {
            "conversationId": CONVERSATION_ID,
            "fileName": "generated.png",
            "mediaType": "image/png",
            "sizeBytes": len(image_bytes),
            "checksumSha256": base64.b64encode(
                hashlib.sha256(image_bytes).digest()
            ).decode("ascii"),
        },
        "authorization": "Bearer bot-token",
    }
    assert object_client.put_calls == [
        {
            "url": (
                "https://objects.example/upload"
                "?X-Amz-Signature=outbound-secret"
            ),
            "headers": {
                "content-type": "image/png",
                "x-amz-checksum-sha256": base64.b64encode(
                    hashlib.sha256(image_bytes).digest()
                ).decode("ascii"),
                "Content-Length": str(len(image_bytes)),
            },
            "body": image_bytes,
        }
    ]
    assert [
        call["path"]
        for call in api_calls
    ] == [
        "/attachments",
        "/attachments/attachment-outbound/complete",
        "/attachments/attachment-outbound",
        "/attachments/attachment-outbound",
        "/hermes-platform/messages",
    ]
    message = api_calls[-1]["json"]
    platform_message_id = message.pop("platformMessageId")
    assert platform_message_id.startswith("hermes-attachment-")
    assert message == {
        "conversationId": CONVERSATION_ID,
        "attachmentIds": ["attachment-outbound"],
        "threadId": "thread-1",
        "content": "caption must not be duplicated",
        "invocationId": INVOCATION_ID,
        "botId": "bot-1",
    }
    assert message["content"] == "caption must not be duplicated"
    assert all(
        call["authorization"] == "Bearer bot-token"
        for call in api_calls
    )
    assert not any(
        key.lower() == "authorization"
        for key in object_client.put_calls[0]["headers"]
    )
    assert allow_object_store_urls == ["https://objects.example/"]


@pytest.mark.asyncio
async def test_outbound_document_preserves_display_name_and_file_media_type(
    attachment_scratch,
    allow_object_store_urls,
):
    document_path = attachment_scratch / "internal-name.bin"
    document_path.write_bytes(b"document")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(document_path),
            file_name="../../user-facing.txt",
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is True
    assert api_calls[0]["json"]["fileName"] == "user-facing.txt"
    assert api_calls[0]["json"]["mediaType"] == "text/plain"
    assert api_calls[-1]["json"]["attachmentIds"] == [
        "attachment-outbound"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "file_name", "expected_media_type"),
    [
        ("send_voice", "voice.mp3", "audio/mpeg"),
        ("send_voice", "voice.wav", "audio/wav"),
        ("send_video", "clip.mp4", "video/mp4"),
    ],
)
async def test_outbound_audio_and_video_use_attachment_lifecycle(
    attachment_scratch,
    allow_object_store_urls,
    method_name,
    file_name,
    expected_media_type,
):
    media_path = attachment_scratch / file_name
    media_path.write_bytes(b"generated media")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
    )
    client = adapter._client
    assert client is not None

    try:
        send_media = getattr(adapter, method_name)
        result = await send_media(
            CONVERSATION_ID,
            str(media_path),
            caption="generated media",
        )
    finally:
        await client.aclose()

    assert result.success is True
    assert api_calls[0]["json"]["fileName"] == file_name
    assert api_calls[0]["json"]["mediaType"] == expected_media_type
    assert api_calls[-1]["json"]["attachmentIds"] == [
        "attachment-outbound"
    ]
    assert api_calls[-1]["json"]["content"] == "generated media"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statuses", "poll_timeout", "expected_error", "expects_delete"),
    [
        (["rejected"], 20.0, "rejected", True),
        (["processing"], 0.0, "Timed out", False),
    ],
)
async def test_outbound_rejected_and_timeout_cleanup_policy(
    attachment_scratch,
    allow_object_store_urls,
    monkeypatch,
    statuses,
    poll_timeout,
    expected_error,
    expects_delete,
):
    monkeypatch.setattr(
        thechat,
        "_ATTACHMENT_POLL_TIMEOUT_SECONDS",
        poll_timeout,
    )
    monkeypatch.setattr(
        thechat,
        "_ATTACHMENT_POLL_INTERVAL_SECONDS",
        0.0,
    )
    document_path = attachment_scratch / "report.pdf"
    document_path.write_bytes(b"%PDF-report")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=statuses,
        object_client=object_client,
        api_calls=api_calls,
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(document_path),
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is False
    assert expected_error in (result.error or "")
    assert not any(
        call["path"] == "/hermes-platform/messages"
        for call in api_calls
    )
    deleted = any(
        call["method"] == "DELETE"
        and call["path"] == "/attachments/attachment-outbound"
        for call in api_calls
    )
    assert deleted is expects_delete
    if not expects_delete:
        assert result.retryable is True


@pytest.mark.asyncio
async def test_outbound_transport_error_redacts_presigned_url(
    attachment_scratch,
    allow_object_store_urls,
    caplog,
):
    secret = "outbound-secret"
    document_path = attachment_scratch / "report.txt"
    document_path.write_bytes(b"report")
    object_client = _ObjectStoreClient(
        upload_error=RuntimeError(
            "upload failed at "
            f"https://objects.example/upload?X-Amz-Signature={secret}"
        )
    )
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
    )
    client = adapter._client
    assert client is not None

    with caplog.at_level(logging.WARNING, logger=thechat.__name__):
        try:
            result = await adapter.send_document(
                CONVERSATION_ID,
                str(document_path),
                metadata={"thread_id": "thread-1"},
            )
        finally:
            await client.aclose()

    assert result.success is False
    assert result.error == "Attachment delivery failed (RuntimeError)"
    assert secret not in caplog.text
    assert secret not in (result.error or "")
    assert not any(
        call["path"] == "/hermes-platform/messages"
        for call in api_calls
    )
    assert any(
        call["method"] == "DELETE"
        and call["path"] == "/attachments/attachment-outbound"
        for call in api_calls
    )


@pytest.mark.asyncio
async def test_outbound_message_failure_discards_ready_attachment(
    attachment_scratch,
    allow_object_store_urls,
    monkeypatch,
):
    monkeypatch.setattr(thechat, "_ATTACHMENT_SEND_RETRY_BASE_SECONDS", 0.0)
    document_path = attachment_scratch / "report.txt"
    document_path.write_bytes(b"report")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
        message_status=503,
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(document_path),
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is False
    assert result.retryable is True
    assert [call["path"] for call in api_calls[-2:]] == [
        "/hermes-platform/messages",
        "/attachments/attachment-outbound",
    ]


@pytest.mark.asyncio
async def test_outbound_accepts_checksum_bound_in_presigned_query(
    attachment_scratch,
    allow_object_store_urls,
):
    file_bytes = b"query-bound checksum"
    file_path = attachment_scratch / "query-checksum.txt"
    file_path.write_bytes(file_bytes)
    api_calls: list[dict[str, Any]] = []
    object_client = _ObjectStoreClient()
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
        checksum_in_query=True,
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(file_path),
        )
    finally:
        await client.aclose()

    assert result.success is True
    put = object_client.put_calls[0]
    assert "x-amz-checksum-sha256" not in {
        key.lower() for key in put["headers"]
    }


@pytest.mark.asyncio
async def test_outbound_message_retry_reuses_platform_message_id(
    attachment_scratch,
    allow_object_store_urls,
    monkeypatch,
):
    monkeypatch.setattr(thechat, "_ATTACHMENT_SEND_RETRY_BASE_SECONDS", 0.0)
    document_path = attachment_scratch / "report.txt"
    document_path.write_bytes(b"report")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
        message_status=[503, 200],
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(document_path),
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is True
    message_calls = [
        call
        for call in api_calls
        if call["path"] == "/hermes-platform/messages"
    ]
    assert len(message_calls) == 2
    assert message_calls[0]["json"]["platformMessageId"] == (
        message_calls[1]["json"]["platformMessageId"]
    )
    assert not any(call["method"] == "DELETE" for call in api_calls)


@pytest.mark.asyncio
async def test_outbound_rejects_untrusted_upload_headers_and_discards_reservation(
    attachment_scratch,
    allow_object_store_urls,
):
    document_path = attachment_scratch / "report.txt"
    document_path.write_bytes(b"report")
    object_client = _ObjectStoreClient()
    api_calls: list[dict[str, Any]] = []
    adapter = _outbound_adapter(
        statuses=["ready"],
        object_client=object_client,
        api_calls=api_calls,
        upload_headers={
            "Content-Type": "text/plain",
            "x-amz-checksum-sha256": "wrong-checksum",
            "Authorization": "must-not-reach-object-storage",
        },
    )
    client = adapter._client
    assert client is not None

    try:
        result = await adapter.send_document(
            CONVERSATION_ID,
            str(document_path),
            metadata={"thread_id": "thread-1"},
        )
    finally:
        await client.aclose()

    assert result.success is False
    assert result.retryable is False
    assert "unsupported upload headers" in (result.error or "")
    assert object_client.put_calls == []
    assert [call["path"] for call in api_calls] == [
        "/attachments",
        "/attachments/attachment-outbound",
    ]
