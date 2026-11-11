"""TheChat direct interactions: a signed approval/clarify callback resolves its exact waiter."""

import asyncio
import json
import time
from typing import cast

import httpx
import pytest

from gateway.platforms import thechat_webhook
from tests.gateway.test_thechat import (
    CONVERSATION_ID,
    INVOCATION_ID,
    OTHER_CONVERSATION_ID,
    OTHER_INVOCATION_ID,
    SESSION_KEY,
    _context,
    _drain_webhook_tasks,
    _FakeClient,
    _FakeResponse,
    _interaction_payload,
    _make_adapter,
    _signed_webhook_request,
)


@pytest.mark.asyncio
async def test_direct_approval_resolves_exact_waiter_without_message_dispatch(
    tmp_path, monkeypatch
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-direct"
    entry = _ApprovalEntry({"command": "rm -rf /important"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    dispatched = []

    async def must_not_dispatch(item):
        dispatched.append(item)
        return True

    adapter._handle_platform_event_safely = must_not_dispatch
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "rm -rf /important",
        SESSION_KEY,
        request_id=request_id,
    )
    adapter._typing_paused.add(CONVERSATION_ID)
    payload = _interaction_payload(request_id=request_id, response="once")
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)

        assert response.status == 200
        assert json.loads(response.text) == {"ok": True, "duplicate": False}
        assert entry.event.is_set()
        assert entry.result == "once"
        assert SESSION_KEY not in _gateway_queues
        assert CONVERSATION_ID not in adapter._typing_paused
        assert dispatched == []
        assert request_id not in adapter._interaction_requests
        resolved = cast(_FakeClient, adapter._client).posts[-1]
        assert resolved["path"] == (
            f"/hermes-platform/invocations/{INVOCATION_ID}/progress"
        )
        assert resolved["json"]["type"] == "approval.resolved"
        assert resolved["json"]["payload"] == {
            "requestId": request_id,
            "choice": "once",
            "sessionKey": SESSION_KEY,
            "resolveAll": False,
            "resolvedCount": 1,
        }
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_approval_cannot_jump_fifo_head(tmp_path, monkeypatch):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    first = _ApprovalEntry({"command": "first"}, request_id="approval-first")
    second = _ApprovalEntry({"command": "second"}, request_id="approval-second")
    _gateway_queues[SESSION_KEY] = [first, second]
    await adapter.send_exec_approval(
        CONVERSATION_ID, "first", SESSION_KEY, request_id=first.request_id
    )
    await adapter.send_exec_approval(
        CONVERSATION_ID, "second", SESSION_KEY, request_id=second.request_id
    )
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter,
                _interaction_payload(
                    interaction_id="later-approval-click",
                    request_id=second.request_id,
                ),
            )
        )

        assert response.status == 409
        assert not first.event.is_set()
        assert not second.event.is_set()
        assert _gateway_queues[SESSION_KEY] == [first, second]
        assert second.request_id in adapter._interaction_requests
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_webhook_acks_before_blocked_resolution_publication(
    tmp_path, monkeypatch
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    class _BlockingResolutionClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.resolution_started = asyncio.Event()
            self.release_resolution = asyncio.Event()

        async def post(self, path, json):
            self.posts.append({"path": path, "json": json})
            if json.get("type") == "approval.resolved":
                self.resolution_started.set()
                await self.release_resolution.wait()
            return _FakeResponse()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    client = _BlockingResolutionClient()
    adapter._client = cast(httpx.AsyncClient, client)
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    entry = _ApprovalEntry({"command": "danger"}, request_id="approval-fast-ack")
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=entry.request_id,
    )
    try:
        response = await asyncio.wait_for(
            adapter._handle_webhook(
                _signed_webhook_request(
                    adapter,
                    _interaction_payload(
                        interaction_id="fast-ack",
                        request_id=entry.request_id,
                    ),
                )
            ),
            timeout=1.0,
        )

        assert response.status == 200
        assert entry.event.is_set()
        await asyncio.wait_for(client.resolution_started.wait(), timeout=1.0)
        assert entry.request_id in adapter._interaction_requests

        client.release_resolution.set()
        await _drain_webhook_tasks(adapter)
        assert entry.request_id not in adapter._interaction_requests
        assert [post["json"]["type"] for post in client.posts] == [
            "approval.request",
            "approval.resolved",
        ]
    finally:
        client.release_resolution.set()
        await _drain_webhook_tasks(adapter)
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_webhook_acks_when_ledger_completion_needs_retry(
    tmp_path, monkeypatch
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    entry = _ApprovalEntry({"command": "danger"}, request_id="approval-ledger-retry")
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=entry.request_id,
    )
    real_complete = thechat_webhook.complete_inbound_event
    completion_attempts = 0

    def flaky_complete(**kwargs):
        nonlocal completion_attempts
        completion_attempts += 1
        if completion_attempts < 3:
            raise OSError("temporary ledger failure")
        return real_complete(**kwargs)

    monkeypatch.setattr(thechat_webhook, "complete_inbound_event", flaky_complete)
    payload = _interaction_payload(
        interaction_id="ledger-retry",
        request_id=entry.request_id,
    )
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )

        assert response.status == 200
        assert entry.event.is_set()
        await _drain_webhook_tasks(adapter)
        assert completion_attempts == 3

        replay = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter,
                payload,
                timestamp=int(time.time()) + 1,
            )
        )
        assert json.loads(replay.text) == {"ok": True, "duplicate": True}
    finally:
        await _drain_webhook_tasks(adapter)
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_response", "multi_select", "expected_waiter_response"),
    [
        ("custom answer", False, "custom answer"),
        (["alpha", "beta"], True, '["alpha", "beta"]'),
    ],
)
async def test_direct_clarify_resolves_string_or_json_encoded_list(
    tmp_path,
    monkeypatch,
    direct_response,
    multi_select,
    expected_waiter_response,
):
    from tools import clarify_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "clarify-request-direct"
    clarify_gateway.register(
        request_id,
        SESSION_KEY,
        "Choose targets",
        ["alpha", "beta"],
        multi_select=multi_select,
    )
    await adapter.send_clarify(
        CONVERSATION_ID,
        "Choose targets",
        ["alpha", "beta"],
        request_id,
        SESSION_KEY,
    )
    adapter._typing_paused.add(CONVERSATION_ID)
    payload = _interaction_payload(
        interaction_id=f"progress-{int(multi_select)}",
        request_type="clarify.request",
        request_id=request_id,
        response=direct_response,
    )
    try:
        webhook_response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)

        assert webhook_response.status == 200
        assert clarify_gateway.wait_for_response(request_id, timeout=0.1) == (
            expected_waiter_response
        )
        assert CONVERSATION_ID not in adapter._typing_paused
        assert request_id not in adapter._interaction_requests
        resolved = cast(_FakeClient, adapter._client).posts[-1]
        assert resolved["path"] == (
            f"/hermes-platform/invocations/{INVOCATION_ID}/progress"
        )
        assert resolved["json"]["type"] == "clarify.resolved"
        assert resolved["json"]["payload"] == {
            "requestId": request_id,
            "sessionKey": SESSION_KEY,
            "response": direct_response,
        }
    finally:
        clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_typed_clarify_winner_rejects_late_direct_callback_and_clears_card(
    tmp_path, monkeypatch
):
    from tools import clarify_gateway

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "clarify-typed-winner"
    clarify_gateway.register(
        request_id,
        SESSION_KEY,
        "Pick one",
        ["alpha", "beta"],
    )
    await adapter.send_clarify(
        CONVERSATION_ID,
        "Pick one",
        ["alpha", "beta"],
        request_id,
        SESSION_KEY,
    )
    resolved = {}
    assert clarify_gateway.resolve_text_response_for_session(
        SESSION_KEY,
        "2",
        resolved=resolved,
    ) is True

    losing = await adapter._handle_webhook(
        _signed_webhook_request(
            adapter,
            _interaction_payload(
                interaction_id="late-direct-clarify",
                request_type="clarify.request",
                request_id=request_id,
                response="alpha",
            ),
        )
    )
    assert losing.status == 409
    assert request_id in adapter._interaction_requests

    published = await adapter.send_clarify_resolution(
        chat_id=CONVERSATION_ID,
        session_key=SESSION_KEY,
        request_id=resolved["request_id"],
        response=resolved["response"],
    )
    assert published.success is True
    assert request_id not in adapter._interaction_requests
    assert cast(_FakeClient, adapter._client).posts[-1]["json"] == {
        "botId": "bot-1",
        "conversationId": CONVERSATION_ID,
        "type": "clarify.resolved",
        "status": "completed",
        "label": "Input received",
        "payload": {
            "requestId": request_id,
            "sessionKey": SESSION_KEY,
            "response": "beta",
        },
    }
    assert clarify_gateway.wait_for_response(request_id, timeout=0.1) == "beta"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        {"requestId": "unknown-request"},
        {"sessionKey": "agent:main:thechat:dm:other"},
        {"invocationId": OTHER_INVOCATION_ID},
        {"conversationId": OTHER_CONVERSATION_ID},
        {"threadId": "other-thread"},
        {"requestType": "clarify.request", "response": "answer"},
    ],
)
async def test_direct_interaction_rejects_cross_context_resolution(
    tmp_path, monkeypatch, mutation
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-isolated"
    entry = _ApprovalEntry({"command": "danger"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    payload = _interaction_payload(
        interaction_id="context-" + next(iter(mutation)),
        request_id=request_id,
    )
    payload["interaction"].update(mutation)
    try:
        response = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )

        assert response.status == 409
        assert not entry.event.is_set()
        assert _gateway_queues[SESSION_KEY] == [entry]
    finally:
        _gateway_queues.pop(SESSION_KEY, None)


@pytest.mark.asyncio
async def test_direct_interaction_returns_stale_when_waiter_is_gone(
    tmp_path, monkeypatch
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-expired"
    _gateway_queues[SESSION_KEY] = [
        _ApprovalEntry({"command": "danger"}, request_id=request_id)
    ]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    _gateway_queues.pop(SESSION_KEY)

    response = await adapter._handle_webhook(
        _signed_webhook_request(
            adapter,
            _interaction_payload(
                interaction_id="progress-stale",
                request_id=request_id,
            ),
        )
    )

    assert response.status == 409
    assert request_id not in adapter._interaction_requests


def test_direct_interaction_envelope_validation_is_strict_and_bounded():
    adapter = _make_adapter()
    valid = _interaction_payload()
    assert adapter._extract_webhook_interaction(valid) is valid["interaction"]

    invalid_payloads = []
    extra = json.loads(json.dumps(valid))
    extra["interaction"]["extra"] = True
    invalid_payloads.append(extra)
    invalid_uuid = json.loads(json.dumps(valid))
    invalid_uuid["interaction"]["invocationId"] = "not-a-uuid"
    invalid_payloads.append(invalid_uuid)
    oversized = json.loads(json.dumps(valid))
    oversized["interaction"]["sessionKey"] = "s" * 1025
    invalid_payloads.append(oversized)
    wrong_approval_shape = json.loads(json.dumps(valid))
    wrong_approval_shape["interaction"]["response"] = ["once"]
    invalid_payloads.append(wrong_approval_shape)
    wrong_request_type_shape = json.loads(json.dumps(valid))
    wrong_request_type_shape["interaction"]["requestType"] = ["approval.request"]
    invalid_payloads.append(wrong_request_type_shape)
    empty_clarify = _interaction_payload(
        request_type="clarify.request", response=[]
    )
    invalid_payloads.append(empty_clarify)

    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            adapter._extract_webhook_interaction(payload)


@pytest.mark.asyncio
async def test_direct_interaction_auth_and_malformed_statuses(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    payload = _interaction_payload()

    bad_signature = _signed_webhook_request(adapter, payload)
    bad_signature.headers["X-Webhook-Signature"] = "0" * 64
    unauthorized = await adapter._handle_webhook(bad_signature)
    assert unauthorized.status == 401

    malformed_payload = _interaction_payload()
    malformed_payload["interaction"]["response"] = ["once"]
    malformed = await adapter._handle_webhook(
        _signed_webhook_request(adapter, malformed_payload)
    )
    assert malformed.status == 400


@pytest.mark.asyncio
async def test_direct_interaction_duplicate_and_conflicting_reuse(
    tmp_path, monkeypatch
):
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    adapter._contexts[CONVERSATION_ID] = _context(INVOCATION_ID)
    request_id = "approval-request-dedupe"
    entry = _ApprovalEntry({"command": "danger"}, request_id=request_id)
    _gateway_queues[SESSION_KEY] = [entry]
    await adapter.send_exec_approval(
        CONVERSATION_ID,
        "danger",
        SESSION_KEY,
        request_id=request_id,
    )
    payload = _interaction_payload(
        interaction_id="progress-dedupe",
        request_id=request_id,
        response="once",
    )
    try:
        first = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload)
        )
        await _drain_webhook_tasks(adapter)
        duplicate = await adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=int(time.time()) + 1)
        )
        conflict_payload = json.loads(json.dumps(payload))
        conflict_payload["interaction"]["response"] = "deny"
        conflict = await adapter._handle_webhook(
            _signed_webhook_request(
                adapter, conflict_payload, timestamp=int(time.time()) + 2
            )
        )

        assert first.status == 200
        assert json.loads(duplicate.text) == {"ok": True, "duplicate": True}
        assert conflict.status == 409
        assert entry.result == "once"
        resolution_posts = [
            post
            for post in cast(_FakeClient, adapter._client).posts
            if post["json"].get("type") == "approval.resolved"
        ]
        assert len(resolution_posts) == 1
    finally:
        _gateway_queues.pop(SESSION_KEY, None)
