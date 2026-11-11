"""TheChat durable webhook inbox: signed events are deduplicated, leased and recovered."""

import asyncio
import json
import time

import pytest

from gateway.platforms import thechat_webhook
from tests.gateway.test_thechat import (
    CONVERSATION_ID,
    INVOCATION_ID,
    _make_adapter,
    _platform_item,
    _signed_webhook_request,
)


@pytest.mark.asyncio
async def test_signed_webhook_replays_are_deduplicated_durably(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    handled = []

    async def handle(item):
        handled.append(item["invocationId"])
        return True

    adapter._handle_platform_event_safely = handle
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }
    timestamp = int(time.time())

    first, concurrent_retry = await asyncio.gather(
        adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=timestamp)
        ),
        adapter._handle_webhook(
            _signed_webhook_request(adapter, payload, timestamp=timestamp)
        ),
    )
    await asyncio.gather(*adapter._webhook_tasks)

    assert first.status == concurrent_retry.status == 200
    assert handled == [INVOCATION_ID]
    assert sorted(
        json.loads(response.text).get("duplicate", False)
        for response in (first, concurrent_retry)
    ) == [False, True]

    fresh_signature_retry = await adapter._handle_webhook(
        _signed_webhook_request(adapter, payload, timestamp=timestamp + 1)
    )
    assert fresh_signature_retry.status == 200
    assert json.loads(fresh_signature_retry.text) == {"ok": True, "duplicate": True}
    assert handled == [INVOCATION_ID]

    restarted_adapter = _make_adapter()
    restarted_adapter._webhook_secret = "whsec-test"
    restarted_adapter._handle_platform_event_safely = handle
    after_restart = await restarted_adapter._handle_webhook(
        _signed_webhook_request(
            restarted_adapter,
            payload,
            timestamp=timestamp + 2,
        )
    )
    assert after_restart.status == 200
    assert json.loads(after_restart.text) == {"ok": True, "duplicate": True}
    assert restarted_adapter._webhook_tasks == set()
    assert handled == [INVOCATION_ID]

    conflicting_payload = json.loads(json.dumps(payload))
    conflicting_payload["event"]["text"] = "different signed body"
    conflict = await restarted_adapter._handle_webhook(
        _signed_webhook_request(
            restarted_adapter,
            conflicting_payload,
            timestamp=timestamp + 3,
        )
    )
    assert conflict.status == 409
    assert handled == [INVOCATION_ID]


@pytest.mark.asyncio
async def test_accepted_webhook_is_recovered_after_worker_cancellation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"
    started = asyncio.Event()
    never = asyncio.Event()

    async def block_until_cancelled(item):
        assert item["invocationId"] == INVOCATION_ID
        started.set()
        await never.wait()
        return True

    adapter._handle_platform_event_safely = block_until_cancelled
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }

    response = await adapter._handle_webhook(_signed_webhook_request(adapter, payload))
    assert response.status == 200
    await asyncio.wait_for(started.wait(), timeout=1)

    tasks = list(adapter._webhook_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    restarted = _make_adapter()
    restarted._webhook_secret = "whsec-test"
    recovered = []

    async def handle_recovered(item):
        recovered.append(item["invocationId"])
        return True

    restarted._handle_platform_event_safely = handle_recovered
    await restarted._process_durable_webhook_event(INVOCATION_ID)

    assert recovered == [INVOCATION_ID]
    replay = await restarted._handle_webhook(
        _signed_webhook_request(restarted, payload, timestamp=int(time.time()) + 1)
    )
    assert replay.status == 200
    assert json.loads(replay.text) == {"ok": True, "duplicate": True}


@pytest.mark.asyncio
async def test_signed_webhook_fails_closed_when_replay_ledger_is_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _make_adapter()
    adapter._webhook_secret = "whsec-test"

    def fail_reservation(**_kwargs):
        raise OSError("state database unavailable")

    monkeypatch.setattr(thechat_webhook, "accept_inbound_event", fail_reservation)
    payload = {
        "type": "thechat.hermes_platform.event",
        "event": _platform_item(),
    }

    response = await adapter._handle_webhook(_signed_webhook_request(adapter, payload))

    assert response.status == 503
    assert adapter._webhook_tasks == set()


def test_webhook_payload_requires_current_envelope_and_event_shape():
    adapter = _make_adapter()
    event = _platform_item()

    assert (
        adapter._extract_webhook_event(
            {"type": "thechat.hermes_platform.event", "event": event}
        )
        is event
    )
    with pytest.raises(ValueError):
        adapter._extract_webhook_event(event)
    with pytest.raises(ValueError):
        adapter._extract_webhook_event({"type": "unknown"})
    with pytest.raises(ValueError):
        adapter._extract_webhook_event(
            {
                "type": "thechat.hermes_platform.event",
                "event": {"invocationId": "invocation-1", "chatId": CONVERSATION_ID},
            }
        )

    branch_event = _platform_item()
    branch_event["sessionIntent"] = {
        "type": "branch",
        "fromThreadId": "source-thread",
        "title": "Alternative",
    }
    assert adapter._validate_platform_event(branch_event) is branch_event

    branch_event["sessionIntent"] = {"type": "resume", "sessionId": "old-session"}
    with pytest.raises(ValueError):
        adapter._validate_platform_event(branch_event)

    for obsolete_chat_id in (
        "direct:user-1",
        "thechat:workspace:workspace-1:conversation:conversation-1:bot:bot-1",
    ):
        obsolete_event = _platform_item()
        obsolete_event["chatId"] = obsolete_chat_id
        obsolete_event["conversation"]["id"] = obsolete_chat_id
        with pytest.raises(ValueError):
            adapter._validate_platform_event(obsolete_event)

    extra_intent_event = _platform_item()
    extra_intent_event["sessionIntent"] = {
        "type": "branch",
        "fromThreadId": "source-thread",
        "title": "Alternative",
        "sessionId": "obsolete-session",
    }
    with pytest.raises(ValueError):
        adapter._validate_platform_event(extra_intent_event)
