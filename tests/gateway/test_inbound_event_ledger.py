from __future__ import annotations

import pytest

from gateway import inbound_event_ledger as ledger


def test_capacity_never_evicts_completed_receipts_inside_replay_window(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(ledger, "_MAX_ROWS", 1)
    monkeypatch.setattr(ledger, "_MIN_REPLAY_RETENTION_SECONDS", 600)

    ledger.accept_inbound_event(
        source="thechat:test",
        event_id="event-1",
        payload=b'{"event":1}',
        now=1_000,
    )
    claim = ledger.claim_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-1",
        lease_seconds=60,
        now=1_000,
    )
    assert claim is not None
    assert ledger.complete_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-1",
        now=1_001,
    )

    with pytest.raises(ledger.InboundEventCapacityError):
        ledger.accept_inbound_event(
            source="thechat:test",
            event_id="event-2",
            payload=b'{"event":2}',
            now=1_002,
        )

    accepted = ledger.accept_inbound_event(
        source="thechat:test",
        event_id="event-2",
        payload=b'{"event":2}',
        now=1_602,
    )
    assert accepted.status == "accepted"


def test_pending_rows_are_leased_and_reclaimable_after_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    payload = b'{"type":"thechat.hermes_platform.event"}'
    ledger.accept_inbound_event(
        source="thechat:test",
        event_id="event-1",
        payload=payload,
        now=100,
    )

    first = ledger.claim_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-1",
        lease_seconds=10,
        now=100,
    )
    assert first is not None
    assert first.payload == payload
    assert (
        ledger.claim_inbound_event(
            source="thechat:test",
            event_id="event-1",
            lease_owner="worker-2",
            lease_seconds=10,
            now=109,
        )
        is None
    )
    assert (
        ledger.list_recoverable_inbound_events(
            source="thechat:test",
            now=109,
        )
        == []
    )
    assert ledger.list_recoverable_inbound_events(
        source="thechat:test",
        now=111,
    ) == ["event-1"]

    reclaimed = ledger.claim_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-2",
        lease_seconds=10,
        now=111,
    )
    assert reclaimed is not None
    assert reclaimed.attempt == 2


def test_failed_work_respects_backoff_then_becomes_retryable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ledger.accept_inbound_event(
        source="thechat:test",
        event_id="event-1",
        payload=b"{}",
        now=200,
    )
    assert ledger.claim_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-1",
        lease_seconds=10,
        now=200,
    )
    ledger.fail_inbound_event(
        source="thechat:test",
        event_id="event-1",
        lease_owner="worker-1",
        error="retry me",
        retry_delay_seconds=5,
        now=201,
    )

    assert (
        ledger.list_recoverable_inbound_events(
            source="thechat:test",
            now=205,
        )
        == []
    )
    assert ledger.list_recoverable_inbound_events(
        source="thechat:test",
        now=206,
    ) == ["event-1"]
