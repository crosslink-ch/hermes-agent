"""Approval observers receive the same target identity as the user prompt."""

from tools import approval, approval_context
from tools.approval_gateway_wait import _ApprovalEntry, _await_gateway_decision


def test_gateway_approval_hooks_preserve_target_metadata(monkeypatch):
    seen = []
    monkeypatch.setattr(
        approval_context, "_fire_approval_hook",
        lambda name, **data: seen.append((name, data)),
    )
    session_key = "metadata-test"
    received = []

    def notify(data):
        received.append(data)
        with approval._lock:
            entry = approval._gateway_queues[session_key][-1]
            entry.result = "deny"
            entry.event.set()

    result = _await_gateway_decision(
        session_key, notify,
        {"command": "danger", "description": "dangerous", "pattern_key": "danger",
         "pattern_keys": ["danger"], "target": "alpha", "backend": "local"},
    )
    assert result["choice"] == "deny"
    assert received[0]["target"] == "alpha"
    for name, data in seen:
        assert name in {"pre_approval_request", "post_approval_response"}
        assert data["target"] == "alpha"
        assert data["backend"] == "local"


def test_gateway_approval_does_not_coalesce_different_targets():
    session_key = "separate-target-prompts"
    base = {"command": "danger", "description": "dangerous", "pattern_key": "danger",
            "pattern_keys": ["danger"], "backend": "local"}
    leader = _ApprovalEntry({**base, "target": "alpha"})
    with approval._lock:
        approval._gateway_queues[session_key] = [leader]
    notices = []

    def notify(data):
        notices.append(data)
        with approval._lock:
            follower = approval._gateway_queues[session_key][-1]
            assert follower is not leader
            follower.result = "deny"
            follower.event.set()

    try:
        result = _await_gateway_decision(
            session_key, notify, {**base, "target": "beta"},
        )
        assert result["choice"] == "deny"
        assert notices[0]["target"] == "beta"
    finally:
        with approval._lock:
            approval._gateway_queues.pop(session_key, None)
