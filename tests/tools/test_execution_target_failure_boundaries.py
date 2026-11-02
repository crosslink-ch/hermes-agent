"""Failure boundaries preserve target/profile authority and redact diagnostics."""

import pytest


def test_profile_identity_failure_never_collapses_approval_scope(monkeypatch):
    from agent import secret_scope
    import hermes_constants
    from tools import approval, execution_targets

    def unavailable_home():
        raise OSError("profile identity unavailable")

    monkeypatch.setattr(secret_scope, "is_multiplex_active", lambda: True)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", unavailable_home)
    with pytest.raises(OSError, match="profile identity unavailable"):
        execution_targets._active_profile_scope()
    with pytest.raises(OSError, match="profile identity unavailable"):
        approval._execution_scoped_pattern_key("command", "box", named=True)


def test_ssh_identity_failure_never_uses_process_profile(monkeypatch):
    from hermes_cli import profiles
    from tools.environments import ssh

    def unavailable_profile():
        raise OSError("profile identity unavailable")

    monkeypatch.setenv("HERMES_PROFILE", "launch-profile")
    monkeypatch.setattr(profiles, "get_active_profile_name", unavailable_profile)
    with pytest.raises(OSError, match="profile identity unavailable"):
        ssh._active_profile_identity()


def test_failed_approval_redaction_logs_traceback_without_payload(monkeypatch, caplog):
    from agent import redact
    from tools import approval, approval_smart

    secret = "sensitive-test-payload-not-for-logs"

    def broken_redaction(*_args, **_kwargs):
        raise RuntimeError(secret)

    hooks = []
    monkeypatch.setattr(redact, "redact_sensitive_text", broken_redaction)
    monkeypatch.setattr(approval_smart, "_smart_approve", lambda *_args: "approve")
    monkeypatch.setattr(approval.approval_context, "_fire_approval_hook", lambda *a, **kw: hooks.append((a, kw)))
    verdict = approval._target_smart_verdict(
        secret, secret, "pattern", ["pattern"], "session", "box", "ssh",
    )
    assert verdict == "approve"
    assert hooks == []
    assert "Approval observer redaction failed" in caplog.text
    assert secret not in caplog.text
    assert any(record.exc_info for record in caplog.records)
