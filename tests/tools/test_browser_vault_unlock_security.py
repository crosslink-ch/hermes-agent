"""Native manager unlock must not undo a Lock acknowledged during the UI prompt."""
import json
import subprocess
from unittest.mock import patch

from agent.vault_backends import unlock
from agent.vault_backends.bitwarden import BitwardenLoginBackend
import pytest

from tools.browser_vault_tool import browser_vault_unlock


@pytest.mark.parametrize("operation", ["unlock", "read"])
def test_backend_exceptions_never_include_cli_secrets(monkeypatch, operation):
    backend = BitwardenLoginBackend({"binary_path": "/fixture/bw"})
    unlock.store_session_token("bitwarden", "synthetic-token")
    def explode(*_a, **_kw):
        raise RuntimeError("synthetic master SECRET STDERR synthetic-token")
    monkeypatch.setattr("agent.vault_backends.bitwarden.run_with_secret_env", explode)
    monkeypatch.setattr("agent.vault_backends.bitwarden.run_cli", explode)
    with pytest.raises(RuntimeError) as failure:
        backend.unlock("synthetic master") if operation == "unlock" else backend.resolve_password("bw:item")
    assert "synthetic" not in str(failure.value)
    assert "SECRET STDERR" not in str(failure.value)
    assert failure.value.__suppress_context__ is True
    unlock.lock("bitwarden")


def test_nonowner_cannot_use_already_unlocked_external_manager(monkeypatch):
    from tools import browser_vault_tool as tool
    from agent.vault_store import VaultItemMeta
    backend = BitwardenLoginBackend({"binary_path": "/fixture/bw"})
    unlock.store_session_token("bitwarden", "synthetic-token")
    calls = []
    monkeypatch.setattr(backend, "list_items", lambda: calls.append("list") or [
        VaultItemMeta(id="bw:item", kind="login", label="private", origin="https://example.test",
                      created_at="", identifier="owner@example.test")])
    monkeypatch.setattr(tool, "_focus_bound_origin", lambda *_: calls.append("browser"))
    with patch("agent.vault_backends.enabled_backends", return_value=[backend]), \
         patch("agent.vault_backends.base.enabled_backends", return_value=[backend]):
        with unlock.external_access_scope(False):
            listed = json.loads(tool.browser_vault_list())
            assert not listed["items"]
            for result in (tool.browser_vault_unlock("bitwarden"), tool.browser_vault_fill("bw:item"),
                           tool.browser_vault_enter_code("bw:item")):
                assert json.loads(result)["error_type"] == "external_vault_access_denied"
    assert calls == []
    unlock.lock("bitwarden")


def test_unlock_errors_do_not_echo_master_or_cli_stderr(monkeypatch, caplog):
    backend = BitwardenLoginBackend({"binary_path": "/fixture/bw"})
    unlock.lock("bitwarden")
    unlock.set_unlock_prompt_callback(lambda *_: "synthetic master")
    monkeypatch.setattr("agent.vault_backends.bitwarden.run_with_secret_env",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "synthetic master SECRET STDERR"))
    try:
        with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            result = browser_vault_unlock("bitwarden")
        assert json.loads(result)["error_type"] == "unlock_failed"
        assert "synthetic master" not in result + caplog.text
        assert "SECRET STDERR" not in result + caplog.text
        with __import__("pytest").raises(RuntimeError) as failure:
            backend.unlock("synthetic master")
        assert "synthetic master" not in str(failure.value)
        assert "SECRET STDERR" not in str(failure.value)
    finally:
        unlock.set_unlock_prompt_callback(None)
        unlock.lock("bitwarden")


def test_lock_during_masked_prompt_cannot_resurrect_unlock(monkeypatch):
    backend = BitwardenLoginBackend({"binary_path": "/fixture/bw"})
    unlock.lock("bitwarden")

    def prompt(*_):
        unlock.lock("bitwarden")
        return "synthetic master"

    unlock.set_unlock_prompt_callback(prompt)
    monkeypatch.setattr("agent.vault_backends.bitwarden.run_with_secret_env",
                        lambda *a, **kw: subprocess.CompletedProcess(a, 0, "synthetic-token", ""))
    try:
        with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            result = json.loads(browser_vault_unlock("bitwarden"))
        assert result["success"] is False
        assert not unlock.is_unlocked("bitwarden")
    finally:
        unlock.set_unlock_prompt_callback(None)
        unlock.lock("bitwarden")


def test_session_release_is_profile_scoped_even_when_session_ids_match(monkeypatch, tmp_path):
    a, b = str(tmp_path / "profile-A"), str(tmp_path / "profile-B")
    unlock.set_current_session_id("same-session-id")
    try:
        monkeypatch.setenv("HERMES_HOME", a)
        unlock.store_session_token("bitwarden", "token-A")
        monkeypatch.setenv("HERMES_HOME", b)
        unlock.store_session_token("bitwarden", "token-B")
        unlock.release_session("same-session-id")
        assert not unlock.is_unlocked("bitwarden")
        monkeypatch.setenv("HERMES_HOME", a)
        assert unlock.get_session_token("bitwarden") == "token-A"
    finally:
        unlock.set_current_session_id(None)
        unlock.lock_all_profiles()


def test_expired_token_does_not_keep_previous_session_as_prompt_owner(monkeypatch):
    unlock.set_current_session_id("previous-session")
    unlock.store_session_token("bitwarden", "expired-token")
    key = unlock._key("bitwarden")
    token, last = unlock._sessions[key]
    unlock._sessions[key] = (token, last - unlock._IDLE_TTL_S - 1)
    assert not unlock.is_unlocked("bitwarden")
    unlock.set_current_session_id("new-session")
    try:
        with unlock.unlock_attempt("bitwarden"):
            unlock.release_session("new-session")
            assert not unlock.unlock_attempt_is_current("bitwarden")
    finally:
        unlock.set_current_session_id(None)
        unlock.lock("bitwarden")
