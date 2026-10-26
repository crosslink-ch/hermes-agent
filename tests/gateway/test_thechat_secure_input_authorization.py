"""Canonical resolution denial must fence every v3 consumer, not just owner cache."""
import asyncio
import hashlib

import httpx
import pytest

from gateway.run_turn_runner_secure_input import TurnSecureInput
from hermes_constants import get_hermes_home
from tests.gateway.test_thechat_secure_input_consumers import ReplyClient, prepare, run_tool
from tests.gateway.test_thechat_vault_unlock import _context, _turn


class DeniedResolutionClient(ReplyClient):
    rejected = False

    async def post(self, path, json):
        if json.get("type") == "secure.input.resolved":
            self.rejected = True
            self.posts.append((path, json))
            # Real API owner/admission transfer without a transport reconnect.
            httpx.Response(403, request=httpx.Request("POST", "https://synthetic.test" + path),
                           json={"error": "Could not deliver the secure input response"}).raise_for_status()
        return await super().post(path, json)


@pytest.mark.asyncio
async def test_server403_with_unchanged_cached_owner_never_persists_login(monkeypatch):
    from agent.vault_store import get_vault_store
    from tools import browser_vault_tool as tool
    adapter = prepare(monkeypatch, {"identifier": "synthetic@example.test", "password": "synthetic-password"})
    adapter._client = DeniedResolutionClient(adapter, adapter._client.values)
    monkeypatch.setattr(tool, "browser_vault_fill", lambda *_a, **_kw: pytest.fail("denial reached browser"))
    result = await run_tool(adapter, "browser_vault_save_login", {})
    assert adapter._client.rejected and adapter._owner_user_id == "owner"
    assert not result["success"]
    assert not get_vault_store().list_items()
    assert not adapter._secure_input.pending and not adapter._secure_input._waiters


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,target,values", [
    ("onepassword.unlock", "1Password", {"password": "synthetic-password"}),
    ("vault.save_login", "https://example.test", {"identifier": "synthetic", "password": "synthetic-password"}),
    ("vault.code", "https://example.test", {"code": "synthetic-code"}),
    ("sudo.password", "Privileged terminal command", {"password": "synthetic-password"}),
    ("skill.secret", "SYNTHETIC_TEST_KEY", {"value": "synthetic-value"}),
])
@pytest.mark.parametrize("failure", ["server403", "exception", "timeout", "unavailable"])
async def test_shared_handoff_clears_taken_answer_on_resolution_failure(monkeypatch, kind, target, values, failure):
    from concurrent.futures import Future
    from gateway.platforms.thechat_secure_input import PendingInput
    import gateway.run_turn_runner_secure_input as runtime
    from hermes_cli.config import atomic_config_write

    atomic_config_write(get_hermes_home() / "config.yaml", {"platforms": {"thechat": {"secure_input": {
        "allow_sudo": True, "allow_skill_secrets": True}}}})
    adapter = prepare(monkeypatch, values)
    if failure == "server403":
        adapter._client = DeniedResolutionClient(adapter, values)
    original_publish = runtime._publish
    class TimedOutFuture(Future):
        def result(self, timeout=None):
            assert timeout is not None  # bounded wait, no real sleep
            raise TimeoutError("synthetic timeout")
    def publish(*args, **kwargs):
        if args[3]["type"] == "secure.input.resolved" and failure != "server403":
            if failure == "exception":
                raise RuntimeError("synthetic publish exception")
            if failure == "unavailable":
                return None
            return TimedOutFuture()
        return original_publish(*args, **kwargs)
    monkeypatch.setattr(runtime, "_publish", publish)
    taken = []
    original_take = PendingInput.take_response
    def take(record):
        answer = original_take(record)
        if answer is not None:
            taken.append(answer)
        return answer
    monkeypatch.setattr(PendingInput, "take_response", take)
    ctx = _context()
    turn = _turn(adapter, ctx, type("Agent", (), {"is_interrupted": False})())
    home = str(get_hermes_home())
    prompts = TurnSecureInput(turn, adapter, ctx, hashlib.sha256(home.encode()).hexdigest(), home, True, "owner")
    try:
        result = await asyncio.to_thread(prompts.request, kind, target)
        assert result is None
        assert taken == [{}], "decrypted answer must be cleared, not just hidden by a later sink"
        assert adapter._owner_user_id == "owner"
        assert not adapter._secure_input.pending and not adapter._secure_input._waiters
    finally:
        prompts.close()
