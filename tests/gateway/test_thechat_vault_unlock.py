"""Encrypted native unlock uses the dedicated signed, memory-only relay."""
import base64
import json
import os
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _context():
    return {"invocation_id": str(uuid.uuid4()), "bot_id": "bot-id", "conversation_id": str(uuid.uuid4()),
            "thread_id": None, "requester_user_id": "owner"}


def _encrypt(record, password="  synthetic é master\n", *, aad=None):
    public = serialization.load_der_public_key(base64.b64decode(record.payload["publicKeySpkiB64"]))
    assert public.key_size == 3072
    key, iv = os.urandom(32), os.urandom(12)
    label = record.aad if aad is None else aad
    wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=label))
    cipher = AESGCM(key).encrypt(iv, password.encode("utf-8"), label)
    return {"wrappedKeyB64": base64.b64encode(wrapped).decode(), "ivB64": base64.b64encode(iv).decode(),
            "ciphertextB64": base64.b64encode(cipher).decode()}


def _relay(record, **overrides):
    p = record.payload
    body = {k: v for k, v in p.items() if k != "publicKeySpkiB64"}
    body.update(id="progress-event", requestType="vault.unlock.request", invocationId=record.context["invocation_id"],
                conversationId=record.context["conversation_id"], threadId=record.context["thread_id"],
                actorUserId="owner", action="submit", **_encrypt(record))
    body.update(overrides)
    return {"type": "thechat.hermes_platform.vault_unlock", "interaction": body}


def test_hybrid_unlock_delivers_exact_utf8_only_to_memory_waiter():
    from gateway.platforms import thechat_vault
    broker = thechat_vault.VaultUnlockBroker()
    context = _context()
    record = broker.create(context=context, owner_user_id="owner", profile_id="opaque-profile", session_key="session-key")
    p = record.payload
    assert set(p) == {"version", "requestId", "sessionKey", "profileId", "backend", "ownerUserId", "requesterUserId",
                      "nonce", "expiresAt", "algorithm", "publicKeySpkiB64"}
    assert p["backend"] == "bitwarden"
    assert p["algorithm"] == "RSA-OAEP-3072-SHA256+A256GCM"
    assert len(base64.urlsafe_b64decode(p["nonce"] + "=")) == 32
    assert json.loads(record.aad) == [1, "bot-id", "owner", "owner", "opaque-profile", "session-key",
                                    context["invocation_id"], context["conversation_id"], None, p["requestId"],
                                    "bitwarden", p["nonce"], p["expiresAt"]]
    payload = _relay(record)
    assert broker.resolve(payload, owner_user_id="owner") is False
    assert record.take_response() == "  synthetic é master\n"
    assert record.take_response() == ""
    assert record.outcome == "submitted"
    assert broker.resolve(payload, owner_user_id="owner") is True
    assert p["requestId"] not in broker.pending
    assert "synthetic" not in repr(record) + repr(broker.tombstones)


import asyncio
import hashlib
import hmac
import time
from types import SimpleNamespace
import pytest


class _Response:
    status_code = 200
    def __init__(self, data):
        self.data = data
    def json(self):
        return self.data
    def raise_for_status(self):
        pass


class _Client:
    def __init__(self):
        self.posts = []
        self.request_ready = asyncio.Event()
        self.closed = False
    async def get(self, *_a, **_kw):
        return _Response({"ok": True, "ownerUserId": "owner"})
    async def post(self, path, json):
        self.posts.append((path, json))
        if json.get("type") == "vault.unlock.request":
            self.request_ready.set()
        return _Response({"event": {"id": "progress-event"}})
    async def aclose(self):
        self.closed = True


def _adapter():
    from gateway.config import PlatformConfig
    from gateway.platforms.thechat import TheChatAdapter
    adapter = TheChatAdapter(PlatformConfig(enabled=True, token="synthetic-bot-token", extra={
        "base_url": "http://thechat.test", "webhook_url": "http://gateway.test/thechat/webhook"}))
    adapter._client = _Client()
    adapter._webhook_secret = "synthetic-signature-key"
    return adapter


def _signed(adapter, payload):
    body = json.dumps(payload, separators=(",", ":"))
    timestamp = str(int(time.time()))
    signature = hmac.new(adapter._webhook_secret.encode(), (timestamp + "." + body).encode(), hashlib.sha256).hexdigest()
    class Request:
        headers = {"X-Webhook-Timestamp": timestamp, "X-Webhook-Signature": signature}
        async def read(self):
            return body.encode()
    return Request()


def _turn(adapter, context, agent, *, sender="owner"):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from gateway.run_turn_runner import TurnRunner
    source = SessionSource(platform=Platform.THECHAT, chat_id=context["conversation_id"], user_id=sender,
                           message_id="original-message", thread_id=context["thread_id"])
    adapter._event_contexts["original-message"] = context
    ctx = TurnContext(source=source, session_key="session-key", session_id="hermes-session", message="unlock",
                      _loop_for_step=asyncio.get_running_loop(), agent_holder=[agent], _run_still_current=lambda: True)
    runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter, _consume_pending_native_image_paths=lambda _s: [])
    return TurnRunner(runner, ctx)


@pytest.mark.asyncio
@pytest.mark.linux_only
async def test_actual_executor_scope_native_tool_signed_relay_never_uses_inbox(tmp_path, monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from tools import browser_vault_tool  # registers the native tools
    from tools.registry import registry
    from tools.thread_context import propagate_context_to_thread
    from concurrent.futures import ThreadPoolExecutor
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    context = _context()
    executable = tmp_path / "bw"
    executable.write_text("#!/usr/bin/env python3\nimport os,sys\nassert os.environ['HERMES_BW_MASTER'] == '  synthetic é master\\n'\nprint('synthetic-token')\n")
    executable.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(executable)})
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    monkeypatch.setattr("gateway.platforms.thechat.accept_inbound_event", lambda **kw: pytest.fail("unlock reached durable inbox"))
    observed = []
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            observed.append(unlock.get_unlock_prompt_callback())
            with ThreadPoolExecutor(max_workers=1) as pool:
                return json.loads(pool.submit(propagate_context_to_thread(lambda: registry.dispatch("browser_vault_unlock", {"backend": "bitwarden"}))).result())
    agent = Agent()
    turn = _turn(adapter, context, agent)
    unlock.lock("bitwarden")
    old_prompt = lambda *_: "old callback must not run"
    def invoke():
        unlock.set_unlock_prompt_callback(old_prompt)
        unlock.set_current_session_id("previous-session")
        try:
            result = turn._run_conversation_with_approval(agent, [], None, None, None)
            assert unlock.get_unlock_prompt_callback() is old_prompt
            assert unlock.get_current_session_id() == "previous-session"
            return result
        finally:
            unlock.set_unlock_prompt_callback(None)
            unlock.set_current_session_id(None)
    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        await asyncio.wait_for(adapter._client.request_ready.wait(), 3)
        record = next(iter(adapter._vault_unlock.pending.values()))
        relay = _relay(record)
        response = await adapter._handle_webhook(_signed(adapter, relay))
        assert response.status == 200
        assert json.loads(response.text) == {"ok": True, "duplicate": False}
        duplicate = await adapter._handle_webhook(_signed(adapter, relay))
        assert json.loads(duplicate.text) == {"ok": True, "duplicate": True}
        conflicting = json.loads(json.dumps(relay))
        conflicting["interaction"]["id"] = "conflicting-event"
        conflict = await adapter._handle_webhook(_signed(adapter, conflicting))
        assert conflict.status == 409
        result = await asyncio.wait_for(task, 5)
        assert result == {"success": True, "backend": "bitwarden"}
        assert unlock.is_unlocked("bitwarden")
        unlock.release_session("unrelated-session")
        assert unlock.is_unlocked("bitwarden")
        unlock.release_session("hermes-session")
        assert not unlock.is_unlocked("bitwarden")
        assert observed and observed[0] is not old_prompt
        events = [p for _, p in adapter._client.posts if p.get("type", "").startswith("vault.unlock.")]
        assert [p["type"] for p in events] == ["vault.unlock.request", "vault.unlock.resolved"]
        assert events[0]["label"] == "Unlock Bitwarden" and events[0]["status"] == "waiting"
        assert events[1]["payload"] == {"version": 1, "requestId": record.payload["requestId"],
                                       "sessionKey": "session-key", "outcome": "submitted"}
        assert "synthetic é master" not in json.dumps(events) + json.dumps(result)
    finally:
        if not task.done():
            agent.is_interrupted = True
            await asyncio.wait_for(task, 5)
        await adapter.disconnect()
        unlock.lock("bitwarden")


@pytest.mark.parametrize("mutation", [
    {"actorUserId": "nonowner"}, {"ownerUserId": "nonowner"}, {"requesterUserId": "nonowner"},
    {"profileId": "other-profile"}, {"sessionKey": "other-session"},
    {"invocationId": "11111111-1111-4111-8111-111111111111"},
    {"conversationId": "11111111-1111-4111-8111-111111111111"}, {"threadId": "other-thread"},
    {"requestId": "11111111-1111-4111-8111-111111111111"}, {"nonce": "other-nonce"},
    {"expiresAt": 1}, {"expiresAt": True}, {"version": True}, {"version": 2},
    {"algorithm": "RSA-OAEP"}, {"backend": "onepassword"}, {"requestType": "clarify.request"},
    {"password": "synthetic forbidden"}, {"response": "synthetic forbidden"}, {"privateKey": "synthetic forbidden"},
    {"wrappedKeyB64": base64.b64encode(b"x" * 383).decode()},
    {"ivB64": base64.b64encode(b"x" * 13).decode()},
    {"ciphertextB64": base64.b64encode(b"x" * 16).decode()},
    {"ciphertextB64": base64.b64encode(b"x" * 4113).decode()}, {"ivB64": "not base64"},
])
def test_relay_rejects_unknown_or_conflicting_context_and_wrong_lengths(mutation):
    from gateway.platforms.thechat_vault import VaultUnlockBroker, VaultUnlockError
    broker = VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="profile-A", session_key="session-A")
    with pytest.raises(VaultUnlockError) as failure:
        broker.resolve(_relay(record, **mutation), owner_user_id="owner")
    assert str(failure.value) == "Invalid or stale vault unlock interaction"
    assert not record.event.is_set() and record.take_response() == ""
    assert record.payload["requestId"] in broker.pending
    broker.close()
    assert record._private_key is None


@pytest.mark.parametrize("attack", ["oaep-label", "tag", "aes-label", "key-length", "utf8", "oversize", "empty"])
def test_crypto_tampering_cannot_deliver_password(attack):
    from gateway.platforms.thechat_vault import VaultUnlockBroker, VaultUnlockError
    broker = VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    relay = _relay(record)
    if attack == "oaep-label":
        relay["interaction"].update(_encrypt(record, aad=b"different-bound-context"))
    elif attack == "tag":
        ciphertext = bytearray(base64.b64decode(relay["interaction"]["ciphertextB64"]))
        ciphertext[-1] ^= 1
        relay["interaction"]["ciphertextB64"] = base64.b64encode(ciphertext).decode()
    else:
        public = serialization.load_der_public_key(base64.b64decode(record.payload["publicKeySpkiB64"]))
        key, iv = os.urandom(24 if attack == "key-length" else 32), os.urandom(12)
        plaintext = b"\xff" if attack == "utf8" else (b"x" * 4097 if attack == "oversize" else (b"" if attack == "empty" else b"synthetic"))
        wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=record.aad))
        ciphertext = AESGCM(key).encrypt(iv, plaintext, b"foreign" if attack == "aes-label" else record.aad)
        relay["interaction"].update(wrappedKeyB64=base64.b64encode(wrapped).decode(), ivB64=base64.b64encode(iv).decode(),
                                     ciphertextB64=base64.b64encode(ciphertext).decode())
    with pytest.raises(VaultUnlockError):
        broker.resolve(relay, owner_user_id="owner")
    assert record.take_response() == "" and not record.event.is_set()
    broker.close()


def test_cancel_expiry_duplicates_and_tombstones_are_bounded(monkeypatch):
    from gateway.platforms import thechat_vault as vault
    broker = vault.VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    relay = _relay(record, action="cancel")
    for field in ("wrappedKeyB64", "ivB64", "ciphertextB64"):
        del relay["interaction"][field]
    assert broker.resolve(relay, owner_user_id="owner") is False
    assert record.outcome == "cancelled" and record.take_response() == ""
    assert broker.resolve(relay, owner_user_id="owner") is True
    changed = json.loads(json.dumps(relay))
    changed["interaction"]["id"] = "another-event"
    with pytest.raises(vault.VaultUnlockError):
        broker.resolve(changed, owner_user_id="owner")
    record2 = broker.create(context=_context(), owner_user_id="owner", profile_id="B", session_key="B")
    payload = _relay(record2)
    record2.deadline = time.monotonic() - 1
    with pytest.raises(vault.VaultUnlockError):
        broker.resolve(payload, owner_user_id="owner")
    assert record2.outcome == "expired" and record2._private_key is None
    monkeypatch.setattr(vault, "MAX_REQUESTS", 1)
    broker.release(record)
    broker.release(record2)
    record3 = broker.create(context=_context(), owner_user_id="owner", profile_id="C", session_key="C")
    broker.resolve(_relay(record3), owner_user_id="owner")
    assert len(broker.tombstones) == 1
    broker.close()
    assert not broker.pending and not broker._waiters


@pytest.mark.asyncio
async def test_health_owner_is_canonical_and_missing_owner_denies_access(monkeypatch):
    from gateway.platforms import thechat
    adapter = _adapter()
    client = adapter._client
    adapter._client = None
    monkeypatch.setattr(thechat.httpx, "AsyncClient", lambda **_kw: client)
    assert await adapter.connect_outbound()
    assert adapter._owner_user_id == "owner"
    adapter._capture_vault_owner({"ok": True})
    context = _context()
    from tools.browser_vault_tool import browser_vault_unlock
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            return json.loads(browser_vault_unlock("bitwarden"))
    agent = Agent()
    turn = _turn(adapter, context, agent)
    result = await asyncio.to_thread(turn._run_conversation_with_approval, agent, [], None, None, None)
    assert result["error_type"] == "external_vault_access_denied"
    assert not client.posts
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("end", ["stop", "disconnect", "new-run", "expiry", "cancel", "lock", "release"])
async def test_waiting_prompt_cancels_on_lifecycle_end_without_secret_or_token(end, monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from tools.browser_vault_tool import browser_vault_unlock
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    client = adapter._client
    backend = BitwardenLoginBackend({"binary_path": "/must-not-run"})
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            return json.loads(browser_vault_unlock("bitwarden"))
    agent = Agent()
    turn = _turn(adapter, _context(), agent)
    task = asyncio.create_task(asyncio.to_thread(turn._run_conversation_with_approval, agent, [], None, None, None))
    try:
        await asyncio.wait_for(client.request_ready.wait(), 3)
        record = next(iter(adapter._vault_unlock.pending.values()))
        if end == "stop":
            agent.is_interrupted = True
        elif end == "disconnect":
            await adapter.disconnect()
        elif end == "new-run":
            turn._ctx._run_still_current = lambda: False
        elif end == "expiry":
            record.deadline = time.monotonic() - 1
        elif end == "lock":
            unlock.lock("bitwarden")
        elif end == "release":
            unlock.release_session("hermes-session")
        else:
            relay = _relay(record, action="cancel")
            for key in ("wrappedKeyB64", "ivB64", "ciphertextB64"):
                del relay["interaction"][key]
            response = await adapter._handle_webhook(_signed(adapter, relay))
            assert response.status == 200
        result = await asyncio.wait_for(task, 5)
        assert not result["success"] and not unlock.is_unlocked("bitwarden")
        assert record.take_response() == "" and record._private_key is None
        assert not adapter._vault_unlock.pending and not adapter._vault_unlock._waiters
        if end != "disconnect":
            resolved = [p for _, p in client.posts if p.get("type") == "vault.unlock.resolved"]
            assert len(resolved) == 1
            assert resolved[0]["payload"]["outcome"] == ("expired" if end == "expiry" else "cancelled")
    finally:
        agent.is_interrupted = True
        await asyncio.wait_for(task, 5)
        await adapter.disconnect()
        unlock.lock("bitwarden")


@pytest.mark.asyncio
@pytest.mark.parametrize("sender,requester", [("nonowner", "owner"), ("owner", "nonowner"), ("nonowner", "nonowner")])
async def test_native_registry_denies_nonowner_even_with_preexisting_tokens(sender, requester, monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.bitwarden import BitwardenLoginBackend
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    from tools import browser_vault_tool as tool
    from tools.registry import registry
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    context = _context()
    context["requester_user_id"] = requester
    backends = [BitwardenLoginBackend(), OnePasswordLoginBackend()]
    backends[1]._service_token = "synthetic-service-token"
    unlock.store_session_token("bitwarden", "synthetic-token")
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: backends)
    monkeypatch.setattr("agent.vault_backends.base.enabled_backends", lambda: backends)
    monkeypatch.setattr(tool, "_focus_bound_origin", lambda *_: pytest.fail("nonowner touched the browser"))
    for backend in backends:
        monkeypatch.setattr(backend, "list_items", lambda: pytest.fail("nonowner accessed manager metadata"))
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            listed = json.loads(registry.dispatch("browser_vault_list", {}))
            assert not listed["items"]
            for backend in backends:
                for name, args in (("browser_vault_unlock", {"backend": backend.name}),
                                   ("browser_vault_fill", {"handle": backend.prefix + "item"}),
                                   ("browser_vault_enter_code", {"handle": backend.prefix + "item"})):
                    result = json.loads(registry.dispatch(name, args))
                    assert result["error_type"] == "external_vault_access_denied"
            return {"denied": True}
    agent = Agent()
    turn = _turn(adapter, context, agent, sender=sender)
    try:
        assert await asyncio.to_thread(turn._run_conversation_with_approval, agent, [], None, None, None) == {"denied": True}
        assert not adapter._client.posts
        assert unlock.get_session_token("bitwarden") == "synthetic-token"
    finally:
        await adapter.disconnect()
        unlock.lock("bitwarden")


@pytest.mark.asyncio
async def test_signature_precedes_memory_dispatch_and_errors_do_not_echo_ciphertext(monkeypatch, caplog):
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    record = adapter._vault_unlock.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    relay = _relay(record)
    monkeypatch.setattr("gateway.platforms.thechat.accept_inbound_event", lambda **kw: pytest.fail("unlock reached inbox"))
    request = _signed(adapter, relay)
    request.headers["X-Webhook-Signature"] = "invalid"
    response = await adapter._handle_webhook(request)
    assert response.status == 401 and not record.event.is_set()
    relay["interaction"]["password"] = "synthetic-secret-error"
    response = await adapter._handle_webhook(_signed(adapter, relay))
    assert response.status == 400
    assert "synthetic-secret-error" not in response.text + caplog.text
    assert relay["interaction"]["ciphertextB64"] not in response.text + caplog.text
    await adapter.disconnect()


def test_submit_cancel_race_has_one_winner_and_drops_private_key():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from gateway.platforms.thechat_vault import VaultUnlockBroker, VaultUnlockError
    broker = VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    submit = _relay(record)
    cancel = _relay(record, action="cancel")
    for key in ("wrappedKeyB64", "ivB64", "ciphertextB64"):
        del cancel["interaction"][key]
    barrier = threading.Barrier(2)
    def resolve(payload):
        barrier.wait(timeout=5)
        try:
            return broker.resolve(payload, owner_user_id="owner")
        except VaultUnlockError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(resolve, submit), pool.submit(resolve, cancel)
        results = [a.result(timeout=5), b.result(timeout=5)]
    assert results.count(False) == 1 and results.count("conflict") == 1
    assert record._private_key is None
    assert record.take_response() == ("  synthetic é master\n" if record.outcome == "submitted" else "")
    broker.close()


@pytest.mark.asyncio
async def test_agent_failure_restores_callbacks_and_fences_escaped_prompt():
    from agent.vault_backends import unlock
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    escaped = []
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            escaped.append(unlock.get_unlock_prompt_callback())
            raise RuntimeError("synthetic agent failure")
    agent = Agent()
    turn = _turn(adapter, _context(), agent)
    previous = lambda *_: "previous callback"
    def invoke():
        unlock.set_unlock_prompt_callback(previous)
        unlock.set_current_session_id("previous-session")
        try:
            with pytest.raises(RuntimeError, match="synthetic agent failure"):
                turn._run_conversation_with_approval(agent, [], None, None, None)
            assert unlock.get_unlock_prompt_callback() is previous
            assert unlock.get_current_session_id() == "previous-session"
            assert escaped[0]("bitwarden", "Bitwarden") == ""
        finally:
            unlock.set_unlock_prompt_callback(None)
            unlock.set_current_session_id(None)
    await asyncio.to_thread(invoke)
    assert not adapter._vault_unlock.pending and not adapter._client.posts
    await adapter.disconnect()


@pytest.mark.parametrize("action", [[], {}, None, True])
def test_invalid_union_action_has_typed_safe_validation_error(action):
    from gateway.platforms.thechat_vault import VaultUnlockBroker, VaultUnlockError
    broker = VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    try:
        with pytest.raises(VaultUnlockError):
            broker.resolve(_relay(record, action=action), owner_user_id="owner")
    finally:
        broker.close()


def test_utf8_byte_limit_and_wall_clock_expiry_are_enforced(monkeypatch):
    from gateway.platforms import thechat_vault as vault
    now = time.time()
    monkeypatch.setattr(vault.time, "time", lambda: now)
    broker = vault.VaultUnlockBroker()
    record = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    assert type(record.payload["expiresAt"]) is int
    assert record.payload["expiresAt"] == int(now * 1000) + 120_000
    relay = _relay(record)
    relay["interaction"].update(_encrypt(record, password="é" * 2048))
    assert broker.resolve(relay, owner_user_id="owner") is False
    assert record.take_response() == "é" * 2048
    broker.release(record)
    expired = broker.create(context=_context(), owner_user_id="owner", profile_id="A", session_key="A")
    payload = _relay(expired)
    now += 121
    with pytest.raises(vault.VaultUnlockError):
        broker.resolve(payload, owner_user_id="owner")
    assert expired.outcome == "expired" and expired._private_key is None
    broker.close()
