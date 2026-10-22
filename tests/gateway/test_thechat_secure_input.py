"""v3 wire/crypto contract; synthetic credentials only, no external service."""
import base64
import json
import os
import uuid

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def context():
    return dict(bot_id="bot-id", requester_user_id="owner", invocation_id=str(uuid.uuid4()),
                conversation_id=str(uuid.uuid4()), thread_id=None)


def encrypt(record, values, *, aad=None, raw=None):
    """Independent client implementation usable by the paired E2E harness."""
    p, c = record.payload, record.context
    label = json.dumps([3, c["bot_id"], p["requesterUserId"], p["profileId"], p["sessionKey"],
                        c["invocation_id"], c["conversation_id"], c["thread_id"], p["requestId"],
                        p["promptKind"], p["fields"], p["target"], p["nonce"], p["expiresAt"]],
                       ensure_ascii=False, separators=(",", ":")).encode() if aad is None else aad
    public = serialization.load_der_public_key(base64.b64decode(p["publicKeySpkiB64"]))
    key, iv = os.urandom(32), os.urandom(12)
    wrapped = public.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=label))
    plaintext = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode() if raw is None else raw
    return dict(zip(("wrappedKeyB64", "ivB64", "ciphertextB64"),
                    [base64.b64encode(v).decode() for v in (wrapped, iv, AESGCM(key).encrypt(iv, plaintext, label))]))


def relay(record, values=None, *, action="submit", **overrides):
    p, c = record.payload, record.context
    item = {k: v for k, v in p.items() if k != "publicKeySpkiB64"}
    item.update(id="progress-event", requestType="secure.input.request", invocationId=c["invocation_id"],
                conversationId=c["conversation_id"], threadId=c["thread_id"], actorUserId=c["requester_user_id"], action=action)
    if action == "submit":
        item.update(encrypt(record, values))
    item.update(overrides)
    return {"type": "thechat.hermes_platform.secure_input", "interaction": item}


def make_record(kind="vault.code", target="https://example.test", **kwargs):
    from gateway.platforms.thechat_secure_input import SecureInputBroker
    broker = SecureInputBroker()
    return broker, broker.create(context=context(), profile_id="opaque-profile", session_key="session-key",
                                  prompt_kind=kind, target=target, authorized=lambda: True, **kwargs)


@pytest.mark.parametrize("kind,target,values", [
    ("onepassword.unlock", "1Password", {"password": "  synthetic é master\n"}),
    ("vault.save_login", "https://example.test", {"identifier": " user@example.test ", "password": "  é\n"}),
    ("vault.code", "example.test", {"code": " aB-cD 42 "}),
    ("sudo.password", "Privileged terminal command", {"password": " synthetic é "}),
    ("skill.secret", "SYNTHETIC_TEST_KEY", {"value": " é key\n"}),
])
def test_exact_masked_values_and_replay_fence(kind, target, values):
    broker, record = make_record(kind, target)
    envelope = relay(record, values)
    assert broker.resolve(envelope) is False
    assert record.take_response() == values
    assert not record.take_response()
    assert record.outcome == "submitted" and record._private_key is None
    assert broker.resolve(envelope) is True
    assert not broker.pending
    broker.release(record)
    assert not broker._waiters
    assert all(value not in repr(record) + repr(broker.tombstones) for value in values.values())


@pytest.mark.asyncio
async def test_signed_lane_negotiation_and_no_durable_inbox(monkeypatch):
    from tests.gateway.test_thechat_vault_unlock import _adapter, _signed
    adapter = _adapter()
    monkeypatch.setattr("gateway.platforms.thechat.accept_inbound_event", lambda **kw: pytest.fail("secret in inbox"))
    broker, record = make_record()
    adapter._secure_input = broker
    envelope = relay(record, {"code": "synthetic-recovery-code"})
    adapter._capture_secure_input_capability({"capabilities": {"secureInput": 3}})
    result = await adapter._handle_webhook(_signed(adapter, envelope))
    assert result.status == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert record.take_response() == {"code": "synthetic-recovery-code"}
    for capability in (None, True, "3", 2, 4):
        adapter._capture_secure_input_capability({"capabilities": {"secureInput": capability}})
        broker, record = make_record()
        adapter._secure_input = broker
        result = await adapter._handle_webhook(_signed(adapter, relay(record, {"code": "secret"})))
        assert result.status == 409
        assert not record.event.is_set()
    invalid = _signed(adapter, envelope)
    invalid.headers["X-Webhook-Signature"] = "invalid"
    result = await adapter._handle_webhook(invalid)
    assert result.status == 401 and result.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("raw", [b'{"code":"first","code":"second"}', b'{"code":"x","password":"x"}',
                                  b'{}', b'[]', b'{"code":1}', b'{"code":null}', b'{"code":""}',
                                  b'{"code":"\\ud800"}', b'not json'])
def test_decrypted_object_rejects_ambiguous_or_malformed_values(raw):
    from gateway.platforms.thechat_secure_input import SecureInputError
    broker, record = make_record()
    envelope = relay(record, {"code": "unused"}, **encrypt(record, None, raw=raw))
    with pytest.raises(SecureInputError, match="Invalid or stale secure input interaction"):
        broker.resolve(envelope)
    assert not record.event.is_set() and not record.take_response()
    broker.close()
    assert record._private_key is None and not broker._waiters


@pytest.mark.parametrize("override", [{"actorUserId": "other"}, {"profileId": "B"}, {"sessionKey": "B"},
                                      {"target": "other.test"}, {"fields": ["password"]}, {"version": True},
                                      {"threadId": "other"}, {"nonce": "A" * 43}, {"extra": "secret"},
                                      {"ivB64": "bad"}])
def test_context_and_wire_mismatch_never_consumes_waiter(override):
    from gateway.platforms.thechat_secure_input import SecureInputError
    broker, record = make_record()
    with pytest.raises(SecureInputError):
        broker.resolve(relay(record, {"code": "synthetic"}, **override))
    assert not record.event.is_set()
    assert broker.resolve(relay(record, {"code": "retry"})) is False
    assert record.take_response() == {"code": "retry"}


def test_cancellation_expiry_revocation_and_shutdown_clear_handoffs():
    from gateway.platforms.thechat_secure_input import SecureInputError
    for lifecycle in ("cancel", "expire", "revoke", "disconnect", "finalize"):
        broker, record = make_record()
        envelope = relay(record, {"code": "synthetic"})
        if lifecycle == "cancel":
            broker.resolve(relay(record, action="cancel"))
        elif lifecycle == "expire":
            record.deadline = 0
        elif lifecycle == "revoke":
            record.authorized = lambda: False
        elif lifecycle == "disconnect":
            broker.close()
        else:
            broker.resolve(envelope)
            broker.cancel_invocation(record.context["invocation_id"])
            assert not record.take_response()
            continue
        with pytest.raises(SecureInputError):
            broker.resolve(envelope)
        assert record.event.is_set() and record._private_key is None
        assert not record.take_response()


@pytest.mark.parametrize("values,accepted", [({"code": "é" * 2048}, True), ({"code": "é" * 2049}, False),
                                             ({"identifier": "x" * 1024, "password": "p"}, True),
                                             ({"identifier": "x" * 1025, "password": "p"}, False)])
def test_utf8_byte_limits(values, accepted):
    from gateway.platforms.thechat_secure_input import SecureInputError
    broker, record = make_record("vault.save_login" if "identifier" in values else "vault.code")
    envelope = relay(record, values)
    if accepted:
        broker.resolve(envelope)
        assert record.take_response() == values
    else:
        with pytest.raises(SecureInputError):
            broker.resolve(envelope)
        assert not record.take_response()


def test_wrong_aad_and_concurrent_conflicting_submissions():
    from concurrent.futures import ThreadPoolExecutor
    from gateway.platforms.thechat_secure_input import SecureInputError
    broker, record = make_record()
    with pytest.raises(SecureInputError):
        broker.resolve(relay(record, {"code": "secret"}, **encrypt(record, {"code": "secret"}, aad=b"wrong context")))
    envelopes = [relay(record, {"code": value}) for value in ("first", "second")]
    def submit(envelope):
        try:
            return broker.resolve(envelope)
        except SecureInputError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, envelopes))
    assert results.count(False) == 1 and results.count("rejected") == 1
    assert record.take_response()["code"] == ("first" if results[0] is False else "second")
    assert not record.take_response()


@pytest.mark.asyncio
async def test_real_loopback_health_and_signed_http_lane(monkeypatch):
    import httpx
    from aiohttp import web
    from aiohttp.test_utils import TestServer
    from gateway.platforms.thechat import TheChatAdapter
    from gateway.config import PlatformConfig
    from tests.gateway.test_thechat_vault_unlock import _signed
    app = web.Application()
    async def health(_):
        return web.json_response({"ok": True, "ownerUserId": "owner", "capabilities": {"secureInput": 3}})
    app.router.add_get("/hermes-platform/health", health)
    async with TestServer(app) as api:
        adapter = TheChatAdapter(PlatformConfig(enabled=True, token="synthetic-token", extra={
            "base_url": str(api.make_url("")), "webhook_url": "http://127.0.0.1/thechat/webhook"}))
        assert await adapter.connect_outbound()
        assert adapter._secure_input_supported and adapter._owner_user_id == "owner"
        adapter._webhook_secret = "synthetic-signing-key"
        receiver = web.Application()
        receiver.router.add_post("/thechat/webhook", adapter._handle_webhook)
        monkeypatch.setattr("gateway.platforms.thechat.accept_inbound_event", lambda **_: pytest.fail("persisted ciphertext"))
        try:
            async with TestServer(receiver) as ingress, httpx.AsyncClient() as http:
                record = adapter._secure_input.create(context=context(), profile_id="A", session_key="session",
                    prompt_kind="vault.code", target="example.test", authorized=lambda: True)
                envelope = relay(record, {"code": "synthetic-code"})
                request = _signed(adapter, envelope)
                for expected in (False, True):
                    result = await http.post(str(ingress.make_url("/thechat/webhook")),
                                             content=await request.read(), headers=request.headers)
                    assert result.status_code == 200 and result.json() == {"ok": True, "duplicate": expected}
                    assert result.headers["cache-control"] == "no-store"
                assert record.take_response() == {"code": "synthetic-code"}
                bad = await http.post(str(ingress.make_url("/thechat/webhook")), content=b'{}')
                assert bad.status_code == 401 and bad.headers["cache-control"] == "no-store"
        finally:
            await adapter.disconnect()
        assert adapter._secure_input.closed and not adapter._secure_input_supported

