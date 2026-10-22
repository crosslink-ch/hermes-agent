"""v3 masked inputs: authenticated metadata, one-shot memory handoff, no inbox.

The v2 broker supplies the bounded cancellation/release lifecycle, unchanged.
Only v3 request creation and strict encrypted-object decoding differ.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from gateway.platforms.thechat_vault import (
    ALGORITHM, MAX_REQUESTS, VaultUnlockBroker, VaultUnlockError, _b64, _token, _uuid,
)

REQUEST_LIFETIME_S = 120
PROMPT_FIELDS = {
    "onepassword.unlock": ("password",),
    "vault.save_login": ("identifier", "password"),
    "vault.code": ("code",),
    "sudo.password": ("password",),
    "skill.secret": ("value",),
}
_REQUEST_FIELDS = {"version", "requestId", "sessionKey", "profileId", "requesterUserId",
                   "promptKind", "fields", "target", "nonce", "expiresAt", "algorithm"}
_RELAY_FIELDS = _REQUEST_FIELDS | {"id", "requestType", "invocationId", "conversationId", "threadId", "actorUserId", "action"}
_CIPHER_FIELDS = {"wrappedKeyB64", "ivB64", "ciphertextB64"}


class SecureInputError(ValueError):
    def __init__(self, *, status=409):
        super().__init__("Invalid or stale secure input interaction")
        self.status = status


def strict_object(pairs):
    """Used for both signed JSON and decrypted JSON; reject ambiguous duplicates."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise SecureInputError(status=400)
        result[key] = value
    return result


def _validate_target(kind, target):
    _token(target, 2000)
    if kind == "onepassword.unlock" and target != "1Password":
        raise SecureInputError(status=400)
    if kind == "skill.secret" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target):
        raise SecureInputError(status=400)


@dataclass(repr=False)
class PendingInput:
    payload: dict
    context: dict
    aad: bytes
    deadline: float
    _private_key: Any
    authorized: Callable[[], bool]
    event: threading.Event = field(default_factory=threading.Event)
    outcome: str | None = None
    _response: dict | None = None
    _lock: Any = field(default_factory=threading.Lock)

    def take_response(self):
        with self._lock:
            response, self._response = self._response, None
            return response

    def settle(self, outcome, response=None):
        with self._lock:
            if self.outcome is not None:
                return False
            self.outcome, self._response, self._private_key = outcome, response, None
            self.event.set()
            return True


class SecureInputBroker(VaultUnlockBroker):
    def create(self, *, context, profile_id, session_key, prompt_kind, target, authorized):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        try:
            if not isinstance(prompt_kind, str) or prompt_kind not in PROMPT_FIELDS:
                raise SecureInputError(status=400)
            requester = _token(context.get("requester_user_id"), 255)
            _token(profile_id, 255)
            _token(session_key)
            for name in ("invocation_id", "conversation_id"):
                _uuid(context.get(name))
            _token(context.get("bot_id"), 255)
            if context.get("thread_id") is not None:
                _token(context["thread_id"], 512)
            _validate_target(prompt_kind, target)
        except VaultUnlockError as exc:
            raise SecureInputError(status=exc.status) from None
        if not authorized():
            raise SecureInputError()
        private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        payload = dict(version=3, requestId=str(uuid.uuid4()), sessionKey=session_key,
                       profileId=profile_id, requesterUserId=requester, promptKind=prompt_kind,
                       fields=list(PROMPT_FIELDS[prompt_kind]), target=target, nonce=secrets.token_urlsafe(32),
                       expiresAt=int(time.time() * 1000) + REQUEST_LIFETIME_S * 1000, algorithm=ALGORITHM,
                       publicKeySpkiB64=base64.b64encode(private.public_key().public_bytes(
                           serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)).decode())
        aad = json.dumps([3, context["bot_id"], requester, profile_id, session_key, context["invocation_id"],
                          context["conversation_id"], context.get("thread_id"), payload["requestId"], prompt_kind,
                          payload["fields"], target, payload["nonce"], payload["expiresAt"]],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        record = PendingInput(payload, dict(context), aad, time.monotonic() + REQUEST_LIFETIME_S, private, authorized)
        with self._lock:
            self._prune()
            if self.closed or len(self._waiters) >= MAX_REQUESTS or not authorized():
                raise SecureInputError()
            self.pending[payload["requestId"]] = record
            self._waiters[payload["requestId"]] = record
        return record

    def resolve(self, payload):
        try:
            return self._resolve(payload)
        except VaultUnlockError as exc:
            raise SecureInputError(status=exc.status) from None

    def _resolve(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {"type", "interaction"}
                or payload["type"] != "thechat.hermes_platform.secure_input"):
            raise SecureInputError(status=400)
        item = payload["interaction"]
        if not isinstance(item, dict):
            raise SecureInputError(status=400)
        action = item.get("action")
        if (not isinstance(action, str) or action not in {"submit", "cancel"}
                or set(item) != _RELAY_FIELDS | (_CIPHER_FIELDS if action == "submit" else set())):
            raise SecureInputError(status=400)
        if type(item["version"]) is not int or item["version"] != 3 or type(item["expiresAt"]) is not int:
            raise SecureInputError(status=400)
        for name, limit in (("id", 1000), ("sessionKey", 1000), ("profileId", 255),
                            ("requesterUserId", 255), ("actorUserId", 255)):
            _token(item[name], limit)
        for name in ("requestId", "invocationId", "conversationId"):
            _uuid(item[name])
        if item["threadId"] is not None:
            _token(item["threadId"], 512)
        kind = item["promptKind"]
        if (not isinstance(kind, str) or kind not in PROMPT_FIELDS or item["fields"] != list(PROMPT_FIELDS[kind])
                or item["algorithm"] != ALGORITHM or item["requestType"] != "secure.input.request"
                or not isinstance(item["nonce"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", item["nonce"])):
            raise SecureInputError(status=400)
        _validate_target(kind, item["target"])
        if item["actorUserId"] != item["requesterUserId"]:
            raise SecureInputError()
        cipher = tuple(_b64(item[k], lo, hi) for k, lo, hi in (
            ("wrappedKeyB64", 384, 384), ("ivB64", 12, 12), ("ciphertextB64", 17, 16400))) if action == "submit" else None
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).digest()
        with self._lock:
            self._prune()
            rid = item["requestId"]
            previous = self.tombstones.get(rid)
            if previous is not None:
                if not self.closed and secrets.compare_digest(previous[0], digest):
                    return True
                raise SecureInputError()
            record = self.pending.get(rid)
            if record is None or self.closed:
                raise SecureInputError()
            if any(item[k] != record.payload[k] for k in _REQUEST_FIELDS):
                raise SecureInputError()
            if any(item[k] != record.context.get(v) for k, v in (
                    ("invocationId", "invocation_id"), ("conversationId", "conversation_id"), ("threadId", "thread_id"))):
                raise SecureInputError()
            if not record.authorized():
                self.cancel(record)
                raise SecureInputError()
            response = self._decrypt(record, cipher) if cipher else None
            if record.deadline <= time.monotonic() or record.payload["expiresAt"] <= int(time.time() * 1000):
                self.cancel(record, outcome="expired")
                raise SecureInputError()
            if not record.authorized():
                self.cancel(record)
                raise SecureInputError()
            record.settle("submitted" if action == "submit" else "cancelled", response)
            self.pending.pop(rid, None)
            self.tombstones[rid] = (digest, time.monotonic() + REQUEST_LIFETIME_S)
            while len(self.tombstones) > MAX_REQUESTS:
                self.tombstones.popitem(last=False)
            return False

    @staticmethod
    def _decrypt(record, cipher):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        key = plaintext = None
        try:
            wrapped, iv, ciphertext = cipher
            key = record._private_key.decrypt(wrapped, padding.OAEP(
                mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=record.aad))
            if len(key) != 32:
                raise ValueError
            plaintext = AESGCM(key).decrypt(iv, ciphertext, record.aad)
            if not 1 <= len(plaintext) <= 16384:
                raise ValueError
            values = json.loads(plaintext.decode("utf-8", errors="strict"), object_pairs_hook=strict_object)
            if not isinstance(values, dict) or set(values) != set(record.payload["fields"]):
                raise ValueError
            for name, value in values.items():
                if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= (1024 if name == "identifier" else 4096):
                    raise ValueError
            return values
        except Exception:
            raise SecureInputError() from None
        finally:
            key = plaintext = None
