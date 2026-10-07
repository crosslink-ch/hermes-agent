"""TheChat's encrypted unlock lane. Ciphertext never enters the durable inbox.

Only public request metadata and bounded replay digests outlive the memory waiter.
The native vault backend remains the sole owner of Bitwarden session tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

ALGORITHM = "RSA-OAEP-3072-SHA256+A256GCM"
REQUEST_LIFETIME_S = 120
MAX_REQUESTS = 512
PREVIEW = "Enter your Bitwarden master password in the secure unlock dialog. It never enters the conversation."
_REQUEST_FIELDS = {"version", "requestId", "sessionKey", "profileId", "backend",
                   "requesterUserId", "nonce", "expiresAt", "algorithm"}
_RELAY_FIELDS = _REQUEST_FIELDS | {"id", "requestType", "invocationId", "conversationId", "threadId", "actorUserId", "action"}
_CIPHER_FIELDS = {"wrappedKeyB64", "ivB64", "ciphertextB64"}


class VaultUnlockError(ValueError):
    """Fixed safe errors only: never reflect an untrusted field or crypto exception."""
    def __init__(self, *, status=409):
        super().__init__("Invalid or stale vault unlock interaction")
        self.status = status


def _token(value: Any, limit=1000):
    if (not isinstance(value, str) or not value or len(value) > limit or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise VaultUnlockError(status=400)
    return value


def _uuid(value):
    _token(value, 36)
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise VaultUnlockError(status=400) from None


def _b64(value, minimum, maximum):
    if not isinstance(value, str) or not value or len(value) > 4 * ((maximum + 2) // 3):
        raise VaultUnlockError(status=400)
    try:
        decoded = base64.b64decode(value, validate=True)
        if not minimum <= len(decoded) <= maximum or base64.b64encode(decoded).decode() != value:
            raise ValueError
    except ValueError:
        raise VaultUnlockError(status=400) from None
    return decoded


@dataclass(repr=False)
class PendingUnlock:
    payload: dict
    context: dict
    aad: bytes
    deadline: float
    _private_key: Any
    event: threading.Event = field(default_factory=threading.Event)
    outcome: str | None = None
    _response: str = ""
    _lock: Any = field(default_factory=threading.Lock)

    def take_response(self) -> str:
        with self._lock:
            response, self._response = self._response, ""
            return response

    def settle(self, outcome, response="") -> bool:
        with self._lock:
            if self.outcome is not None:
                return False
            self.outcome, self._response, self._private_key = outcome, response, None
            self.event.set()
            return True


class VaultUnlockBroker:
    def __init__(self):
        self.pending: dict[str, PendingUnlock] = {}
        self._waiters: dict[str, PendingUnlock] = {}
        self.tombstones: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._lock = threading.RLock()
        self.closed = False

    def create(self, *, context, profile_id, session_key) -> PendingUnlock:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        requester = _token(context.get("requester_user_id"), 255)
        _token(profile_id, 255)
        _token(session_key)
        for key in ("invocation_id", "conversation_id"):
            _uuid(context.get(key))
        _token(context.get("bot_id"), 255)
        if context.get("thread_id") is not None:
            _token(context["thread_id"], 512)
        private = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        payload = {"version": 2, "requestId": str(uuid.uuid4()), "sessionKey": session_key,
                   "profileId": profile_id, "backend": "bitwarden",
                   "requesterUserId": requester, "nonce": secrets.token_urlsafe(32),
                   "expiresAt": int(time.time() * 1000) + REQUEST_LIFETIME_S * 1000, "algorithm": ALGORITHM,
                   "publicKeySpkiB64": base64.b64encode(private.public_key().public_bytes(
                       serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)).decode()}
        aad = json.dumps([2, context["bot_id"], requester, profile_id, session_key,
                          context["invocation_id"], context["conversation_id"], context.get("thread_id"),
                          payload["requestId"], "bitwarden", payload["nonce"], payload["expiresAt"]],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        record = PendingUnlock(payload, dict(context), aad, time.monotonic() + REQUEST_LIFETIME_S, private)
        with self._lock:
            self._prune()
            if self.closed or len(self._waiters) >= MAX_REQUESTS:
                raise VaultUnlockError()
            self.pending[payload["requestId"]] = record
            self._waiters[payload["requestId"]] = record
        return record

    def _prune(self):
        now = time.monotonic()
        for record in list(self.pending.values()):
            if record.deadline <= now or record.payload["expiresAt"] <= int(time.time() * 1000):
                self.cancel(record, outcome="expired")
        for rid, (_, expiry) in list(self.tombstones.items()):
            if expiry <= now:
                del self.tombstones[rid]

    def resolve(self, payload) -> bool:
        if (not isinstance(payload, dict) or set(payload) != {"type", "interaction"}
                or payload["type"] != "thechat.hermes_platform.vault_unlock"):
            raise VaultUnlockError(status=400)
        item = payload["interaction"]
        if not isinstance(item, dict):
            raise VaultUnlockError(status=400)
        action = item.get("action")
        if (not isinstance(action, str) or action not in {"submit", "cancel"}
                or set(item) != (_RELAY_FIELDS | (_CIPHER_FIELDS if action == "submit" else set()))):
            raise VaultUnlockError(status=400)
        if type(item["version"]) is not int or item["version"] != 2 or type(item["expiresAt"]) is not int:
            raise VaultUnlockError(status=400)
        for key in ("id", "requestId", "sessionKey", "profileId", "requesterUserId", "actorUserId", "nonce"):
            _token(item[key])
        for key in ("requestId", "invocationId", "conversationId"):
            _uuid(item[key])
        if item["threadId"] is not None:
            _token(item["threadId"], 512)
        if (item["backend"] != "bitwarden" or item["algorithm"] != ALGORITHM
                or item["requestType"] != "vault.unlock.request"):
            raise VaultUnlockError(status=400)
        if item["actorUserId"] != item["requesterUserId"]:
            raise VaultUnlockError()
        cipher = tuple(_b64(item[k], lo, hi) for k, lo, hi in (
            ("wrappedKeyB64", 384, 384), ("ivB64", 12, 12), ("ciphertextB64", 17, 4112))) if action == "submit" else None
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).digest()
        with self._lock:
            self._prune()
            rid = item["requestId"]
            previous = self.tombstones.get(rid)
            if previous is not None:
                if secrets.compare_digest(previous[0], digest):
                    return True
                raise VaultUnlockError()
            record = self.pending.get(rid)
            if record is None or self.closed:
                raise VaultUnlockError()
            if any(item[k] != record.payload[k] for k in _REQUEST_FIELDS):
                raise VaultUnlockError()
            if any(item[k] != record.context[v] for k, v in (
                    ("invocationId", "invocation_id"), ("conversationId", "conversation_id"), ("threadId", "thread_id"))):
                raise VaultUnlockError()
            response = ""
            if cipher is not None:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                try:
                    wrapped, iv, ciphertext = cipher
                    key = record._private_key.decrypt(wrapped, padding.OAEP(
                        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=record.aad))
                    if len(key) != 32:
                        raise ValueError
                    plaintext = AESGCM(key).decrypt(iv, ciphertext, record.aad)
                    if not 1 <= len(plaintext) <= 4096:
                        raise ValueError
                    response = plaintext.decode("utf-8", errors="strict")
                except Exception:
                    raise VaultUnlockError() from None
                finally:
                    key = plaintext = None
            if record.deadline <= time.monotonic() or record.payload["expiresAt"] <= int(time.time() * 1000):
                self.cancel(record, outcome="expired")
                raise VaultUnlockError()
            record.settle("submitted" if action == "submit" else "cancelled", response)
            self.pending.pop(rid, None)
            self.tombstones[rid] = (digest, time.monotonic() + REQUEST_LIFETIME_S)
            while len(self.tombstones) > MAX_REQUESTS:
                self.tombstones.popitem(last=False)
            return False

    def cancel(self, record, *, outcome="cancelled"):
        with self._lock:
            won = record.settle(outcome)
            self.pending.pop(record.payload["requestId"], None)
            # Even a submitted-but-not-consumed response must not survive a disconnect/finalization.
            record.take_response()
            return won

    def release(self, record):
        with self._lock:
            self.cancel(record)
            self._waiters.pop(record.payload["requestId"], None)

    def cancel_invocation(self, invocation_id):
        with self._lock:
            for record in list(self._waiters.values()):
                if record.context["invocation_id"] == invocation_id:
                    self.cancel(record)

    def close(self):
        with self._lock:
            self.closed = True
            for record in list(self._waiters.values()):
                self.release(record)
