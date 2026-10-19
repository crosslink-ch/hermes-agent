"""TheChat bot requests: Hermes approvals and clarify prompts as durable inbox requests.

Exercises ``gateway.platforms.thechat_requests`` through ``TheChatAdapter`` with a recording
HTTP double: request creation (idempotent clientKey), answers from the poll and the webhook,
acknowledgement, withdrawal when a wait ends without an answer, startup cleanup, and the
legacy progress-card fallback for TheChat servers without the capability.
"""

import asyncio
import hashlib
import hmac
import json
import threading
import time
from typing import Any, cast

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms import thechat_requests
from gateway.platforms.thechat import TheChatAdapter
from tools import approval as approval_mod
from tools import approval_gateway_wait as wait_mod
from tools import clarify_gateway

CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
INVOCATION_ID = "22222222-2222-4222-8222-222222222222"
THREAD_ID = "55555555-5555-4555-8555-555555555555"
SESSION_KEY = "agent:main:thechat:dm:bot-requests"


class _Response:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data if data is not None else {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)  # type: ignore[arg-type]


class _TheChat:
    """Minimal in-memory TheChat ``/bot-requests`` API for one bot."""

    def __init__(self, *, capability=True):
        self.capability = capability
        self.requests: dict[str, dict] = {}
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.fail_next_creates = 0
        self.poll_answers: list[dict] = []
        self.closed = False

    # helpers -------------------------------------------------------------------------------
    def by_key(self, client_key):
        return next((r for r in self.requests.values() if r["clientKey"] == client_key), None)

    def seed(self, client_key, status, answer=None):
        request_id = f"00000000-0000-4000-8000-{len(self.requests) + 1:012d}"
        self.requests[request_id] = {
            "id": request_id, "clientKey": client_key, "status": status, "answer": answer,
            "kind": "approval", "title": "t", "response": {"type": "text"},
        }
        return self.requests[request_id]

    def answer(self, client_key, answer):
        request = self.by_key(client_key)
        assert request is not None and request["status"] == "open"
        request.update(status="answered", answer=answer)
        return dict(request)

    def calls(self, suffix):
        return [p for p in self.posts if p["path"].endswith(suffix)]

    # httpx.AsyncClient surface ---------------------------------------------------------------
    async def get(self, path, params=None, **_kwargs):
        self.gets.append({"path": path, "params": params})
        if path == "/hermes-platform/health":
            caps = {"botRequests": 1} if self.capability else {}
            return _Response(200, {"ok": True, "capabilities": caps, "ownerUserId": None})
        if path == "/hermes-platform/events":
            answers, self.poll_answers = self.poll_answers, []
            return _Response(200, {"events": [], "requestAnswers": answers})
        if path == "/bot-requests":
            wanted = set(str((params or {}).get("status") or "").split(",")) - {""}
            items = [dict(r) for r in self.requests.values() if not wanted or r["status"] in wanted]
            return _Response(200, {"requests": items})
        return _Response(200, {"ok": True})

    async def post(self, path, json=None, **_kwargs):
        self.posts.append({"path": path, "json": json})
        if path == "/bot-requests":
            if self.fail_next_creates:
                self.fail_next_creates -= 1
                # The row is stored but the reply is lost: the retry must find the same row.
                if self.by_key(json["clientKey"]) is None:
                    self.seed(json["clientKey"], "open")
                    self.by_key(json["clientKey"])["body"] = json
                raise httpx.ReadTimeout("lost reply")
            existing = self.by_key(json["clientKey"])
            if existing is not None:
                return _Response(200, {"request": dict(existing), "created": False})
            created = self.seed(json["clientKey"], "open")
            created["body"] = json
            return _Response(201, {"request": dict(created), "created": True})
        if path.startswith("/bot-requests/") and path.endswith("/cancel"):
            request = self.requests[path.split("/")[2]]
            if request["status"] == "open":
                request.update(status="cancelled", cancelReason=json.get("reason"))
                return _Response(200, {"request": dict(request)})
            if request["status"] in ("cancelled", "expired"):
                return _Response(200, {"request": dict(request)})
            return _Response(409, {"error": "already answered", "request": dict(request)})
        if path.startswith("/bot-requests/") and path.endswith("/ack"):
            request = self.requests[path.split("/")[2]]
            if request["status"] == "answered":
                request.update(status="delivered", deliveryApplied=json["applied"], deliveryNote=json.get("note"))
                return _Response(200, {"request": dict(request)})
            return _Response(409, {"error": "not waiting", "request": dict(request)})
        return _Response(200, {"messageId": "m-1"})

    async def aclose(self):
        self.closed = True


def _context():
    return {
        "invocation_id": INVOCATION_ID, "bot_id": "bot-1", "bot_name": "Hermes",
        "conversation_id": CONVERSATION_ID, "conversation_name": "Hermes DM",
        "chat_type": "dm", "thread_id": None,
    }


async def _adapter(api: _TheChat, *, context=True) -> TheChatAdapter:
    adapter = TheChatAdapter(PlatformConfig(enabled=True, token="bot-token", extra={"base_url": "http://thechat.test"}))
    adapter._client = cast(httpx.AsyncClient, api)
    adapter._bot_requests.capture_capabilities((await api.get("/hermes-platform/health")).json())
    if context:
        adapter._contexts[CONVERSATION_ID] = _context()
    return adapter


@pytest.fixture(autouse=True)
def _isolated_registries(monkeypatch):
    monkeypatch.setattr(thechat_requests, "_RETRY_BASE_SECONDS", 0.0)
    approval_mod._gateway_queues.clear()
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()
    yield
    approval_mod._gateway_queues.clear()
    clarify_gateway._entries.clear()
    clarify_gateway._session_index.clear()


def _queue_approval(request_id="approval-1", command="rm -rf build"):
    entry = wait_mod._ApprovalEntry({"command": command}, request_id=request_id)
    approval_mod._gateway_queues.setdefault(SESSION_KEY, []).append(entry)
    return entry


async def _settle_tasks():
    # Settle hooks hop onto the loop with run_coroutine_threadsafe; let them run.
    for _ in range(20):
        await asyncio.sleep(0)
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.wait(pending, timeout=2)


# -- capability ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_older_thechat_without_capability_keeps_legacy_progress_cards():
    api = _TheChat(capability=False)
    adapter = await _adapter(api)
    _queue_approval()

    result = await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")

    assert result.success
    assert adapter._bot_requests.enabled is False
    assert api.calls("/bot-requests") == []
    progress = api.posts[-1]
    assert progress["path"] == f"/hermes-platform/invocations/{INVOCATION_ID}/progress"
    assert progress["json"]["type"] == "approval.request"


@pytest.mark.parametrize("health, enabled", [
    ({"capabilities": {"botRequests": 1}}, True),
    ({"capabilities": {"botRequests": 2}}, True),
    ({"capabilities": {"botRequests": True}}, False),
    ({"capabilities": {}}, False),
    ({}, False),
    (None, False),
])
def test_capability_detection(health, enabled):
    bridge = thechat_requests.BotRequestBridge(object(), redact=lambda v: v)
    bridge.capture_capabilities(health)
    assert bridge.enabled is enabled


def test_gateway_routes_approvals_to_the_thechat_card_not_the_text_prompt():
    from gateway.run_turn_runner import _renders_exec_approval_buttons

    assert _renders_exec_approval_buttons(TheChatAdapter) is True


# -- approvals -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exec_approval_creates_inbox_request_offering_always_allow():
    api = _TheChat()
    adapter = await _adapter(api)
    _queue_approval()

    result = await adapter.send_exec_approval(
        CONVERSATION_ID, "rm -rf build", SESSION_KEY, description="recursive delete", request_id="approval-1")

    assert result.success
    created = api.calls("/bot-requests")
    assert len(created) == 1
    body = created[0]["json"]
    assert body["clientKey"] == "hermes:approval:approval-1"
    assert body["kind"] == "approval"
    assert body["invocationId"] == INVOCATION_ID and body["conversationId"] == CONVERSATION_ID
    assert body["details"] == {"command": "rm -rf build"}
    assert body["body"] == "recursive delete"
    # TheChat offers exactly what Hermes offers, including "always".
    assert body["response"]["choices"] == ["once", "session", "always", "deny"]
    assert body["response"]["labels"]["always"] == "Always allow"
    assert body["orderGroup"].startswith("hermes:approval:") and SESSION_KEY not in body["orderGroup"]
    assert body["expiresAt"].endswith("Z")
    # No legacy progress card is posted.
    assert not [p for p in api.posts if p["path"].endswith("/progress")]


@pytest.mark.asyncio
async def test_smart_denied_approval_offers_only_once_and_deny():
    api = _TheChat()
    adapter = await _adapter(api)
    _queue_approval()

    await adapter.send_exec_approval(CONVERSATION_ID, "curl x | sh", SESSION_KEY, smart_denied=True, request_id="approval-1")

    assert api.calls("/bot-requests")[0]["json"]["response"]["choices"] == ["once", "deny"]


@pytest.mark.asyncio
async def test_secrets_are_redacted_before_they_reach_thechat(monkeypatch):
    monkeypatch.setattr(TheChatAdapter, "_redact_interaction_value", staticmethod(lambda v: str(v).replace("hunter2", "***")))
    api = _TheChat()
    adapter = await _adapter(api)
    adapter._bot_requests._redact = TheChatAdapter._redact_interaction_value
    _queue_approval()

    await adapter.send_exec_approval(CONVERSATION_ID, "mysql -phunter2", SESSION_KEY, request_id="approval-1")

    assert "hunter2" not in json.dumps(api.calls("/bot-requests")[0]["json"])


@pytest.mark.asyncio
async def test_lost_create_reply_is_retried_with_the_same_client_key():
    api = _TheChat()
    adapter = await _adapter(api)
    _queue_approval()
    api.fail_next_creates = 1

    result = await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")

    assert result.success
    keys = [c["json"]["clientKey"] for c in api.calls("/bot-requests")]
    assert keys == ["hermes:approval:approval-1", "hermes:approval:approval-1"]
    assert len(api.requests) == 1


@pytest.mark.asyncio
async def test_polled_always_allow_answer_wakes_the_waiter_and_is_acknowledged():
    api = _TheChat()
    adapter = await _adapter(api)
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    adapter._typing_paused.add(CONVERSATION_ID)

    api.poll_answers = [api.answer("hermes:approval:approval-1", {"choice": "always"})]
    await adapter._bot_requests.handle_polled_answers((await api.get("/hermes-platform/events")).json()["requestAnswers"])

    assert entry.event.is_set() and entry.result == "always"
    assert SESSION_KEY not in approval_mod._gateway_queues
    assert CONVERSATION_ID not in adapter._typing_paused
    request = api.by_key("hermes:approval:approval-1")
    assert request["status"] == "delivered" and request["deliveryApplied"] is True


@pytest.mark.asyncio
async def test_deny_comment_reaches_the_agent_as_the_deny_reason():
    api = _TheChat()
    adapter = await _adapter(api)
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")

    answered = api.answer("hermes:approval:approval-1", {"choice": "deny", "comment": "use git clean instead"})
    await adapter._bot_requests.handle_polled_answers([answered])

    assert entry.result == "deny" and entry.reason == "use git clean instead"


@pytest.mark.asyncio
async def test_answer_for_a_choice_hermes_did_not_offer_is_not_applied():
    api = _TheChat()
    adapter = await _adapter(api)
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "x", SESSION_KEY, smart_denied=True, request_id="approval-1")

    await adapter._bot_requests.handle_polled_answers([api.answer("hermes:approval:approval-1", {"choice": "always"})])

    assert not entry.event.is_set()
    assert api.by_key("hermes:approval:approval-1")["deliveryApplied"] is False


@pytest.mark.asyncio
async def test_answers_to_requests_other_tools_created_are_left_alone():
    api = _TheChat()
    adapter = await _adapter(api)
    foreign = api.seed("deploy:prod:42", "answered", {"choice": "yes"})

    await adapter._bot_requests.handle_polled_answers([dict(foreign)])

    assert api.calls("/ack") == []
    assert api.requests[foreign["id"]]["status"] == "answered"


@pytest.mark.asyncio
async def test_answer_after_the_waiter_is_gone_is_acknowledged_as_not_applied():
    api = _TheChat()
    adapter = await _adapter(api)
    stale = api.seed("hermes:approval:gone", "answered", {"choice": "once"})

    await adapter._bot_requests.handle_polled_answers([dict(stale)])

    request = api.requests[stale["id"]]
    assert request["status"] == "delivered"
    assert request["deliveryApplied"] is False
    assert request["deliveryNote"] == thechat_requests.STALE_ACK_NOTE


@pytest.mark.asyncio
async def test_webhook_answer_is_acknowledged_in_the_reply_and_applied_once():
    api = _TheChat()
    adapter = await _adapter(api)
    adapter._webhook_secret = "whsec-test"
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    answered = api.answer("hermes:approval:approval-1", {"choice": "session"})
    payload = {"type": "thechat.bot_request.answered", "request": answered}

    first = await adapter._handle_webhook(_signed(adapter, payload))
    # TheChat retries a push whose reply was lost; the second must not re-resolve anything.
    _queue_approval(request_id="approval-2")
    second = await adapter._handle_webhook(_signed(adapter, payload))

    assert first.status == 200 and json.loads(first.text) == {"ok": True, "applied": True}
    assert json.loads(second.text) == {"ok": True, "applied": True}
    assert entry.result == "session"
    assert len(approval_mod._gateway_queues[SESSION_KEY]) == 1  # approval-2 untouched


@pytest.mark.asyncio
async def test_unsigned_answer_webhook_is_rejected():
    api = _TheChat()
    adapter = await _adapter(api)
    adapter._webhook_secret = "whsec-test"
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    payload = {"type": "thechat.bot_request.answered", "request": api.answer("hermes:approval:approval-1", {"choice": "once"})}
    forged = _signed(adapter, payload)
    forged.headers["X-Webhook-Signature"] = "0" * 64

    response = await adapter._handle_webhook(forged)

    assert response.status == 401
    assert not entry.event.is_set()


def _signed(adapter, payload):
    body = json.dumps(payload, separators=(",", ":"))
    timestamp = str(int(time.time()))
    signature = hmac.new(adapter._webhook_secret.encode(), f"{timestamp}.{body}".encode(), hashlib.sha256).hexdigest()

    class _Request:
        headers = {"X-Webhook-Timestamp": timestamp, "X-Webhook-Signature": signature}

        async def read(self):
            return body.encode()

    return _Request()


@pytest.mark.asyncio
async def test_approval_timeout_withdraws_the_request(monkeypatch):
    """Real wait: the approval times out in the agent thread and the card is withdrawn."""
    api = _TheChat()
    adapter = await _adapter(api)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 1)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    decision: dict[str, Any] = {}

    def notify(data):
        asyncio.run_coroutine_threadsafe(
            adapter.send_exec_approval(CONVERSATION_ID, data["command"], SESSION_KEY, request_id=data["request_id"]),
            loop,
        ).result(timeout=5)

    thread = threading.Thread(
        target=lambda: decision.update(wait_mod._await_gateway_decision(SESSION_KEY, notify, {"command": "rm -rf build"})))
    thread.start()
    await asyncio.to_thread(thread.join, 10)
    await _settle_tasks()

    assert decision["resolved"] is False
    (request,) = api.requests.values()
    assert request["status"] == "cancelled"
    assert request["cancelReason"] == thechat_requests.SETTLE_REASONS["timeout"]


@pytest.mark.asyncio
async def test_inbox_answer_during_a_real_wait_runs_the_command_path(monkeypatch):
    api = _TheChat()
    adapter = await _adapter(api)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 30)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    decision: dict[str, Any] = {}
    sent = threading.Event()

    def notify(data):
        asyncio.run_coroutine_threadsafe(
            adapter.send_exec_approval(CONVERSATION_ID, data["command"], SESSION_KEY, request_id=data["request_id"]),
            loop,
        ).result(timeout=5)
        sent.set()

    thread = threading.Thread(
        target=lambda: decision.update(wait_mod._await_gateway_decision(SESSION_KEY, notify, {"command": "make deploy"})))
    thread.start()
    await asyncio.to_thread(sent.wait, 5)
    (request,) = api.requests.values()
    await adapter._bot_requests.handle_polled_answers([api.answer(request["clientKey"], {"choice": "always"})])
    await asyncio.to_thread(thread.join, 10)
    await _settle_tasks()

    assert decision == {"resolved": True, "choice": "always", "reason": None}
    assert request["status"] == "delivered" and request["deliveryApplied"] is True
    assert api.calls("/cancel") == []


@pytest.mark.asyncio
async def test_slash_approve_closes_the_inbox_card():
    api = _TheChat()
    adapter = await _adapter(api)
    _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    resolved: list[str] = []
    assert approval_mod.resolve_gateway_approval(SESSION_KEY, "once", resolved_request_ids=resolved) == 1

    await adapter.send_approval_resolution(CONVERSATION_ID, SESSION_KEY, "once", request_ids=resolved)

    request = api.by_key("hermes:approval:approval-1")
    assert request["status"] == "cancelled"
    assert request["cancelReason"] == "Allowed once in the conversation."


@pytest.mark.asyncio
async def test_withdrawal_that_races_an_inbox_answer_reports_the_answer_as_too_late():
    api = _TheChat()
    adapter = await _adapter(api)
    _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    api.answer("hermes:approval:approval-1", {"choice": "once"})  # answered, not yet delivered

    await adapter._bot_requests.cancel("hermes:approval:approval-1", "timed out")

    request = api.by_key("hermes:approval:approval-1")
    assert request["status"] == "delivered" and request["deliveryApplied"] is False


def test_settle_hooks_chain_instead_of_replacing_each_other():
    entry = _queue_approval()
    calls: list[tuple[str, str]] = []
    assert approval_mod.register_gateway_settle(SESSION_KEY, entry.request_id, lambda r: calls.append(("card", r)))
    assert approval_mod.register_gateway_settle(SESSION_KEY, entry.request_id, lambda r: calls.append(("notice", r)))

    entry.settle("timeout")

    assert calls == [("card", "timeout"), ("notice", "timeout")]


@pytest.mark.asyncio
async def test_named_target_inbox_approval_is_scoped_to_that_target(monkeypatch):
    """Named execution targets: the card says which target, and Always allow covers only that one."""
    from tools import approval_context

    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(wait_mod._ctx, "_get_approval_timeout", lambda: 30)
    monkeypatch.setattr(wait_mod._ctx, "_fire_approval_hook", lambda *a, **k: None)
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda *_a, **_k: {"action": "allow", "findings": [], "summary": ""},
    )
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    api = _TheChat()
    adapter = await _adapter(api)
    loop = asyncio.get_running_loop()
    command = "rm -rf /tmp/hermes-thechat-target-scope"

    def notify(data):
        # What the gateway runner sends for a button-capable adapter.
        asyncio.run_coroutine_threadsafe(
            adapter.send_exec_approval(
                CONVERSATION_ID, data["command"], SESSION_KEY, description=data["description"],
                allow_permanent=data["allow_permanent"], allow_session=data["allow_session"],
                request_id=data["request_id"],
            ),
            loop,
        ).result(timeout=5)

    def guard(target):
        token = approval_context.set_current_session_key(SESSION_KEY)
        try:
            return approval_mod.check_all_command_guards(
                command, "ssh", execution_target=target, execution_backend="ssh",
                execution_target_named=True, execution_target_scope=f"{target}-scope",
            )
        finally:
            approval_context.reset_current_session_key(token)

    async def open_request(count):
        for _ in range(500):
            if len(api.requests) == count:
                return list(api.requests.values())[-1]
            await asyncio.sleep(0.01)
        raise AssertionError(f"expected {count} TheChat request(s), got {len(api.requests)}")

    approval_mod.register_gateway_notify(SESSION_KEY, notify)
    try:
        devbox_wait = asyncio.ensure_future(asyncio.to_thread(guard, "devbox"))
        devbox = await open_request(1)
        assert "execution target: 'devbox'; backend: ssh" in devbox["body"]["body"]
        assert devbox["body"]["details"] == {"command": command}
        assert devbox["body"]["response"]["choices"] == ["once", "session", "always", "deny"]
        await adapter._bot_requests.handle_polled_answers([api.answer(devbox["clientKey"], {"choice": "always"})])
        assert (await asyncio.wait_for(devbox_wait, 10))["approved"] is True
        assert devbox["status"] == "delivered" and devbox["deliveryApplied"] is True
        with approval_mod._lock:
            permanent = set(approval_mod._permanent_set())
        assert any(key.startswith("target:devbox-scope:") for key in permanent), permanent

        # The same target is now allowed without a new card.
        assert (await asyncio.to_thread(guard, "devbox"))["approved"] is True
        assert len(api.requests) == 1

        # A sibling target is not covered by that answer: TheChat asks again.
        prod_wait = asyncio.ensure_future(asyncio.to_thread(guard, "prod"))
        prod = await open_request(2)
        assert "execution target: 'prod'; backend: ssh" in prod["body"]["body"]
        await adapter._bot_requests.handle_polled_answers([api.answer(prod["clientKey"], {"choice": "deny"})])
        assert (await asyncio.wait_for(prod_wait, 10))["approved"] is False
        assert prod["status"] == "delivered" and prod["deliveryApplied"] is True
    finally:
        approval_mod.unregister_gateway_notify(SESSION_KEY)
        with approval_mod._lock:
            approval_mod._session_approved.pop(SESSION_KEY, None)
            approval_mod._permanent_approved.clear()


# -- clarify ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clarify_with_choices_becomes_a_question_with_other_and_keeps_typed_fallback():
    api = _TheChat()
    adapter = await _adapter(api)
    clarify_gateway.register("clarify-1", SESSION_KEY, "Which env?", ["staging", "prod"])

    result = await adapter.send_clarify(CONVERSATION_ID, "Which env?", ["staging", "prod"], "clarify-1", SESSION_KEY)

    assert result.success
    body = api.calls("/bot-requests")[0]["json"]
    assert body["clientKey"] == "hermes:clarify:clarify-1"
    assert body["kind"] == "question" and body["title"] == "Which env?"
    assert body["response"] == {"type": "choice", "choices": ["staging", "prod"], "multiple": False, "allowText": True}
    assert clarify_gateway._entries["clarify-1"].awaiting_text is True


@pytest.mark.asyncio
async def test_recommended_marker_is_a_label_not_part_of_the_answer():
    api = _TheChat()
    adapter = await _adapter(api)
    shown = ["staging (Recommended)", "prod"]
    clarify_gateway.register("clarify-1", SESSION_KEY, "Which env?", shown)
    await adapter.send_clarify(CONVERSATION_ID, "Which env?", shown, "clarify-1", SESSION_KEY)

    response = api.calls("/bot-requests")[0]["json"]["response"]
    assert response["choices"] == ["staging", "prod"]
    assert response["labels"] == {"staging": "staging (Recommended)"}
    await adapter._bot_requests.handle_polled_answers([api.answer("hermes:clarify:clarify-1", {"choice": "staging"})])
    assert clarify_gateway.wait_for_response("clarify-1", timeout=1) == "staging"


@pytest.mark.asyncio
async def test_open_clarify_answer_resolves_the_waiter():
    api = _TheChat()
    adapter = await _adapter(api)
    clarify_gateway.register("clarify-1", SESSION_KEY, "Name the branch", None)
    await adapter.send_clarify(CONVERSATION_ID, "Name the branch", None, "clarify-1", SESSION_KEY)
    assert api.calls("/bot-requests")[0]["json"]["response"] == {"type": "text"}

    await adapter._bot_requests.handle_polled_answers([api.answer("hermes:clarify:clarify-1", {"text": "feat/inbox"})])

    assert clarify_gateway.wait_for_response("clarify-1", timeout=1) == "feat/inbox"
    assert api.by_key("hermes:clarify:clarify-1")["deliveryApplied"] is True


@pytest.mark.asyncio
async def test_multi_select_answer_is_passed_to_the_tool_as_a_json_list():
    api = _TheChat()
    adapter = await _adapter(api)
    clarify_gateway.register("clarify-1", SESSION_KEY, "Which?", ["a", "b", "c"], multi_select=True)
    await adapter.send_clarify(CONVERSATION_ID, "Which?", ["a", "b", "c"], "clarify-1", SESSION_KEY)
    assert api.calls("/bot-requests")[0]["json"]["response"]["multiple"] is True

    await adapter._bot_requests.handle_polled_answers([api.answer("hermes:clarify:clarify-1", {"choices": ["a", "c"]})])

    assert json.loads(clarify_gateway.wait_for_response("clarify-1", timeout=1)) == ["a", "c"]


@pytest.mark.asyncio
async def test_typed_clarify_answer_closes_the_card_without_echoing_the_answer():
    api = _TheChat()
    adapter = await _adapter(api)
    clarify_gateway.register("clarify-1", SESSION_KEY, "Password hint?", None)
    await adapter.send_clarify(CONVERSATION_ID, "Password hint?", None, "clarify-1", SESSION_KEY)
    assert clarify_gateway.resolve_text_response_for_session(SESSION_KEY, "my secret answer")

    await adapter.send_clarify_resolution(CONVERSATION_ID, SESSION_KEY, "clarify-1", "my secret answer")
    await adapter.retire_clarify_card("clarify-1", "✅ answered: my secret answer")

    request = api.by_key("hermes:clarify:clarify-1")
    assert request["status"] == "cancelled"
    assert request["cancelReason"] == thechat_requests.ANSWERED_IN_CHAT_REASON
    assert len(api.calls("/cancel")) == 1


@pytest.mark.asyncio
async def test_expired_clarify_is_withdrawn_with_the_gateway_notice():
    api = _TheChat()
    adapter = await _adapter(api)
    clarify_gateway.register("clarify-1", SESSION_KEY, "Which env?", ["staging", "prod"])
    await adapter.send_clarify(CONVERSATION_ID, "Which env?", ["staging", "prod"], "clarify-1", SESSION_KEY)

    await adapter.retire_clarify_card("clarify-1", "⏳ This prompt expired — please send a new request.")

    request = api.by_key("hermes:clarify:clarify-1")
    assert request["status"] == "cancelled"
    assert request["cancelReason"] == "This prompt expired — please send a new request."


@pytest.mark.asyncio
async def test_clarify_too_large_for_a_card_falls_back_to_text():
    api = _TheChat()
    adapter = await _adapter(api)
    choices = [f"option {i}" for i in range(25)]
    clarify_gateway.register("clarify-1", SESSION_KEY, "Pick", choices)

    result = await adapter.send_clarify(CONVERSATION_ID, "Pick", choices, "clarify-1", SESSION_KEY)

    assert not result.success and not result.retryable
    assert api.calls("/bot-requests") == []


@pytest.mark.asyncio
async def test_threaded_request_targets_the_invocation_thread():
    api = _TheChat()
    adapter = await _adapter(api, context=False)
    adapter._contexts[f"{CONVERSATION_ID}:thread:{THREAD_ID}"] = {**_context(), "thread_id": THREAD_ID}
    clarify_gateway.register("clarify-1", SESSION_KEY, "Q?", None)

    await adapter.send_clarify(CONVERSATION_ID, "Q?", None, "clarify-1", SESSION_KEY, metadata={"thread_id": THREAD_ID})

    body = api.calls("/bot-requests")[0]["json"]
    assert body["threadId"] == THREAD_ID and body["invocationId"] == INVOCATION_ID


# -- startup ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_cancels_leftover_requests_and_rejects_answers_it_cannot_use():
    api = _TheChat()
    leftover = api.seed("hermes:approval:old", "open")
    late_answer = api.seed("hermes:clarify:old", "answered", {"text": "too late"})
    foreign = api.seed("ci:deploy:7", "open")
    adapter = await _adapter(api)

    await adapter._bot_requests.reconcile()

    assert leftover["status"] == "cancelled"
    assert leftover["cancelReason"] == thechat_requests.RESTART_CANCEL_REASON
    assert late_answer["status"] == "delivered" and late_answer["deliveryApplied"] is False
    assert late_answer["deliveryNote"] == thechat_requests.RESTART_ACK_NOTE
    assert foreign["status"] == "open"


@pytest.mark.asyncio
async def test_reconnect_keeps_requests_this_process_is_still_waiting_on():
    api = _TheChat()
    adapter = await _adapter(api)
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    api.answer("hermes:approval:approval-1", {"choice": "once"})  # answered while disconnected

    await adapter._bot_requests.reconcile()

    assert entry.result == "once"
    assert api.by_key("hermes:approval:approval-1")["deliveryApplied"] is True


@pytest.mark.asyncio
async def test_connect_starts_cleanup_only_when_thechat_supports_requests(monkeypatch):
    for capability, expected in ((True, True), (False, False)):
        api = _TheChat(capability=capability)
        api.seed("hermes:approval:old", "open")
        adapter = TheChatAdapter(PlatformConfig(enabled=True, token="t", extra={"base_url": "http://thechat.test"}))
        monkeypatch.setattr(thechat_requests.httpx, "AsyncClient", httpx.AsyncClient)
        monkeypatch.setattr("gateway.platforms.thechat.httpx.AsyncClient", lambda **_kw: api)

        async def _no_commands(self):
            return None

        monkeypatch.setattr(TheChatAdapter, "_register_commands", _no_commands)
        monkeypatch.setattr(TheChatAdapter, "_poll_loop", lambda self: asyncio.sleep(0))
        assert await adapter.connect()
        await _settle_tasks()
        await adapter.disconnect()

        (request,) = api.requests.values()
        assert (request["status"] == "cancelled") is expected


@pytest.mark.asyncio
async def test_poll_loop_applies_answers_before_messages(monkeypatch):
    api = _TheChat()
    adapter = await _adapter(api)
    entry = _queue_approval()
    await adapter.send_exec_approval(CONVERSATION_ID, "rm -rf build", SESSION_KEY, request_id="approval-1")
    api.poll_answers = [api.answer("hermes:approval:approval-1", {"choice": "once"})]
    adapter._running = True
    adapter.poll_interval = 0.01

    task = asyncio.create_task(adapter._poll_loop())
    for _ in range(100):
        if entry.event.is_set():
            break
        await asyncio.sleep(0.01)
    adapter._running = False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert entry.result == "once"
