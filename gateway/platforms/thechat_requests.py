"""TheChat bot requests: approvals and clarify prompts as durable inbox requests.

TheChat stores each prompt as a ``bot_requests`` row, shows it inline and in the Inbox, and
hands the answer back through the poll (``requestAnswers``) or a signed webhook. Hermes still
blocks on its in-memory ``threading.Event``; this module only connects the two sides:

* create: ``send_exec_approval`` / ``send_clarify`` call ``POST /bot-requests``. The Hermes
  request id is the ``clientKey``, so retrying an ambiguous create never makes a second card.
* answer: the waiter is found by ``clientKey`` and woken, then the answer is acknowledged with
  ``applied`` (false when the wait had already ended, so the card says so).
* cancel: when the wait ends any other way (timeout, /stop, run end, typed answer) the request
  is withdrawn, so no answerable card is left behind.
* startup: requests a previous process left open are cancelled ("Hermes restarted"); answers
  that arrived while Hermes was down are acknowledged as not applied. A blocked tool call cannot
  be resumed in a new process, so nothing tries to.

Only used when TheChat advertises ``capabilities.botRequests``; older servers keep the legacy
progress-event cards in ``gateway.platforms.thechat``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, UTC
from typing import Any, Callable, Dict, Optional

import httpx

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

CAPABILITY = "botRequests"
CLIENT_KEY_PREFIX = "hermes:"
WEBHOOK_TYPE = "thechat.bot_request.answered"

APPROVAL_TITLE = "Command approval required"
APPROVAL_LABELS = {
    "once": "Allow once",
    "session": "Allow for this session",
    "always": "Always allow",
    "deny": "Deny",
}
_RESOLVED_IN_CHAT = {
    "once": "Allowed once in the conversation.",
    "session": "Allowed for this session in the conversation.",
    "always": "Always allowed in the conversation.",
    "deny": "Denied in the conversation.",
}

RESTART_CANCEL_REASON = "Hermes restarted. Ask again."
RESTART_ACK_NOTE = "Hermes restarted before it could use this answer."
STALE_ACK_NOTE = "Hermes was no longer waiting for this answer."
ANSWERED_IN_CHAT_REASON = "Answered in the conversation."
# Why an approval wait ended without a TheChat answer (``tools.approval_gateway_wait`` settle reasons).
SETTLE_REASONS = {
    "timeout": "Nobody answered in time, so Hermes did not run the command.",
    "interrupted": "The run was stopped before anyone answered.",
    "session_closed": "The run ended before anyone answered.",
    "notify_failed": "Hermes could not deliver this request.",
    "resolved": ANSWERED_IN_CHAT_REASON,
}
DEFAULT_CANCEL_REASON = "Hermes withdrew this request."

# Mirrors TheChat's limits (packages/api/src/services/bot-requests.ts).
_TITLE_MAX = 300
_BODY_MAX = 20_000
_DETAILS_MAX_BYTES = 128 * 1024
_CHOICE_MAX = 500
_CHOICE_COUNT_MAX = 20
_LABEL_MAX = 100
_REASON_MAX = 500
_CLIENT_KEY_MAX = 255
_MAX_EXPIRY_SECONDS = 30 * 24 * 60 * 60 - 60
# TheChat expiry is only a safety net for a Hermes that died; Hermes' own timer cancels first.
_EXPIRY_GRACE_SECONDS = 60
_HTTP_MAX_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.25
_PENDING_MAX = 512
_OUTCOME_CACHE_MAX = 1024
_LIST_PAGE = 200
_LIST_MAX_PAGES = 10
_ANSWER_SWEEP_SECONDS = 30.0

_TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)


@dataclass
class _Pending:
    """One prompt this process is waiting on, keyed by ``client_key``."""

    client_key: str
    kind: str  # "approval" | "clarify"
    local_id: str
    session_key: str
    chat_id: str
    create_body: dict[str, Any]
    choices: list[str] = field(default_factory=list)
    multi_select: bool = False
    remote_id: Optional[str] = None


def client_key(kind: str, local_id: str) -> str:
    return f"{CLIENT_KEY_PREFIX}{kind}:{local_id}"


def _bounded(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _expires_at(timeout_seconds: Any) -> Optional[str]:
    try:
        seconds = int(timeout_seconds)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    seconds = min(seconds + _EXPIRY_GRACE_SECONDS, _MAX_EXPIRY_SECONDS)
    moment = datetime.now(UTC) + timedelta(seconds=seconds)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _title_and_body(text: str) -> tuple[str, Optional[str]]:
    """A long question keeps a short title and carries the full text in the body."""
    if len(text) <= _TITLE_MAX:
        return text, None
    first_line = text.splitlines()[0] if text.splitlines() else text
    return _bounded(first_line, _TITLE_MAX), _bounded(text, _BODY_MAX)


def _order_group(session_key: str) -> str:
    # Approvals of one session are answered oldest first, like /approve. Opaque on purpose.
    return "hermes:approval:" + hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:40]


def _error_text(response: Any) -> str:
    try:
        data = response.json()
    except Exception:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    return _bounded(error, 200) if isinstance(error, str) else ""


def _is_hermes_request(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and bool(item["id"])
        and isinstance(item.get("clientKey"), str)
        and item["clientKey"].startswith(CLIENT_KEY_PREFIX)
    )


class BotRequestBridge:
    """Owned by ``TheChatAdapter``; all state is touched on the adapter's event loop only.

    Settle hooks from ``tools.approval`` run on the agent thread and are hopped onto the loop.
    """

    def __init__(self, adapter: Any, *, redact: Callable[[Any], Any]):
        self._adapter = adapter
        self._redact = redact
        self.enabled = False
        self._pending: dict[str, _Pending] = {}
        # TheChat request id -> (applied, note); answers can arrive twice (poll + ack lost, or
        # a webhook retry) and must be acknowledged identically, never applied twice.
        self._outcomes: OrderedDict[str, tuple[bool, Optional[str]]] = OrderedDict()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: set[asyncio.Task] = set()
        self._sweep_task: Optional[asyncio.Task] = None

    # -- lifecycle ---------------------------------------------------------------------------

    def capture_capabilities(self, health: Any) -> None:
        capabilities = health.get("capabilities") if isinstance(health, dict) else None
        version = capabilities.get(CAPABILITY) if isinstance(capabilities, dict) else None
        self.enabled = isinstance(version, int) and not isinstance(version, bool) and version >= 1

    def start(self, *, sweep_answers: bool) -> None:
        """Clean up after a previous process; in webhook mode also sweep for stuck answers."""
        if not self.enabled:
            return
        self._loop = asyncio.get_running_loop()
        self._spawn(self.reconcile(), "thechat-bot-requests-reconcile")
        if sweep_answers and self._sweep_task is None:
            self._sweep_task = asyncio.create_task(
                self._answer_sweep_loop(), name="thechat-bot-requests-sweep"
            )

    async def stop(self) -> None:
        tasks = [task for task in (self._sweep_task, *self._tasks) if task is not None]
        self._sweep_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _spawn(self, coro: Any, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _call_soon(self, coro: Any) -> None:
        """Run *coro* on the adapter loop from any thread (settle hooks run on the agent thread)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            coro.close()

    @property
    def _client(self) -> Any:
        return getattr(self._adapter, "_client", None)

    # -- create --------------------------------------------------------------------------------

    @staticmethod
    def _target(chat_id: str, context: Optional[dict[str, Any]], thread_id: Optional[str]) -> dict[str, Any]:
        if context:
            target: dict[str, Any] = {
                "invocationId": str(context["invocation_id"]),
                "conversationId": str(context["conversation_id"]),
            }
            thread_id = str(context.get("thread_id") or "") or None
        else:
            target = {"conversationId": str(chat_id)}
        if thread_id:
            target["threadId"] = str(thread_id)
        return target

    def _track(self, pending: _Pending) -> Optional[SendResult]:
        if pending.client_key in self._pending:
            return SendResult(success=False, error="Duplicate pending TheChat bot request")
        if len(self._pending) >= _PENDING_MAX:
            return SendResult(success=False, error="Too many pending TheChat bot requests")
        if len(pending.client_key) > _CLIENT_KEY_MAX:
            return SendResult(success=False, error="TheChat bot request id is too long")
        self._pending[pending.client_key] = pending
        return None

    async def send_exec_approval(
        self,
        *,
        chat_id: str,
        command: str,
        description: str,
        session_key: str,
        request_id: str,
        choices: list[str],
        context: Optional[dict[str, Any]],
        thread_id: Optional[str],
    ) -> SendResult:
        from gateway.platforms.base_exec_approval import approval_timeout_seconds

        self._loop = asyncio.get_running_loop()
        safe_command = str(self._redact(str(command)))
        safe_description = _bounded(self._redact(str(description or "")), _BODY_MAX)
        details = {"command": safe_command}
        if len(json.dumps(details, ensure_ascii=False).encode("utf-8")) > _DETAILS_MAX_BYTES:
            return SendResult(success=False, error="TheChat approval request is too large")
        key = client_key("approval", request_id)
        body: dict[str, Any] = {
            "clientKey": key,
            "kind": "approval",
            "title": APPROVAL_TITLE,
            "body": safe_description or None,
            "details": details,
            # "always" is offered whenever Hermes offers it; TheChat does not narrow the choices.
            "response": {
                "type": "choice",
                "choices": list(choices),
                "labels": {choice: APPROVAL_LABELS[choice] for choice in choices if choice in APPROVAL_LABELS},
                "allowComment": True,
            },
            "orderGroup": _order_group(session_key),
            **self._target(chat_id, context, thread_id),
        }
        expires_at = _expires_at(approval_timeout_seconds())
        if expires_at:
            body["expiresAt"] = expires_at
        pending = _Pending(
            client_key=key, kind="approval", local_id=request_id, session_key=session_key,
            chat_id=str(chat_id), create_body=body, choices=list(choices),
        )
        rejected = self._track(pending)
        if rejected is not None:
            return rejected

        from tools.approval import register_gateway_settle

        armed = register_gateway_settle(session_key, request_id, self._approval_settle_hook(key))
        result = await self._create(pending)
        if not result.success and not result.retryable:
            self._pending.pop(key, None)
        elif not armed:
            # The wait ended before the hook could attach: withdraw what was just posted.
            await self.cancel(key, DEFAULT_CANCEL_REASON)
        return result

    async def send_clarify(
        self,
        *,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        multi_select: bool,
        context: Optional[dict[str, Any]],
        thread_id: Optional[str],
    ) -> SendResult:
        from tools import clarify_gateway

        self._loop = asyncio.get_running_loop()
        from tools.clarify_tool import strip_recommended

        safe_question = str(self._redact(str(question))).strip()
        shown = [str(self._redact(str(choice))).strip() for choice in choices] if choices else []
        # "(Recommended)" is presentation: the answer value stays the bare choice the agent offered.
        safe_choices = [strip_recommended(choice) for choice in shown]
        labels = {
            value: display
            for value, display in zip(safe_choices, shown)
            if display != value and len(display) <= _LABEL_MAX
        }
        if (
            not clarify_id
            or not safe_question
            or len(safe_question) > _BODY_MAX
            or len(safe_choices) > _CHOICE_COUNT_MAX
            or any(not choice or len(choice) > _CHOICE_MAX for choice in safe_choices)
            or len(set(safe_choices)) != len(safe_choices)
        ):
            # Non-retryable: the gateway falls back to the plain-text prompt.
            return SendResult(success=False, error="TheChat clarify request cannot be shown as a card")
        title, long_body = _title_and_body(safe_question)
        response: dict[str, Any] = (
            {"type": "choice", "choices": safe_choices, "multiple": bool(multi_select), "allowText": True}
            if safe_choices
            else {"type": "text"}
        )
        if labels:
            response["labels"] = labels
        key = client_key("clarify", clarify_id)
        body: dict[str, Any] = {
            "clientKey": key,
            "kind": "question",
            "title": title,
            "body": long_body,
            "response": response,
            **self._target(chat_id, context, thread_id),
        }
        expires_at = _expires_at(clarify_gateway.get_clarify_timeout())
        if expires_at:
            body["expiresAt"] = expires_at
        pending = _Pending(
            client_key=key, kind="clarify", local_id=clarify_id, session_key=session_key,
            chat_id=str(chat_id), create_body=body, choices=safe_choices,
            multi_select=bool(multi_select and safe_choices),
        )
        rejected = self._track(pending)
        if rejected is not None:
            return rejected
        result = await self._create(pending)
        if result.success:
            # Typed answers keep working next to the card; slash commands still bypass capture.
            clarify_gateway.mark_awaiting_text(clarify_id)
            if clarify_gateway.get_request_state(clarify_id) != "pending":
                await self.cancel(key, DEFAULT_CANCEL_REASON)
        elif not result.retryable:
            self._pending.pop(key, None)
        return result

    async def _create(self, pending: _Pending) -> SendResult:
        """POST the request; the same ``clientKey`` makes every retry return the same row."""
        client = self._client
        if client is None:
            return SendResult(success=False, error="TheChat client is not connected", retryable=True)
        error = "TheChat bot request was not created"
        for attempt in range(_HTTP_MAX_ATTEMPTS):
            try:
                response = await client.post("/bot-requests", json=pending.create_body)
            except _TRANSPORT_ERRORS as exc:
                error = f"TheChat bot request transport failed: {type(exc).__name__}"
            else:
                status = int(getattr(response, "status_code", 0) or 0)
                if status in (200, 201):
                    try:
                        request = response.json().get("request")
                    except Exception:
                        request = None
                    if isinstance(request, dict) and isinstance(request.get("id"), str):
                        pending.remote_id = request["id"]
                        return SendResult(success=True, raw_response={"requestId": request["id"]})
                    error = "TheChat returned an invalid bot request"
                elif status >= 500 or status in (408, 429):
                    error = f"TheChat bot request failed with HTTP {status}"
                else:
                    detail = _error_text(response)
                    error = f"TheChat rejected the bot request (HTTP {status})" + (f": {detail}" if detail else "")
                    logger.warning("TheChat: %s", error)
                    return SendResult(success=False, error=error)
            if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                await asyncio.sleep(_RETRY_BASE_SECONDS * (2**attempt))
        logger.warning("TheChat: %s", error)
        return SendResult(success=False, error=error, retryable=True)

    # -- answers -------------------------------------------------------------------------------

    def apply_answer(self, request: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Wake the waiter for an answered request; returns the ``(applied, note)`` to ack."""
        remote_id = str(request["id"])
        cached = self._outcomes.get(remote_id)
        if cached is not None:
            return cached
        pending = self._pending.get(str(request.get("clientKey")))
        outcome = (False, STALE_ACK_NOTE) if pending is None else self._wake(pending, request.get("answer"))
        self._outcomes[remote_id] = outcome
        while len(self._outcomes) > _OUTCOME_CACHE_MAX:
            self._outcomes.popitem(last=False)
        return outcome

    def _wake(self, pending: _Pending, answer: Any) -> tuple[bool, Optional[str]]:
        if not isinstance(answer, dict):
            return False, "Hermes could not read this answer."
        if pending.kind == "approval":
            choice = answer.get("choice")
            if choice not in pending.choices:
                return False, "Hermes did not offer this choice."
            comment = answer.get("comment")
            from tools.approval import resolve_gateway_approval

            applied = resolve_gateway_approval(
                pending.session_key,
                str(choice),
                request_id=pending.local_id,
                reason=comment.strip() if isinstance(comment, str) and comment.strip() else None,
            ) == 1
        else:
            if isinstance(answer.get("text"), str):
                value = answer["text"]
            elif isinstance(answer.get("choice"), str) and answer["choice"] in pending.choices:
                value = answer["choice"]
            elif (
                pending.multi_select
                and isinstance(answer.get("choices"), list)
                and answer["choices"]
                and all(isinstance(item, str) and item in pending.choices for item in answer["choices"])
            ):
                value = json.dumps(answer["choices"], ensure_ascii=False)
            else:
                return False, "Hermes could not read this answer."
            from tools.clarify_gateway import resolve_gateway_clarify

            applied = resolve_gateway_clarify(pending.local_id, value)
        if not applied:
            return False, STALE_ACK_NOTE
        # Forget before yielding: the approval settle hook that the wake triggers must find
        # nothing to withdraw.
        self._pending.pop(pending.client_key, None)
        try:
            self._adapter.resume_typing_for_chat(pending.chat_id)
        except Exception:
            logger.debug("TheChat: resume typing after a bot request answer failed", exc_info=True)
        return True, None

    async def handle_polled_answers(self, items: Any) -> None:
        if not self.enabled or not isinstance(items, list):
            return
        for item in items:
            # Answers to requests other tools created with this bot token are not Hermes' to ack.
            if not _is_hermes_request(item):
                continue
            applied, note = self.apply_answer(item)
            await self.ack(str(item["id"]), applied, note)

    def handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Answer pushed by TheChat's outbox; a 2xx reply with ``applied`` is the acknowledgement."""
        request = payload.get("request")
        if not isinstance(request, dict) or not isinstance(request.get("id"), str):
            raise ValueError("Bot request webhook does not contain a request")
        if not self.enabled or not _is_hermes_request(request) or request.get("status") != "answered":
            return {"ok": True}
        applied, note = self.apply_answer(request)
        body: dict[str, Any] = {"ok": True, "applied": applied}
        if note:
            body["note"] = note
        return body

    async def ack(self, remote_id: str, applied: bool, note: Optional[str]) -> bool:
        body: dict[str, Any] = {"applied": bool(applied)}
        if note:
            body["note"] = _bounded(note, _REASON_MAX)
        return await self._post_with_retry(f"/bot-requests/{remote_id}/ack", body, "acknowledge")

    # -- withdraw ------------------------------------------------------------------------------

    def _approval_settle_hook(self, key: str) -> Callable[[str], None]:
        def settle(reason: str) -> None:
            self._call_soon(self._after_approval_wait(key, reason))

        return settle

    async def _after_approval_wait(self, key: str, reason: str) -> None:
        # A TheChat answer already removed the entry; anything left ended without one.
        if key in self._pending:
            await self.cancel(key, SETTLE_REASONS.get(reason, DEFAULT_CANCEL_REASON))

    async def cancel(self, key: str, reason: str) -> None:
        pending = self._pending.pop(key, None)
        if pending is None or self._client is None:
            return
        if pending.remote_id is None:
            # The create may have landed with its reply lost; the same clientKey finds it.
            created = await self._create(pending)
            if not created.success:
                logger.warning("TheChat: could not withdraw a bot request that may be open")
                return
        await self._cancel_remote(str(pending.remote_id), reason)

    async def _cancel_remote(self, remote_id: str, reason: str) -> None:
        client = self._client
        if client is None:
            return
        body = {"reason": _bounded(self._redact(str(reason)), _REASON_MAX)}
        for attempt in range(_HTTP_MAX_ATTEMPTS):
            try:
                response = await client.post(f"/bot-requests/{remote_id}/cancel", json=body)
            except _TRANSPORT_ERRORS:
                pass
            else:
                status = int(getattr(response, "status_code", 0) or 0)
                if status == 409:
                    # Answered in the meantime: say the answer came too late, so the card does
                    # not wait for a pickup that will never happen.
                    try:
                        current = response.json().get("request") or {}
                    except Exception:
                        current = {}
                    if current.get("status") == "answered":
                        self._outcomes[remote_id] = (False, STALE_ACK_NOTE)
                        await self.ack(remote_id, False, STALE_ACK_NOTE)
                    return
                if status < 500 and status not in (408, 429):
                    return
            if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                await asyncio.sleep(_RETRY_BASE_SECONDS * (2**attempt))
        logger.warning("TheChat: could not withdraw bot request %s", remote_id)

    async def _post_with_retry(self, path: str, body: dict[str, Any], what: str) -> bool:
        client = self._client
        if client is None:
            return False
        for attempt in range(_HTTP_MAX_ATTEMPTS):
            try:
                response = await client.post(path, json=body)
            except _TRANSPORT_ERRORS:
                pass
            else:
                status = int(getattr(response, "status_code", 0) or 0)
                if 200 <= status < 300 or status == 409:
                    return True
                if status < 500 and status not in (408, 429):
                    logger.warning("TheChat: could not %s a bot request (HTTP %s)", what, status)
                    return False
            if attempt + 1 < _HTTP_MAX_ATTEMPTS:
                await asyncio.sleep(_RETRY_BASE_SECONDS * (2**attempt))
        logger.warning("TheChat: could not %s a bot request", what)
        return False

    # -- gateway callbacks ---------------------------------------------------------------------

    async def approvals_resolved_in_chat(self, request_ids: list[str], choice: str) -> None:
        """``/approve`` or ``/deny`` answered these in the conversation; close their cards."""
        reason = _RESOLVED_IN_CHAT.get(choice, ANSWERED_IN_CHAT_REASON)
        for request_id in request_ids:
            await self.cancel(client_key("approval", str(request_id)), reason)

    async def clarify_closed(self, clarify_id: str, reason: str) -> None:
        await self.cancel(client_key("clarify", str(clarify_id)), reason or DEFAULT_CANCEL_REASON)

    # -- recovery ------------------------------------------------------------------------------

    async def _list(self, status: str) -> list[dict[str, Any]]:
        client = self._client
        if client is None:
            return []
        response = await client.get("/bot-requests", params={"status": status, "limit": _LIST_PAGE})
        response.raise_for_status()
        items = response.json().get("requests")
        return [item for item in items if _is_hermes_request(item)] if isinstance(items, list) else []

    async def reconcile(self) -> None:
        """Close what a previous process left behind; deliver answers this one is waiting for.

        Requests this process is still waiting on (a reconnect, not a restart) are kept.
        """
        try:
            for status in ("open", "answered"):
                seen: set[str] = set()
                for _ in range(_LIST_MAX_PAGES):
                    items = await self._list(status)
                    fresh = [item for item in items if item["id"] not in seen]
                    for item in fresh:
                        seen.add(item["id"])
                        live = item["clientKey"] in self._pending
                        if status == "open":
                            if not live:
                                await self._cancel_remote(item["id"], RESTART_CANCEL_REASON)
                        elif live:
                            applied, note = self.apply_answer(item)
                            await self.ack(item["id"], applied, note)
                        else:
                            self._outcomes[item["id"]] = (False, RESTART_ACK_NOTE)
                            await self.ack(item["id"], False, RESTART_ACK_NOTE)
                    if len(items) < _LIST_PAGE or not fresh:
                        break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("TheChat: bot request cleanup after startup failed", exc_info=True)

    async def _answer_sweep_loop(self) -> None:
        """Webhook mode safety net: pick up answers whose push never succeeded."""
        while True:
            await asyncio.sleep(_ANSWER_SWEEP_SECONDS)
            if not self._pending:
                continue
            try:
                for item in await self._list("answered"):
                    if item["clientKey"] in self._pending or item["id"] in self._outcomes:
                        applied, note = self.apply_answer(item)
                        await self.ack(item["id"], applied, note)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("TheChat: bot request answer sweep failed", exc_info=True)
