"""TheChat invocation-bound workspace authority (never a session credential)."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
from datetime import datetime, timezone
import asyncio
import logging
import json
import re
import uuid
import httpx

from gateway.inbound_event_ledger import (
    accept_inbound_event, claim_inbound_event, complete_inbound_event,
    InboundEventConflictError,
)

logger = logging.getLogger(__name__)
_bound_work = ContextVar("thechat_invocation_work", default=None)


def bound_work_context(adapter, chat_id, metadata=None):
    bound = _bound_work.get()
    if bound is None or bound[0] is not adapter:
        return None
    context = bound[1]
    thread = (metadata or {}).get("thread_id") or (metadata or {}).get("message_thread_id")
    if context and context.get("conversation_id") == chat_id and (not thread or thread == context.get("thread_id")):
        return context
    return None


def validate_work_token(item: dict, *, legacy: bool) -> None:
    token = item.get("workToken")
    required = bool(item["conversation"]["workspaceId"]) and not legacy
    if token is None and not required:
        return
    if not isinstance(token, str) or not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", token
    ):
        raise ValueError("TheChat event requires a valid workToken")


@dataclass(frozen=True)
class WorkAuthority:
    invocation_id: str
    token: str = field(repr=False)


def work_payload(context: dict, payload: dict) -> dict:
    """The received epoch wins over caller-supplied progress/message metadata."""
    result = {key: value for key, value in payload.items() if key != "workToken"}
    authority = context.get("work_authority")
    if authority is not None:
        result["workToken"] = authority.token
    return result


class WorkLeaseLost(RuntimeError):
    """Safe, terminal authority failure; never a reason to replay tool effects."""


async def post_work(adapter, context: dict, path: str, payload: dict, *, raise_status=True):
    lease = context.get("work_lease")
    if lease is not None:
        lease.ensure_live()
    response = await adapter._client.post(path, json=work_payload(context, payload))
    if lease is not None and response.status_code in {400, 401, 403, 404, 409, 410, 422}:
        lease.lose(f"callback HTTP {response.status_code}")
        raise WorkLeaseLost("TheChat rejected invocation authority")
    if raise_status:
        response.raise_for_status()
    return response


def work_is_queued(adapter, event, context):
    if not context or not context.get("work_lease"):
        return False
    key = adapter._event_session_key(event)
    queued = [adapter._pending_messages.get(key)]
    if adapter.gateway_runner is not None:
        queued.extend(adapter.gateway_runner._overflow_queue(key) or [])
    return any(candidate is not None and candidate.message_id == event.message_id for candidate in queued)


def defer_workspace_followups(adapter, session_key):
    queued = [getattr(adapter, "_pending_messages", {}).get(session_key)]
    runner = getattr(adapter, "gateway_runner", None)
    if runner is not None:
        queued.extend(runner._overflow_queue(session_key) or [])
    if any(isinstance(getattr(event, "raw_message", None), dict) and event.raw_message.get("workToken")
           for event in queued):
        return True
    return any(lease.event is not None and adapter._event_session_key(lease.event) == session_key
               for lease in getattr(adapter, "_work_leases", ()))


@dataclass
class WorkAdmissionLane:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class WorkLease:
    """One acknowledged remote epoch, independent of typing and model activity."""

    def __init__(self, adapter, authority: WorkAuthority):
        self.adapter = adapter
        self.authority = authority
        self.deadline = 0.0
        self.closed = False
        self.task = None
        self.run_task = None
        self.intake_task = None
        self.event = None

    def ensure_live(self):
        if self.closed or self.deadline <= asyncio.get_running_loop().time():
            self.lose("acknowledged deadline expired")
            raise WorkLeaseLost("TheChat invocation is no longer authorized")

    async def start(self):
        await self.renew()
        self.task = asyncio.create_task(self._heartbeat(), name=f"thechat-work-{self.authority.invocation_id}")

    async def renew(self):
        loop = asyncio.get_running_loop()
        started = loop.time()
        timeout = min(10.0, self.deadline - started) if self.deadline else 10.0
        if timeout <= 0 or self.closed:
            raise WorkLeaseLost("TheChat acknowledged work lease expired")
        async with asyncio.timeout(timeout):
            response = await self.adapter._client.post(
                f"/hermes-platform/invocations/{self.authority.invocation_id}/heartbeat",
                json={"workToken": self.authority.token},
            )
        response.raise_for_status()
        body = response.json()
        expires = body.get("leaseExpiresAt")
        if body.get("ok") is not True or not isinstance(expires, str):
            raise WorkLeaseLost("TheChat heartbeat omitted its lease deadline")
        expiry = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expiry.tzinfo is None:
            raise WorkLeaseLost("TheChat heartbeat deadline must include a timezone")
        # Subtract the complete request duration conservatively. Subsequent waits
        # use monotonic time, never a moving wall clock or an assumed server TTL.
        self.deadline = started + (expiry - datetime.now(timezone.utc)).total_seconds()
        if self.deadline <= loop.time():
            raise WorkLeaseLost("TheChat acknowledged work lease expired")

    async def _heartbeat(self):
        try:
            while not self.closed:
                remaining = self.deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise WorkLeaseLost("TheChat acknowledged work lease expired")
                await asyncio.sleep(min(20.0, max(0.01, remaining / 3)))
                try:
                    await self.renew()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in {408, 429} and exc.response.status_code < 500:
                        raise
                    # Only a successful heartbeat can extend this deadline.
                except (TimeoutError, httpx.TransportError):
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("TheChat work heartbeat stopped: invocation=%s cause=%s",
                           self.authority.invocation_id, type(exc).__name__)
            self.lose(type(exc).__name__)

    def lose(self, reason):
        if self.closed:
            return
        logger.warning("TheChat work authority lost: invocation=%s reason=%s", self.authority.invocation_id, reason)
        self.close()
        self.adapter._vault_unlock.cancel_invocation(self.authority.invocation_id)
        from tools.approval import withdraw_gateway_approval
        from tools.clarify_gateway import resolve_gateway_clarify
        for request_id, record in list(self.adapter._interaction_requests.items()):
            if record["invocation_id"] != self.authority.invocation_id:
                continue
            if record["request_type"] == "approval.request":
                withdraw_gateway_approval(record["session_key"], request_id, "thechat_work_lease_lost")
            else:
                resolve_gateway_clarify(request_id, "")
            self.adapter._forget_interaction_request(request_id)
        runner = self.adapter.gateway_runner
        if runner is not None and self.event is not None:
            key = self.adapter._event_session_key(self.event)
            state = runner._peek_session_state(key)
            running_event = state.turn.event if state else None
            # An inline control or an old snapshot must never interrupt a newer
            # turn in the same session. Use the runner's actual event and task.
            if (running_event is not None
                    and running_event.message_id == self.event.message_id
                    and self.adapter._session_tasks.get(key) is self.run_task):
                with runner._profile_scope_for_source(self.event.source):
                    generation = runner._interrupt_running_turn(
                        key, interrupt_reason="TheChat work authority lost",
                        invalidation_reason="thechat_work_lease_lost",
                        tool_reason="thechat_work_lease_lost",
                    )
                    runner._drop_turn_slot(key, run_generation=generation)
        if self.run_task is not None and not self.run_task.done():
            self.adapter._expected_cancelled_tasks.add(self.run_task)
            self.run_task.cancel()
        if self.intake_task is not None and not self.intake_task.done():
            self.intake_task.cancel()

    def close(self):
        self.closed = True
        if self.task is not None and self.task is not asyncio.current_task():
            self.task.cancel()
        self.adapter._work_leases.discard(self)


class TheChatWorkMixin:
    """Invocation lifetime seams around the existing base adapter pipeline."""

    async def _handle_platform_event(self, item):
        item = self._validate_platform_event(item)
        context = self._context_from_platform_event(item)
        if context is None:
            raise ValueError("TheChat event missing invocation context")
        context["source"] = self._source_from_platform_event(item, context)
        key = self._source_session_key(context["source"])
        lane = self._work_admission_lanes.setdefault(key, WorkAdmissionLane())
        lane.users += 1
        # Reserve FIFO position before the inbox/heartbeat awaits can reorder
        # arrivals. Acquiring in a separate task leaves every initial ACK live.
        admission = asyncio.create_task(lane.lock.acquire())
        try:
            await self._handle_work_platform_event(item, context, admission)
        finally:
            if not admission.done():
                admission.cancel()
            await asyncio.gather(admission, return_exceptions=True)
            if not admission.cancelled() and admission.result():
                lane.lock.release()
            lane.users -= 1
            if not lane.users:
                self._work_admission_lanes.pop(key, None)

    async def _handle_work_platform_event(self, item, context, admission):
        authority = context.get("work_authority")
        if authority is None:
            await asyncio.shield(admission)
            return await self._deliver_platform_event(item, context)
        if not await asyncio.to_thread(self._consume_work_once, item):
            logger.info("TheChat skipped consumed work: invocation=%s", authority.invocation_id)
            return
        lease = context["work_lease"] = WorkLease(self, authority)
        try:
            await lease.start()
        except Exception as exc:
            lease.close()
            raise WorkLeaseLost("TheChat could not acknowledge work authority") from exc
        self._work_leases.add(lease)
        # Cancel only this delivery, never the shared polling/recovery consumer.
        lease.intake_task = asyncio.create_task(self._deliver_bound_work(item, context, admission))
        try:
            await lease.intake_task
        except asyncio.CancelledError:
            if lease.closed:
                return
            lease.lose("intake cancelled")
            raise
        except Exception as exc:
            try:
                await post_work(self, context,
                                f"/hermes-platform/invocations/{authority.invocation_id}/failed",
                                {"error": "Hermes gateway failed to process platform event"})
            finally:
                lease.close()
            raise WorkLeaseLost("TheChat invocation intake failed; not replayable") from exc

    async def _deliver_bound_work(self, item, context, admission):
        token = _bound_work.set((self, context))
        try:
            await asyncio.shield(admission)
            context["work_lease"].ensure_live()
            await self._deliver_platform_event(item, context)
        finally:
            _bound_work.reset(token)

    async def _process_message_background(self, event, session_key):
        context = self._event_contexts.get(str(event.message_id or ""))
        token = _bound_work.set((self, context))
        try:
            await super()._process_message_background(event, session_key)
        finally:
            _bound_work.reset(token)

    def _consume_work_once(self, item):
        # Reuse the inbox CAS, with a separate namespace: "completed" here means
        # the execution permit was consumed, NOT that the remote turn succeeded.
        # Commit before external effects. A crash after this point is uncertain
        # work, never automatically replayable (even while the DB epoch is live).
        source = f"{self._inbound_event_source()}:work:{item['bot']['id']}"
        event_id = item["invocationId"]
        payload = json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        try:
            acceptance = accept_inbound_event(source=source, event_id=event_id, payload=payload)
        except InboundEventConflictError:
            raise WorkLeaseLost("TheChat invocation authority or payload changed") from None
        if acceptance.status == "completed":
            return False
        owner = uuid.uuid4().hex
        claim = claim_inbound_event(source=source, event_id=event_id, lease_owner=owner,
                                    lease_seconds=60, max_attempts=1)
        if claim is None:
            return False
        if not complete_inbound_event(source=source, event_id=event_id, lease_owner=owner):
            raise WorkLeaseLost("TheChat execution fence could not be persisted")
        return True

    async def handle_message(self, event):
        raw = event.raw_message
        requires_work = bool(getattr(event.source, "guild_id", None)) and self.config.extra.get("workspace_work_lease_mode") != "legacy"
        if requires_work or (isinstance(raw, dict) and raw.get("workToken")):
            context = self._event_contexts.get(str(event.message_id or "")) or {}
            lease = context.get("work_lease")
            if lease is None or lease.closed:
                logger.warning("TheChat refused replay without live local work: message=%s", event.message_id)
                event._gateway_accepted = False
                return
            lease.ensure_live()
        await super().handle_message(event)

    async def _handle_message_while_active(self, event, session_key):
        context = self._event_contexts.get(str(event.message_id or "")) or {}
        if (context.get("work_lease") and not event.get_command()
                and self._busy_session_handler is None):
            from tools.clarify_gateway import get_pending_for_session
            # A wired runner owns approval/auth routing before its leased FIFO.
            # Without that handler, keep the standalone adapter's FIFO fallback.
            if get_pending_for_session(session_key, include_choice_prompts=True) is None:
                if self.gateway_runner is not None:
                    self.gateway_runner._enqueue_fifo(session_key, event, self)
                elif session_key not in self._pending_messages:
                    self._pending_messages[session_key] = event
                    event._gateway_accepted = True
                else:
                    from gateway.platforms.base import ProcessingOutcome
                    await self.on_processing_complete(event, ProcessingOutcome.CANCELLED)
                return
        await super()._handle_message_while_active(event, session_key)

    def _spawn_drain_task(self, pending_event, session_key):
        raw = pending_event.raw_message
        if isinstance(raw, dict) and raw.get("workToken") and self.gateway_runner is not None:
            # The next adapter task, not recursive _run_agent, owns this epoch.
            # Stage overflow now so the runner cannot misclassify it as orphaned
            # work and substitute a different event under this task's hooks.
            self.gateway_runner._promote_queued_event(session_key, self, pending_event)
        super()._spawn_drain_task(pending_event, session_key)
