"""Bind TheChat's native masked vault prompt on the actual agent executor thread."""
from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from contextlib import contextmanager


def _publish(turn, adapter, context, event):
    loop = turn._ctx._loop_for_step
    if loop is None or loop.is_closed():
        return None
    return asyncio.run_coroutine_threadsafe(adapter.send_invocation_progress(
        context["conversation_id"], event, context=context), loop)


@contextmanager
def vault_turn_scope(turn):
    from gateway.config import Platform
    ctx = turn._ctx
    if ctx.source.platform != Platform.THECHAT:
        yield
        return
    from agent.vault_backends import unlock
    from gateway.platforms.thechat import TheChatAdapter
    from gateway.platforms.thechat_vault import PREVIEW
    from hermes_constants import get_hermes_home

    adapter = turn._runner._delivery_adapter_for(ctx.source)
    home = str(get_hermes_home())
    profile_id = hashlib.sha256(home.encode()).hexdigest()
    context = adapter._event_contexts.get(str(ctx.source.message_id or "")) if isinstance(adapter, TheChatAdapter) else None
    context = dict(context) if context else None
    owner = adapter._owner_user_id if isinstance(adapter, TheChatAdapter) else ""
    allowed = bool(context and ctx.source.user_id and context.get("requester_user_id") == ctx.source.user_id
                   and context["conversation_id"] == ctx.source.chat_id
                   and context.get("thread_id") == ctx.source.thread_id)
    active = threading.Event()
    active.set()

    def live():
        return bool(active.is_set() and allowed
                    and not adapter._vault_unlock.closed and adapter._client is not None
                    and ctx._run_still_current() and not turn._agent_interrupted()
                    and str(get_hermes_home()) == home)

    records = []

    def prompt(backend, _display):
        if backend != "bitwarden" or not live():
            return ""
        record = adapter._vault_unlock.create(context=context, profile_id=profile_id, session_key=ctx.session_key)
        records.append(record)
        future = None
        try:
            if not live() or not unlock.unlock_attempt_is_current(backend):
                adapter._vault_unlock.cancel(record)
                return ""
            future = _publish(turn, adapter, context, {
                "type": "vault.unlock.request", "status": "waiting", "label": "Unlock Bitwarden",
                "preview": PREVIEW, "payload": dict(record.payload)})
            if future is None:
                adapter._vault_unlock.cancel(record, outcome="failed")
            while not record.event.wait(0.1):
                if not live() or not unlock.unlock_attempt_is_current(backend):
                    adapter._vault_unlock.cancel(record)
                elif record.deadline <= time.monotonic() or record.payload["expiresAt"] <= int(time.time() * 1000):
                    adapter._vault_unlock.cancel(record, outcome="expired")
                elif future is not None and future.done():
                    try:
                        sent = future.result()
                        if not sent.success:
                            adapter._vault_unlock.cancel(record, outcome="failed")
                    except Exception:
                        adapter._vault_unlock.cancel(record, outcome="failed")
            if not live() or not unlock.unlock_attempt_is_current(backend):
                adapter._vault_unlock.cancel(record)
                return ""
            response = record.take_response()
            if record.outcome == "failed":
                raise RuntimeError("Secure vault prompt could not be delivered")
            return response
        finally:
            # Public outcome is prompt delivery only; native backend success follows separately.
            if record.outcome is None:
                adapter._vault_unlock.cancel(record)
            if future is not None and not future.done():
                future.cancel()
            resolved = _publish(turn, adapter, context, {
                "type": "vault.unlock.resolved", "status": "completed", "label": "Bitwarden unlock prompt closed",
                "preview": "Secure unlock prompt closed.", "payload": {
                    "version": 2, "requestId": record.payload["requestId"], "sessionKey": ctx.session_key,
                    "outcome": record.outcome}}) if adapter._client is not None else None
            if resolved is not None:
                try:
                    resolved.result(timeout=3)
                except Exception:
                    resolved.cancel()
            adapter._vault_unlock.release(record)

    previous = unlock.get_unlock_prompt_callback()
    previous_session = unlock.get_current_session_id()
    # Do not borrow a Desktop/CLI thread-local prompt on a reused gateway executor.
    unlock.set_unlock_prompt_callback(prompt if allowed and adapter.webhook_url else None)
    unlock.set_current_session_id(ctx.session_id)
    try:
        with unlock.external_access_scope(
                lambda: bool(owner and allowed and ctx.source.user_id == owner and adapter._owner_user_id == owner),
                live=live, bitwarden_allowed=allowed):
            yield
    finally:
        active.clear()
        for record in records:
            adapter._vault_unlock.release(record)
        unlock.set_unlock_prompt_callback(previous)
        unlock.set_current_session_id(previous_session)
