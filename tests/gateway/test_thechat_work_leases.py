"""Workspace epoch contract through real adapter lifecycle and loopback HTTP.

Only the local HTTP service and agent body are synthetic. No provider or external
TheChat endpoint is contacted; the base adapter, HTTP client and inbox are real.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import uuid

from aiohttp import web
import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter


TOKEN = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
CONVERSATION = "11111111-1111-4111-8111-111111111111"


def platform_event(*, token=TOKEN, workspace="workspace-1", thread="thread-1"):
    invocation = str(uuid.uuid4())
    event = {
        "id": invocation, "invocationId": invocation,
        "chatId": CONVERSATION, "chatType": "dm", "threadId": thread,
        "messageId": str(uuid.uuid4()), "text": "synthetic task",
        "sender": {"id": "human-1", "name": "Human"},
        "bot": {"id": "bot-1", "userId": "bot-user-1", "name": "Hermes"},
        "conversation": {"id": CONVERSATION, "type": "direct", "name": "Test", "workspaceId": workspace},
    }
    if token is not None:
        event["workToken"] = token
    return event


class LoopbackBackend:
    def __init__(self):
        self.calls = []
        self.terminal = asyncio.Event()
        self.heartbeat = asyncio.Event()
        self.heartbeats = asyncio.Queue()
        self.lease_seconds = 30
        self.reject = {}
        self.events = []
        # In-memory contract model, not evidence of durable TheChat DB state.
        self.active_invocations = set()
        self.terminal_invocations = {}
        self.heartbeat_invocations = asyncio.Queue()

    async def handle(self, request):
        if request.method == "GET":
            events, self.events = self.events, []
            return web.json_response({"events": events})
        body = await request.json() if request.can_read_body else {}
        self.calls.append((request.path, body))
        status = self.reject.get(request.path, 200)
        if status != 200:
            return web.json_response({"error": "synthetic rejection"}, status=status)
        if request.path.endswith("/heartbeat"):
            invocation = request.path.split("/")[-2]
            if invocation in self.terminal_invocations:
                return web.json_response({"error": "terminal invocation"}, status=409)
            self.active_invocations.add(invocation)
            self.heartbeat_invocations.put_nowait(invocation)
            self.heartbeat.set()
            self.heartbeats.put_nowait(body)
            return web.json_response({"ok": True, "leaseExpiresAt": (datetime.now(timezone.utc) + timedelta(seconds=self.lease_seconds)).isoformat()})
        if request.path.endswith(("/completed", "/failed", "/cancelled")):
            invocation = request.path.split("/")[-2]
            self.active_invocations.discard(invocation)
            self.terminal_invocations[invocation] = request.path.rsplit("/", 1)[-1]
            self.terminal.set()
        return web.json_response({"ok": True, "messageId": "reply-1"})


@asynccontextmanager
async def connected_adapter(backend=None, **extra):
    backend = backend or LoopbackBackend()
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", backend.handle)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    adapter = TheChatAdapter(PlatformConfig(enabled=True, token="bot_synthetic", extra={"base_url": f"http://127.0.0.1:{port}", **extra}))
    adapter.config.typing_indicator = False
    adapter._client = httpx.AsyncClient(base_url=adapter.base_url, headers={"Authorization": "Bearer bot_synthetic"})
    try:
        yield adapter, backend
    finally:
        await adapter.cancel_background_tasks()
        await adapter.disconnect()
        await server.cleanup()


async def settled(adapter):
    while adapter._background_tasks:
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks), return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_received_epoch_is_immutable_on_every_success_callback():
    item = platform_event()
    async with connected_adapter() as (adapter, backend):
        async def agent(event):
            # Neither mutable input nor progress payload may replace the epoch.
            item["workToken"] = str(uuid.uuid4())
            metadata = {"message_id": event.message_id, "thread_id": event.source.thread_id}
            assert (await adapter.send_invocation_progress(CONVERSATION, {"type": "tool.started", "workToken": "forged"}, metadata)).success
            await adapter.send_typing(CONVERSATION, metadata)
            assert (await adapter.send_session_title_update(CONVERSATION, "Synthetic title", metadata)).success
            return "Synthetic result"
        adapter.set_message_handler(agent)
        await adapter._handle_platform_event(item)
        await settled(adapter)
        paths = {path.rsplit("/", 1)[-1] for path, _ in backend.calls}
        assert {"progress", "typing", "messages", "completed"} <= paths
        assert all(body.get("workToken") == TOKEN for _, body in backend.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "ingress_failure"])
async def test_failure_and_cancellation_echo_received_epoch(outcome):
    from gateway.platforms.base import ProcessingOutcome
    item = platform_event()
    async with connected_adapter() as (adapter, backend):
        async def receive(event):
            await adapter.on_processing_start(event)
            item["workToken"] = str(uuid.uuid4())
            await adapter.on_processing_complete(event, ProcessingOutcome(outcome))
        if outcome == "ingress_failure":
            async def broken_download(_):
                raise RuntimeError("synthetic download failure")
            adapter._download_inbound_attachments = broken_download
            await adapter._handle_platform_event_safely(item)
        else:
            adapter.handle_message = receive
            await adapter._handle_platform_event(item)
        terminal = [(path, body) for path, body in backend.calls if path.endswith(("/failed", "/cancelled"))]
        assert len(terminal) == 1
        assert terminal[0][1].get("workToken") == TOKEN
        assert not adapter._work_leases


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, "", " AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "aaaaaaaaaaaa4aaa8aaaaaaaaaaaaaaa", 42])
async def test_required_workspace_epoch_is_rejected_before_download_or_execution(token):
    async with connected_adapter() as (adapter, backend):
        effects = []
        async def download(_):
            effects.append("download")
            return [], [], []
        async def agent(_):
            effects.append("execution")
        adapter._download_inbound_attachments = download
        adapter.set_message_handler(agent)
        with pytest.raises(ValueError, match="workToken"):
            await adapter._handle_platform_event(platform_event(token=token))
        assert effects == []
        assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace,extra", [(None, {}), ("old-workspace", {"workspace_work_lease_mode": "legacy"})])
async def test_explicit_legacy_and_personal_events_keep_tokenless_contract(workspace, extra):
    async with connected_adapter(**extra) as (adapter, backend):
        async def agent(_):
            return "Compatible response"
        adapter.set_message_handler(agent)
        await adapter._handle_platform_event(platform_event(token=None, workspace=workspace))
        await settled(adapter)
        assert any(path.endswith("/completed") for path, _ in backend.calls)
        assert all("workToken" not in body for _, body in backend.calls)
        assert not any(path.endswith("/heartbeat") for path, _ in backend.calls)


@pytest.mark.asyncio
async def test_silent_heartbeat_covers_attachment_and_paused_human_wait_lifetime():
    backend = LoopbackBackend()
    backend.lease_seconds = 0.9
    async with connected_adapter(backend) as (adapter, _):
        attachment_started, release_attachment = asyncio.Event(), asyncio.Event()
        human_wait, release_human = asyncio.Event(), asyncio.Event()
        async def download(_):
            attachment_started.set()
            await release_attachment.wait()
            return [], [], []
        async def agent(event):
            adapter.pause_typing_for_chat(CONVERSATION)
            human_wait.set()
            await release_human.wait()
            return "Done after human wait"
        adapter._download_inbound_attachments = download
        adapter.set_message_handler(agent)
        intake = asyncio.create_task(adapter._handle_platform_event(platform_event()))
        try:
            await asyncio.wait_for(attachment_started.wait(), 5)
            for _ in range(3):
                assert (await asyncio.wait_for(backend.heartbeats.get(), 5))["workToken"] == TOKEN
            assert all(path.endswith("/heartbeat") for path, _ in backend.calls)
            release_attachment.set()
            await intake
            await asyncio.wait_for(human_wait.wait(), 5)
            for _ in range(3):
                await asyncio.wait_for(backend.heartbeats.get(), 5)
            assert all(path.endswith("/heartbeat") for path, _ in backend.calls)
            release_human.set()
            await settled(adapter)
            assert backend.calls[-1][0].endswith("/completed")
        finally:
            release_attachment.set()
            release_human.set()
            await asyncio.gather(intake, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("route,status", [("progress", 403), ("typing", 404), ("messages", 409), ("heartbeat", 401), ("heartbeat", 500)])
async def test_authority_loss_cancels_only_exact_invocation_and_fences_late_egress(route, status):
    backend = LoopbackBackend()
    backend.lease_seconds = 0.6
    first, sibling = platform_event(thread="first"), platform_event(thread="sibling", token=str(uuid.uuid4()))
    started = {item["invocationId"]: asyncio.Event() for item in (first, sibling)}
    cancelled = {key: asyncio.Event() for key in started}
    release = asyncio.Event()
    async with connected_adapter(backend) as (adapter, _):
        async def agent(event):
            key = event.raw_message["invocationId"]
            started[key].set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled[key].set()
                raise
            return "Sibling survives"
        adapter.set_message_handler(agent)
        for item in (first, sibling):
            await adapter._handle_platform_event(item)
            await asyncio.wait_for(started[item["invocationId"]].wait(), 5)
        context = dict(adapter._event_contexts[first["messageId"]])
        path = f"/hermes-platform/invocations/{first['invocationId']}/{route}" if route in {"progress", "heartbeat"} else f"/hermes-platform/{route}"
        backend.reject[path] = status
        metadata = {"message_id": first["messageId"], "thread_id": "first"}
        if route == "progress":
            result = await adapter.send_invocation_progress(CONVERSATION, {"type": "tool.started"}, metadata)
            assert not result.success and not result.retryable
        elif route == "typing":
            await adapter.send_typing(CONVERSATION, metadata)
        elif route == "messages":
            result = await adapter.send(CONVERSATION, "Rejected", metadata=metadata)
            assert not result.success and not result.retryable
        await asyncio.wait_for(cancelled[first["invocationId"]].wait(), 5)
        assert not cancelled[sibling["invocationId"]].is_set()
        # Captured title contexts and retries must not become unscoped sends or
        # borrow the other invocation after cleanup removes a chat context.
        before = len(backend.calls)
        result = await adapter.send_session_title_update(CONVERSATION, "late", context=context)
        assert not result.success and not result.retryable
        result = await adapter.send(CONVERSATION, "late", metadata=metadata)
        assert not result.success and not result.retryable
        assert len(backend.calls) == before
        backend.reject.clear()
        release.set()
        await settled(adapter)
        messages = [body for path, body in backend.calls if path.endswith("/messages") and body.get("content") == "Sibling survives"]
        assert [body["invocationId"] for body in messages] == [sibling["invocationId"]]


@pytest.mark.asyncio
async def test_lease_loss_uses_native_runner_hard_interrupt_before_task_unwinds():
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    interrupted = []
    started = asyncio.Event()
    class NativeAgent:
        def hard_interrupt(self, message, *, tool_reason=None):
            interrupted.append((message, tool_reason))
    async with connected_adapter() as (adapter, backend):
        adapter.gateway_runner = runner
        runner.adapters[Platform.THECHAT] = adapter
        adapter.set_message_handler(runner._handle_message)
        async def native_turn(event, source, key, generation):
            runner._session_state(key).turn.agent = NativeAgent()
            started.set()
            await asyncio.Event().wait()
        runner._handle_message_with_agent = native_turn
        item = platform_event()
        await adapter._handle_platform_event(item)
        await asyncio.wait_for(started.wait(), 5)
        backend.reject[f"/hermes-platform/invocations/{item['invocationId']}/progress"] = 403
        await adapter.send_invocation_progress(CONVERSATION, {"type": "tool.started"}, {"message_id": item["messageId"]})
        await settled(adapter)
        assert len(interrupted) == 1
        assert interrupted[0][1] == "thechat_work_lease_lost"


@pytest.mark.asyncio
async def test_workspace_queue_ack_keeps_lease_until_actual_queued_turn_finishes():
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    first_started, release_first = asyncio.Event(), asyncio.Event()
    queued_started, release_queued = asyncio.Event(), asyncio.Event()
    async with connected_adapter() as (adapter, backend):
        adapter.gateway_runner = runner
        runner.adapters[Platform.THECHAT] = adapter
        adapter.set_message_handler(runner._handle_message)
        first, queued = platform_event(), platform_event(token=str(uuid.uuid4()))
        queued["text"] = "/queue second prompt"
        async def native_turn(event, source, key, generation):
            if event.message_id == first["messageId"]:
                first_started.set()
                await release_first.wait()
            else:
                queued_started.set()
                await release_queued.wait()
            return event.text
        runner._handle_message_with_agent = native_turn
        await adapter._handle_platform_event(first)
        await asyncio.wait_for(first_started.wait(), 5)
        await adapter._handle_platform_event(queued)
        assert not any(path.endswith(f"/{queued['invocationId']}/completed") for path, _ in backend.calls)
        assert any(body.get("content", "").startswith("Queued for the next turn") for _, body in backend.calls)
        release_first.set()
        await asyncio.wait_for(queued_started.wait(), 5)
        release_queued.set()
        await settled(adapter)
        completed = [path.split("/")[-2] for path, _ in backend.calls if path.endswith("/completed")]
        assert completed == [first["invocationId"], queued["invocationId"]]
        assert [body["workToken"] for path, body in backend.calls if path.endswith("/messages") and body.get("content") == "second prompt"] == [queued["workToken"]]


@pytest.mark.asyncio
async def test_terminal_event_replay_cannot_rehydrate_an_epoch_into_another_turn():
    saved = []
    async with connected_adapter() as (adapter, backend):
        async def agent(event):
            saved.append(event)
            return "once"
        adapter.set_message_handler(agent)
        await adapter._handle_platform_event(platform_event())
        await settled(adapter)
        await adapter.handle_message(saved[0])
        await settled(adapter)
        assert len(saved) == 1
        assert len([path for path, _ in backend.calls if path.endswith("/messages")]) == 1


@pytest.mark.asyncio
async def test_authority_rejection_cancels_attachment_intake_without_killing_consumer():
    backend = LoopbackBackend()
    backend.lease_seconds = 0.6
    item = platform_event()
    started, cancelled = asyncio.Event(), asyncio.Event()
    async with connected_adapter(backend) as (adapter, _):
        async def download(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        adapter._download_inbound_attachments = download
        intake = asyncio.create_task(adapter._handle_platform_event_safely(item))
        try:
            await asyncio.wait_for(started.wait(), 5)
            backend.reject[f"/hermes-platform/invocations/{item['invocationId']}/heartbeat"] = 403
            await asyncio.wait_for(cancelled.wait(), 5)
            assert await asyncio.wait_for(intake, 5) is True
            assert not any(path.endswith(("/messages", "/failed", "/completed")) for path, _ in backend.calls)
        finally:
            intake.cancel()
            await asyncio.gather(intake, return_exceptions=True)


@pytest.mark.asyncio
async def test_restart_and_duplicate_delivery_do_not_execute_consumed_work_twice():
    effects = []
    async with connected_adapter() as (adapter, backend):
        async def agent(event):
            effects.append(event.message_id)
            return "effect already attempted"
        item = platform_event()
        adapter.set_message_handler(agent)
        await adapter._handle_platform_event(item)
        await settled(adapter)
        # A new adapter represents restart; backend acceptance alone is not
        # evidence that an uncertain prior external effect is safe to replay.
        restarted = TheChatAdapter(adapter.config)
        restarted.config.typing_indicator = False
        restarted._client = httpx.AsyncClient(base_url=adapter.base_url)
        restarted.set_message_handler(agent)
        try:
            await restarted._handle_platform_event(item)
            await settled(restarted)
            assert effects == [item["messageId"]]
            assert len([path for path, _ in backend.calls if path.endswith("/heartbeat")]) == 1
        finally:
            await restarted.cancel_background_tasks()
            await restarted.disconnect()


@pytest.mark.asyncio
async def test_polling_batch_acknowledges_sibling_while_first_download_is_blocked():
    backend = LoopbackBackend()
    slow, fast = platform_event(thread="slow"), platform_event(thread="fast")
    slow["attachments"] = [{"synthetic": "barrier"}]
    backend.events = [slow, fast]
    download_started, release_download, fast_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async with connected_adapter(backend) as (adapter, _):
        async def download(attachments):
            if attachments:
                download_started.set()
                await release_download.wait()
            return [], [], []
        async def agent(event):
            if event.message_id == fast["messageId"]:
                fast_done.set()
            return "local result"
        adapter._download_inbound_attachments = download
        adapter.set_message_handler(agent)
        adapter._running = True
        poll = asyncio.create_task(adapter._poll_loop())
        try:
            await asyncio.wait_for(download_started.wait(), 5)
            await asyncio.wait_for(fast_done.wait(), 5)
            assert not release_download.is_set()
            assert any(path.endswith(f"/{fast['invocationId']}/heartbeat") for path, _ in backend.calls)
        finally:
            release_download.set()
            adapter._running = False
            poll.cancel()
            await asyncio.gather(poll, return_exceptions=True)


@pytest.mark.asyncio
async def test_distinct_workspace_messages_queue_without_merging_or_interrupting_epochs():
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    started, release = asyncio.Event(), asyncio.Event()
    seen = []
    async with connected_adapter() as (adapter, backend):
        runner.adapters[Platform.THECHAT] = adapter
        adapter.gateway_runner = runner
        adapter.set_message_handler(runner._handle_message)
        items = [platform_event(token=str(uuid.uuid4())) for _ in range(3)]
        async def turn(event, source, key, generation):
            seen.append(event.message_id)
            if event.message_id == items[0]["messageId"]:
                started.set()
                await release.wait()
            return event.message_id
        runner._handle_message_with_agent = turn
        await adapter._handle_platform_event(items[0])
        await asyncio.wait_for(started.wait(), 5)
        for item in items[1:]:
            await adapter._handle_platform_event(item)
        release.set()
        await settled(adapter)
        assert seen == [item["messageId"] for item in items]
        terminal = [path.split("/")[-2] for path, _ in backend.calls if path.endswith("/completed")]
        assert terminal == [item["invocationId"] for item in items]


@pytest.mark.asyncio
async def test_native_inband_drain_leaves_leased_followup_for_its_own_adapter_task():
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    async with connected_adapter() as (adapter, _):
        item = platform_event()
        source = adapter.build_source(chat_id=CONVERSATION, chat_type="dm", user_id="human-1", thread_id="thread-1")
        event = MessageEvent(text="queued", source=source, message_id=item["messageId"], raw_message=item)
        key = adapter._event_session_key(event)
        adapter._pending_messages[key] = event
        result = await runner._run_agent_drain_pending({"final_response": "first done"}, adapter, source, key)
        assert result == (None, None)
        assert adapter._pending_messages[key] is event


@pytest.mark.asyncio
async def test_attachment_callback_authority_rejection_is_not_retried_or_detached(tmp_path):
    started, cancelled = asyncio.Event(), asyncio.Event()
    path = tmp_path / "synthetic.txt"
    path.write_text("synthetic attachment")
    async with connected_adapter() as (adapter, backend):
        async def agent(event):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        async def reserve(*args, **kwargs):
            return "attachment-synthetic", {}
        async def transfer(*args, **kwargs):
            return None
        # Object storage is a separate protocol; only its upload is synthetic.
        adapter._reserve_attachment = reserve
        adapter._upload_attachment = transfer
        adapter._complete_and_wait_for_attachment = transfer
        adapter.set_message_handler(agent)
        item = platform_event()
        await adapter._handle_platform_event(item)
        await asyncio.wait_for(started.wait(), 5)
        backend.reject["/hermes-platform/messages"] = 409
        result = await adapter.send_document(CONVERSATION, str(path), metadata={"message_id": item["messageId"]})
        assert not result.success and not result.retryable
        await asyncio.wait_for(cancelled.wait(), 5)
        posts = [body for path, body in backend.calls if path.endswith("/messages")]
        assert len(posts) == 1 and posts[0]["workToken"] == TOKEN


@pytest.mark.asyncio
async def test_signed_webhook_recovery_rejects_terminal_epoch_without_execution():
    import hashlib
    import hmac
    import json
    import time
    from gateway.inbound_event_ledger import list_recoverable_inbound_events
    effects = []
    async with connected_adapter(webhook_port=0) as (adapter, backend):
        adapter._webhook_secret = "synthetic-webhook-key"
        await adapter._start_webhook_server()
        async def agent(event):
            effects.append(event.message_id)
        adapter.set_message_handler(agent)
        item = platform_event()
        backend.reject[f"/hermes-platform/invocations/{item['invocationId']}/heartbeat"] = 409
        body = json.dumps({"type": "thechat.hermes_platform.event", "event": item})
        timestamp = str(int(time.time()))
        signature = hmac.new(adapter._webhook_secret.encode(), f"{timestamp}.{body}".encode(), hashlib.sha256).hexdigest()
        async with httpx.AsyncClient() as client:
            response = await client.post(adapter.webhook_url, content=body, headers={"X-Webhook-Timestamp": timestamp, "X-Webhook-Signature": signature})
            assert response.status_code == 200
            while adapter._webhook_tasks:
                await asyncio.gather(*list(adapter._webhook_tasks))
            backend.reject.clear()
            response = await client.post(adapter.webhook_url, content=body, headers={"X-Webhook-Timestamp": timestamp, "X-Webhook-Signature": signature})
            assert response.json() == {"ok": True, "duplicate": True}
        assert effects == []
        assert list_recoverable_inbound_events(source=adapter._inbound_event_source()) == []
        assert len(backend.calls) == 1  # only rejected heartbeat; no failure/reclaim/progress


@pytest.mark.asyncio
async def test_disconnect_revokes_live_work_instead_of_only_stopping_heartbeat():
    started, cancelled = asyncio.Event(), asyncio.Event()
    async with connected_adapter() as (adapter, _):
        async def agent(event):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        adapter.set_message_handler(agent)
        await adapter._handle_platform_event(platform_event())
        await asyncio.wait_for(started.wait(), 5)
        await adapter.disconnect()
        await asyncio.wait_for(cancelled.wait(), 5)
        await settled(adapter)


@pytest.mark.asyncio
async def test_unanchored_in_turn_callbacks_do_not_borrow_a_queued_invocation():
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    started, release = asyncio.Event(), asyncio.Event()
    async with connected_adapter() as (adapter, backend):
        runner.adapters[Platform.THECHAT] = adapter
        adapter.gateway_runner = runner
        adapter.set_message_handler(runner._handle_message)
        first, queued = platform_event(), platform_event(token=str(uuid.uuid4()))
        queued["text"] = "/queue later"
        async def turn(event, source, key, generation):
            if event.message_id == first["messageId"]:
                started.set()
                await release.wait()
                await adapter.send_invocation_progress(CONVERSATION, {"type": "tool.started"}, {"thread_id": "thread-1"})
            return "result"
        runner._handle_message_with_agent = turn
        await adapter._handle_platform_event(first)
        await asyncio.wait_for(started.wait(), 5)
        await adapter._handle_platform_event(queued)
        release.set()
        await settled(adapter)
        progress = [(path, body) for path, body in backend.calls if body.get("type") == "tool.started"]
        assert len(progress) == 1
        assert progress[0][0].endswith(f"/{first['invocationId']}/progress")
        assert progress[0][1]["workToken"] == first["workToken"]


@pytest.mark.asyncio
async def test_lease_loss_withdraws_only_its_exact_native_human_waiters():
    from tools.approval_gateway_wait import _ApprovalEntry
    from tools.approval import _gateway_queues
    from tools import clarify_gateway
    from gateway.platforms.thechat import _DirectInteractionError
    key = "synthetic-shared-wait-key"
    own = _ApprovalEntry({"command": "true"}, request_id="own-approval")
    sibling = _ApprovalEntry({"command": "true"}, request_id="sibling-approval")
    _gateway_queues[key] = [sibling, own]
    own_clarify = clarify_gateway.register("own-clarify", key, "Pick?", ["one"])
    sibling_clarify = clarify_gateway.register("sibling-clarify", key, "Pick?", ["one"])
    started = asyncio.Event()
    try:
        async with connected_adapter() as (adapter, backend):
            item = platform_event()
            async def agent(event):
                metadata = {"message_id": event.message_id}
                await adapter.send_exec_approval(CONVERSATION, "true", key, request_id="own-approval", metadata=metadata)
                await adapter.send_clarify(CONVERSATION, "Pick?", ["one"], "own-clarify", key, metadata=metadata)
                started.set()
                await asyncio.Event().wait()
            adapter.set_message_handler(agent)
            await adapter._handle_platform_event(item)
            await asyncio.wait_for(started.wait(), 5)
            backend.reject[f"/hermes-platform/invocations/{item['invocationId']}/progress"] = 403
            await adapter.send_invocation_progress(CONVERSATION, {"type": "tool.started"}, {"message_id": item["messageId"]})
            await settled(adapter)
            assert own.event.is_set() and own.cancelled == "thechat_work_lease_lost"
            assert own_clarify.event.is_set()
            assert not sibling.event.is_set() and not sibling_clarify.event.is_set()
            assert _gateway_queues[key] == [sibling]
            with pytest.raises(_DirectInteractionError):
                await adapter._resolve_direct_interaction({"requestId": "own-approval"})
    finally:
        _gateway_queues.pop(key, None)
        clarify_gateway.clear_session(key)


@pytest.mark.asyncio
async def test_restored_workspace_wake_without_epoch_cannot_start_a_new_agent_turn():
    from gateway.platforms.event import MessageEvent
    effects = []
    async with connected_adapter() as (adapter, backend):
        async def agent(event):
            effects.append(event.text)
            return "must not run"
        adapter.set_message_handler(agent)
        source = adapter.build_source(chat_id=CONVERSATION, chat_type="dm", user_id="human-1", guild_id="workspace-1")
        restored = MessageEvent(text="Resume interrupted work", source=source, internal=True)
        await adapter.handle_message(restored)
        await settled(adapter)
        assert effects == []
        assert backend.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", [False, True])
async def test_startup_restore_wait_keeps_heartbeat_and_never_replays_revoked_work(revoke):
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    runner = make_runner(Platform.THECHAT)
    runner._startup_restore_in_progress = True
    effects = []
    backend = LoopbackBackend()
    backend.lease_seconds = 0.6
    async with connected_adapter(backend) as (adapter, _):
        adapter.gateway_runner = runner
        runner.adapters[Platform.THECHAT] = adapter
        adapter.set_message_handler(runner._handle_message)
        async def turn(event, source, key, generation):
            effects.append(event.message_id)
            return "restored"
        runner._handle_message_with_agent = turn
        item = platform_event()
        await adapter._handle_platform_event(item)
        await settled(adapter)
        assert effects == []
        assert not any(path.endswith("/completed") for path, _ in backend.calls)
        lease = adapter._event_contexts[item["messageId"]]["work_lease"]
        if revoke:
            backend.reject[f"/hermes-platform/invocations/{item['invocationId']}/heartbeat"] = 403
            await asyncio.wait_for(asyncio.shield(lease.task), 5)
        else:
            for _ in range(3):
                await asyncio.wait_for(backend.heartbeats.get(), 5)
        queued = runner._startup_restore_queue.pop(0)
        queued._hermes_startup_restore_replay = True
        runner._startup_restore_in_progress = False
        await adapter.handle_message(queued)
        await settled(adapter)
        assert effects == ([] if revoke else [item["messageId"]])
        assert len([path for path, _ in backend.calls if path.endswith("/completed")]) == (0 if revoke else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_native_discard_terminalizes_only_removed_queued_work(command):
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner

    runner = make_runner(Platform.THECHAT)
    backend = LoopbackBackend()
    backend.lease_seconds = 0.9
    first, head, overflow = [platform_event(token=str(uuid.uuid4())) for _ in range(3)]
    sibling = platform_event(thread="sibling", token=str(uuid.uuid4()))
    control = platform_event(token=str(uuid.uuid4()))
    control["text"] = command
    started = {item["messageId"]: asyncio.Event() for item in (first, sibling)}
    release = asyncio.Event()
    seen = []
    async with connected_adapter(backend) as (adapter, _):
        adapter.gateway_runner = runner
        runner.adapters[Platform.THECHAT] = adapter
        adapter.set_message_handler(runner._handle_message)

        async def turn(event, source, key, generation):
            seen.append(event.message_id)
            started[event.message_id].set()
            await release.wait()
            return "unrelated sibling result"

        runner._handle_message_with_agent = turn
        try:
            for item in (first, sibling):
                await adapter._handle_platform_event(item)
                await asyncio.wait_for(started[item["messageId"]].wait(), 5)
            for item in (head, overflow):
                await adapter._handle_platform_event(item)
            leases = {item["invocationId"]: adapter._event_contexts[item["messageId"]]["work_lease"]
                      for item in (first, head, overflow, sibling)}
            key = adapter._event_session_key(leases[first["invocationId"]].event)
            assert adapter._pending_messages[key].message_id == head["messageId"]
            assert [e.message_id for e in runner._overflow_queue(key)] == [overflow["messageId"]]

            # Native adapter bypass -> runner /stop or /new -> real queue removal.
            await adapter._handle_platform_event(control)
            assert key not in adapter._pending_messages
            removed = (head, overflow) if command == "/new" else (head,)
            for item in removed:
                invocation = item["invocationId"]
                assert backend.terminal_invocations.get(invocation) == "cancelled"
                assert invocation not in backend.active_invocations
                assert leases[invocation].closed
            if command == "/stop":
                assert [e.message_id for e in runner._overflow_queue(key)] == [overflow["messageId"]]
                assert not leases[overflow["invocationId"]].closed
                assert overflow["invocationId"] in backend.active_invocations
            else:
                assert not runner._overflow_queue(key)
            assert not leases[sibling["invocationId"]].closed
            assert not leases[sibling["invocationId"]].run_task.done()
            assert seen == [first["messageId"], sibling["messageId"]]

            # Observe two actual sibling renewal cycles, not a guessed sleep.
            after_discard = len(backend.calls)
            while not backend.heartbeat_invocations.empty():
                backend.heartbeat_invocations.get_nowait()
            renewals = 0
            while renewals < 2:
                invocation = await asyncio.wait_for(backend.heartbeat_invocations.get(), 5)
                renewals += invocation == sibling["invocationId"]
            removed_ids = {item["invocationId"] for item in removed}
            assert not any(path.endswith("/heartbeat") and path.split("/")[-2] in removed_ids
                           for path, _ in backend.calls[after_discard:])
            for item in removed:
                callbacks = [(path, body) for path, body in backend.calls
                             if path.endswith(f"/{item['invocationId']}/cancelled")]
                assert len(callbacks) == 1
                assert callbacks[0][1]["workToken"] == item["workToken"]
        finally:
            release.set()
            await settled(adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,decision", [("yes", "once"), ("no", "deny")])
async def test_leased_plaintext_reply_resolves_real_approval_before_fifo(reply, decision):
    from gateway.config import Platform
    from tests.e2e.conftest import make_runner
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    runner = make_runner(Platform.THECHAT)
    started, release = asyncio.Event(), asyncio.Event()
    first, ordinary, answer, conversational = [platform_event(token=str(uuid.uuid4())) for _ in range(4)]
    ordinary["text"] = "Keep this ordinary follow-up verbatim"
    answer["text"] = conversational["text"] = reply
    entry = _ApprovalEntry({"command": "synthetic command; never executed"})
    seen = []
    keys = []
    async with connected_adapter() as (adapter, backend):
        adapter.gateway_runner = runner
        runner.adapters[Platform.THECHAT] = adapter
        adapter.set_message_handler(runner._handle_message)
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

        async def turn(event, source, key, generation):
            seen.append((event.message_id, event.text))
            if event.message_id == first["messageId"]:
                keys.append(key)
                approval._gateway_queues[key] = [entry]
                started.set()
                await release.wait()
            return event.text

        runner._handle_message_with_agent = turn
        try:
            await adapter._handle_platform_event(first)
            await asyncio.wait_for(started.wait(), 5)
            first_lease = adapter._event_contexts[first["messageId"]]["work_lease"]
            await adapter._handle_platform_event(ordinary)
            assert not entry.event.is_set()
            await adapter._handle_platform_event(answer)
            assert entry.event.is_set(), "plain reply was queued behind its own blocking approval"
            assert entry.result == decision
            assert backend.terminal_invocations.get(answer["invocationId"]) == "completed"
            assert not first_lease.closed and not first_lease.run_task.done()
            assert first["invocationId"] in backend.active_invocations
            assert ordinary["invocationId"] in backend.active_invocations
            assert not approval.has_blocking_approval(keys[0])
            # Identical bare text with no pending approval is a normal new turn.
            await adapter._handle_platform_event(conversational)
            assert seen == [(first["messageId"], first["text"])]
            release.set()
            await settled(adapter)
            assert seen == [(item["messageId"], item["text"]) for item in (first, ordinary, conversational)]
            for item in (first, ordinary, answer, conversational):
                assert backend.terminal_invocations[item["invocationId"]] == "completed"
                callbacks = [(path, body) for path, body in backend.calls
                             if path.endswith(f"/{item['invocationId']}/completed")]
                assert len(callbacks) == 1
                assert callbacks[0][1]["workToken"] == item["workToken"]
                messages = [body for path, body in backend.calls if path.endswith("/messages")
                            and body.get("invocationId") == item["invocationId"]]
                assert messages and all(body["workToken"] == item["workToken"] for body in messages)
            assert not backend.active_invocations
        finally:
            release.set()
            await settled(adapter)
            for key in keys:
                approval._gateway_queues.pop(key, None)


@pytest.mark.asyncio
async def test_poll_preserves_session_order_across_blocked_attachment_with_live_sibling():
    backend = LoopbackBackend()
    backend.lease_seconds = 0.9
    first, second = [platform_event(token=str(uuid.uuid4())) for _ in range(2)]
    sibling = platform_event(thread="independent", token=str(uuid.uuid4()))
    first["attachments"] = [{"synthetic": "blocked attachment"}]
    backend.events = [first, second, sibling]
    download_started, release_download = asyncio.Event(), asyncio.Event()
    sibling_started, first_started, release_turns = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen = []
    async with connected_adapter(backend) as (adapter, _):
        async def download(attachments):
            if attachments:
                download_started.set()
                await release_download.wait()
            return [], [], []

        async def agent(event):
            seen.append(event.message_id)
            if event.message_id == sibling["messageId"]:
                sibling_started.set()
            if event.message_id == first["messageId"]:
                first_started.set()
            await release_turns.wait()
            return event.message_id

        adapter._download_inbound_attachments = download
        adapter.set_message_handler(agent)
        adapter._running = True
        poll = asyncio.create_task(adapter._poll_loop())
        try:
            await asyncio.wait_for(download_started.wait(), 5)
            await asyncio.wait_for(sibling_started.wait(), 5)
            # Every claimed epoch receives its initial ACK and keeps renewing
            # while the first session's attachment is still behind the barrier.
            counts = {item["invocationId"]: 0 for item in (first, second, sibling)}
            while min(counts.values()) < 3:
                invocation = await asyncio.wait_for(backend.heartbeat_invocations.get(), 5)
                counts[invocation] += 1
            assert not release_download.is_set()
            assert seen == [sibling["messageId"]], "later same-session text overtook the attachment"
            release_download.set()
            await asyncio.wait_for(first_started.wait(), 5)
            release_turns.set()
            await settled(adapter)
            assert [message for message in seen if message != sibling["messageId"]] == [first["messageId"], second["messageId"]]
            for item in (first, second, sibling):
                assert backend.terminal_invocations[item["invocationId"]] == "completed"
            assert not backend.active_invocations
        finally:
            release_download.set()
            release_turns.set()
            adapter._running = False
            poll.cancel()
            await asyncio.gather(poll, return_exceptions=True)
            await settled(adapter)
