"""Application acknowledgements and mixed-version receiver safety."""

import json
import httpx
import pytest
from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter


@pytest.mark.asyncio
async def test_non_boolean_ack_never_clears_durable_pending(tmp_path):
    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
    )
    requests = []

    def response(request):
        data = json.loads(request.content)
        requests.append(data)
        return httpx.Response(
            200, json={"ok": 1, "taskId": data["taskId"], "sequence": data["sequence"]}
        )

    a._client = httpx.AsyncClient(
        base_url=a.base_url, transport=httpx.MockTransport(response)
    )
    try:
        original = turn.snapshot()
        await a.publish_activity_once()
        assert turn.store.pending() == [original], (
            "Only a schema-valid application ack is delivery"
        )
        await a.publish_activity_once()
        assert requests == [original, original]
    finally:
        await a._client.aclose()


@pytest.mark.asyncio
async def test_owner_loop_backs_off_an_old_receiver_without_fake_ack(tmp_path):
    import asyncio
    from aiohttp import web
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    first, second = asyncio.Event(), asyncio.Event()
    requests = []

    async def receive(request):
        requests.append(await request.json())
        (first if len(requests) == 1 else second).set()
        return web.json_response({"error": "old receiver"}, status=404)

    app = web.Application()
    app.router.add_post("/hermes-platform/activity", receive)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    a = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="isolated",
            extra={
                "base_url": f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
            },
        )
    )
    a._client = httpx.AsyncClient(base_url=a.base_url)
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
    )
    gateway = object.__new__(GatewayRunner)
    gateway._running = True
    gateway.adapters = {Platform.THECHAT: a}
    gateway._profile_adapters = {}
    gateway._served_profile_homes = {"default": tmp_path}
    watcher = asyncio.create_task(
        gateway._background_activity_watcher(interval=0.01, heartbeat_seconds=0.05)
    )
    try:
        await asyncio.wait_for(first.wait(), 5)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second.wait(), 2)
        assert turn.store.pending(), (
            "Unsupported endpoints are never delivered receipts"
        )
        assert turn.snapshot()["sequence"] > requests[0]["sequence"]
    finally:
        gateway._running = False
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await a._client.aclose()
        await server.cleanup()


@pytest.mark.asyncio
async def test_stop_profile_boundary_survives_identical_session_keys(tmp_path):
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from unittest.mock import AsyncMock

    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    first = a.begin_activity_turn(
        home=tmp_path / "A",
        session_key="owned",
        session_id="same-session",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
    )
    other = a.begin_activity_turn(
        home=tmp_path / "B",
        session_key="owned",
        session_id="same-session",
        origin_invocation_id="22222222-2222-4222-8222-222222222222",
    )
    first.event("subagent.start", subagent_id="A-worker")
    other.event("subagent.start", subagent_id="B-worker")
    first.finish()
    other.finish()
    before = other.snapshot()
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_key="owned", session_id="same-session")
        ),
    )
    runner._running_agents = {}
    runner._delivery_adapter_for = lambda source: a
    runner._resolve_profile_home_for_source = lambda source: tmp_path / "A"
    await runner._handle_stop_command(MessageEvent(text="/stop", source=source))
    assert other.snapshot() == before, (
        "Stop must bind owning home even if a session key collides"
    )
    assert first.snapshot()["parent"]["action"] == "Stop requested"


@pytest.mark.asyncio
async def test_cancelled_parent_receipt_waits_for_owned_child_exit(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent, ProcessingOutcome

    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    key = a._source_session_key(source)
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
        event_message_id="parent",
    )
    turn.event("subagent.start", subagent_id="still-exiting")
    await a.interrupt_session_activity(key, source.chat_id, profile_home=tmp_path)
    turn.parent_returned(stopped=True)
    await a.on_processing_complete(
        MessageEvent(text="parent", source=source, message_id="parent"),
        ProcessingOutcome.CANCELLED,
    )
    assert turn.snapshot()["parent"]["state"] == "waiting_worker", (
        "Parent cancellation is not proof that detached children exited"
    )
    turn.event("subagent.complete", subagent_id="still-exiting", status="interrupted")
    assert turn.snapshot()["parent"]["state"] == "stopped"


@pytest.mark.asyncio
async def test_actual_detached_stop_does_not_cross_profile_collision(tmp_path):
    import threading
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from tools import async_delegation as ad
    from tools.process_registry import process_registry
    from agent.secret_scope import set_multiplex_active

    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        home.mkdir()
    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    interrupted = [threading.Event(), threading.Event()]
    ready = [threading.Event(), threading.Event()]
    release = threading.Event()
    handles = []
    ad._reset_for_tests()
    set_multiplex_active(True)
    try:
        for index, home in enumerate(homes):
            with _profile_runtime_scope(home, {}):

                def child(index=index):
                    ready[index].set()
                    release.wait(10)
                    return {"status": "completed", "summary": "offline scoped unit"}

                handle = ad.dispatch_async_delegation(
                    goal="scoped unit",
                    context=None,
                    toolsets=None,
                    role="leaf",
                    model="offline",
                    session_key="owned",
                    parent_session_id="same-session",
                    runner=child,
                    interrupt_fn=interrupted[index].set,
                )
                handles.append(handle["delegation_id"])
                turn = a.begin_activity_turn(
                    home=home,
                    session_key="owned",
                    session_id="same-session",
                    origin_invocation_id=f"{index + 1:08d}-1111-4111-8111-111111111111",
                )
                turn.event(
                    "subagent.start",
                    subagent_id=f"worker-{index}",
                    delegation_id=handle["delegation_id"],
                )
                turn.finish()
        assert all(event.wait(5) for event in ready)
        source = SessionSource(
            platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
        )
        runner = object.__new__(GatewayRunner)
        runner.session_store = SimpleNamespace()
        runner._async_session_store = SimpleNamespace(
            _store=runner.session_store,
            get_or_create_session=AsyncMock(
                return_value=SimpleNamespace(
                    session_key="owned", session_id="same-session"
                )
            ),
        )
        runner._running_agents = {}
        runner._delivery_adapter_for = lambda source: a
        runner._resolve_profile_home_for_source = lambda source: homes[0]
        with _profile_runtime_scope(homes[0], {}):
            await runner._handle_stop_command(MessageEvent(text="/stop", source=source))
        assert interrupted[0].is_set()
        assert not interrupted[1].is_set(), (
            "Real detached child selection must include owning profile, not only session IDs"
        )
    finally:
        release.set()
        received = set()
        while len(received) < len(handles):
            event = process_registry.completion_queue.get(timeout=10)
            if event.get("delegation_id") in handles:
                received.add(event["delegation_id"])
        ad._reset_for_tests()
        set_multiplex_active(False)
