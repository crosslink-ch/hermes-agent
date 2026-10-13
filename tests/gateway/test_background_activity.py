"""Messaging-owner activity: real adapter, callback and durable state paths."""

import json
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.run_turn_runner import TurnRunner

ORIGIN = "11111111-1111-4111-8111-111111111111"


def adapter():
    return TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="isolated-test-bot",
            extra={"base_url": "http://127.0.0.1"},
        )
    )


def context(activity):
    return SimpleNamespace(
        _background_activity=activity,
        _structured_progress_supported=False,
        _run_still_current=lambda: False,
    )


def test_parent_reply_keeps_real_child_callback_alive(tmp_path):
    a = adapter()
    assert callable(getattr(a, "begin_activity_turn", None)), (
        "Messaging owner needs an independent durable activity lane"
    )
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    runner = TurnRunner(None, context(turn))
    runner._record_structured_tool_progress(
        "subagent.start",
        subagent_id="child-A",
        goal="SECRET goal",
        summary="SECRET result",
    )
    turn.finish()
    snapshot = turn.snapshot()
    assert snapshot["parent"]["state"] == "waiting_worker"
    assert snapshot["workers"][0]["state"] == "running"
    runner._record_structured_tool_progress(
        "subagent.tool",
        "terminal",
        "SECRET command",
        subagent_id="child-A",
        tool_call_id="quiet-call",
    )
    assert turn.snapshot()["activities"][0]["state"] == "running"
    runner._record_structured_tool_progress(
        "subagent.complete",
        subagent_id="child-A",
        status="completed",
        summary="SECRET result",
    )
    snapshot = turn.snapshot()
    assert snapshot["parent"]["state"] == "verifying"
    assert snapshot["workers"][0]["state"] == "finished"
    assert "SECRET" not in json.dumps(snapshot)
    followup = a.begin_activity_turn(
        home=tmp_path, session_key="owned", session_id="session-A", verification=True
    )
    assert followup.task_id == turn.task_id
    assert followup.execution_id != turn.execution_id
    followup.finish()
    assert followup.snapshot()["parent"]["state"] == "finished"
    assert followup.snapshot()["originInvocationId"] == ORIGIN


@pytest.mark.asyncio
async def test_durable_retry_full_snapshot_ack_heartbeat_and_restart(tmp_path):
    from aiohttp import web
    import httpx

    received = []
    responses = [503, 404, 200]

    async def receive(request):
        assert request.headers["Authorization"] == "Bearer isolated-test-bot"
        data = await request.json()
        received.append(data)
        status = responses.pop(0) if responses else 200
        return web.json_response(
            {"ok": True, "taskId": data["taskId"], "sequence": data["sequence"]},
            status=status,
        )

    app = web.Application()
    app.router.add_post("/hermes-platform/activity", receive)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    a = adapter()
    a.base_url = f"http://127.0.0.1:{port}"
    a._client = httpx.AsyncClient(
        base_url=a.base_url, headers={"Authorization": "Bearer isolated-test-bot"}
    )
    try:
        turn = a.begin_activity_turn(
            home=tmp_path,
            session_key="owned",
            session_id="session-A",
            origin_invocation_id=ORIGIN,
        )
        turn.event("subagent.start", subagent_id="child-A")
        turn.event(
            "subagent.tool", "terminal", subagent_id="child-A", tool_call_id="quiet"
        )
        turn.finish()
        assert callable(getattr(a, "publish_activity_once", None)), (
            "Full snapshots need durable delivery and application ack"
        )
        before = turn.snapshot()
        await a.publish_activity_once()
        await a.publish_activity_once()
        assert turn.store.pending()[0] == before
        await a.publish_activity_once()
        assert received == [before, before, before]
        assert turn.store.pending() == []
        turn.store.heartbeat()
        heartbeat = turn.snapshot()
        assert heartbeat["sequence"] > before["sequence"]
        assert (
            heartbeat["workers"][0]["lastActionAt"]
            == before["workers"][0]["lastActionAt"]
        )
        assert heartbeat["activities"][0]["state"] == "running"
        replacement = adapter()
        replacement.base_url = a.base_url
        replacement.begin_activity_turn(
            home=tmp_path, session_key="no-binding", session_id="none"
        )
        recovered = next(iter(replacement._activity_stores.values())).snapshot(
            turn.task_id
        )
        assert recovered["parent"]["state"] == "unknown"
        assert recovered["workers"][0]["state"] == "unknown"
        assert recovered["activities"][0]["state"] == "unknown"
        assert recovered["sequence"] > heartbeat["sequence"]
        assert (
            next(iter(replacement._activity_stores.values())).pending()[0] == recovered
        )
    finally:
        await a._client.aclose()
        await server.cleanup()


@pytest.mark.asyncio
async def test_gateway_binds_original_invocation_and_real_child_relay(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from tools.delegate_tool_progress import _build_child_progress_callback

    a = adapter()
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="chat",
        chat_type="dm",
        user_id="u",
        message_id="message",
    )
    a._event_contexts["message"] = {"invocation_id": ORIGIN}
    gateway = object.__new__(GatewayRunner)
    gateway.hooks = SimpleNamespace(loaded_hooks=False)
    gateway._delivery_adapter_for = lambda source: a
    gateway._resolve_profile_home_for_source = lambda source: tmp_path
    gateway._run_agent_progress_threading = lambda *args: (
        None,
        None,
        {"message_id": "message"},
    )
    ctx = TurnContext(
        source=source,
        session_key="owned",
        session_id="session-A",
        _run_still_current=lambda: False,
    )
    runner = TurnRunner(gateway, ctx)
    gateway._run_agent_bind_turn_wiring(ctx, runner, source, "message", False)
    assert getattr(ctx, "_background_activity", None) is not None, (
        "Real gateway wiring must create task independently of progress"
    )
    parent = SimpleNamespace(tool_progress_callback=runner.progress_callback)
    relay = _build_child_progress_callback(
        0, "SECRET goal", parent, subagent_id="child-A"
    )
    relay("subagent.start")
    ctx._background_activity.finish()
    relay(
        "tool.started",
        "terminal",
        "SECRET command",
        {"command": "SECRET"},
        tool_call_id="call-A",
    )
    relay(
        "tool.completed",
        "terminal",
        tool_call_id="call-A",
        is_error=False,
        result="SECRET result",
    )
    snapshot = ctx._background_activity.snapshot()
    assert snapshot["activities"][0]["state"] == "finished"
    assert snapshot["activities"][0]["workerId"] == "child-A"
    relay("subagent.complete", status="completed")
    assert ctx._background_activity.snapshot()["parent"]["state"] == "verifying"
    assert "SECRET" not in json.dumps(ctx._background_activity.snapshot())
    follow_ctx = TurnContext(
        source=source,
        session_key="owned",
        session_id="session-A",
        persist_user_display_kind="async_delegation_complete",
    )
    a._event_contexts.clear()
    follow_runner = TurnRunner(gateway, follow_ctx)
    gateway._run_agent_bind_turn_wiring(follow_ctx, follow_runner, source, None, False)
    assert follow_ctx._background_activity.task_id == ctx._background_activity.task_id
    assert (
        follow_ctx._background_activity.execution_id
        != ctx._background_activity.execution_id
    )


def test_real_agent_tool_executor_preserves_call_identity(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from agent.tool_executor import execute_tool_calls_sequential
    from tests.agent.test_run_agent import _mock_tool_call, _mock_assistant_msg
    from unittest.mock import patch

    a = adapter()
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    runner = TurnRunner(None, context(turn))
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="isolated-test",
            model="offline-test",
            base_url="http://127.0.0.1:1",
            enabled_toolsets=["terminal"],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            session_id="session-A",
        )
    agent.tool_progress_callback = (
        lambda event, name=None, preview=None, args=None, **kw: (
            runner._record_structured_tool_progress(event, name, preview, **kw)
        )
    )
    tool = _mock_tool_call(
        "terminal",
        json.dumps({"command": "printf 'SECRET'", "timeout": 10}),
        "actual-call",
    )
    msg = _mock_assistant_msg(tool_calls=[tool])
    messages = [
        {"role": "user", "content": "offline tool acceptance"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "actual-call",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": tool.function.arguments,
                    },
                }
            ],
        },
    ]
    try:
        execute_tool_calls_sequential(agent, msg, messages, "owned", finalize=False)
        assert "SECRET" in str(messages[-1]["content"]), (
            "Real native tool must actually execute"
        )
        assert turn.snapshot()["activities"], (
            "Real executor must supply call identity to durable callback"
        )
        assert turn.snapshot()["activities"][0]["state"] == "finished"
        assert "SECRET" not in json.dumps(turn.snapshot())
    finally:
        agent.close()


def test_real_background_process_lifecycle_survives_parent_reply(tmp_path, monkeypatch):
    from run_agent import AIAgent
    from agent.tool_executor import execute_tool_calls_sequential
    from tools.process_registry import process_registry
    from tests.agent.test_run_agent import _mock_tool_call, _mock_assistant_msg
    from unittest.mock import patch

    gate = tmp_path / "release"
    a = adapter()
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    runner = TurnRunner(None, context(turn))
    with patch("agent.process_bootstrap.OpenAI"):
        agent = AIAgent(
            api_key="isolated-test",
            model="offline-test",
            base_url="http://127.0.0.1:1",
            enabled_toolsets=["terminal"],
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            session_id="session-A",
        )
    agent.tool_progress_callback = (
        lambda event, name=None, preview=None, args=None, **kw: (
            runner._record_structured_tool_progress(event, name, preview, **kw)
        )
    )
    tool = _mock_tool_call(
        "terminal",
        json.dumps({
            "command": f"while [ ! -f '{gate}' ]; do sleep 0.02; done; printf 'SECRET'",
            "background": True,
        }),
        "spawn-call",
    )
    messages = []
    process = None
    try:
        execute_tool_calls_sequential(
            agent,
            _mock_assistant_msg(tool_calls=[tool]),
            messages,
            "owned",
            finalize=False,
        )
        result = json.loads(messages[-1]["content"])
        process = process_registry.get(result["session_id"])
        assert process is not None and not process.exited
        assert turn.snapshot()["workers"], (
            "Actual background process needs an owner callback"
        )
        turn.finish()
        assert turn.snapshot()["parent"]["state"] == "waiting_worker"
        assert turn.snapshot()["workers"][0]["kind"] == "process"
        gate.touch()
        assert process._completion_event.wait(5)
        assert turn.snapshot()["workers"][0]["state"] == "finished"
        assert turn.snapshot()["parent"]["state"] == "verifying"
        assert "SECRET" not in json.dumps(turn.snapshot())
    finally:
        gate.touch()
        if process is not None:
            process_registry.kill_process(
                process.id, source="test_cleanup", consume_output=True
            )
        agent.close()


@pytest.mark.asyncio
async def test_actual_gateway_idle_watcher_publishes_and_heartbeats(tmp_path):
    import asyncio
    import httpx
    from aiohttp import web
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    received = []
    observed = asyncio.Event()

    async def receive(request):
        data = await request.json()
        received.append(data)
        return web.json_response({
            "ok": True,
            "taskId": data["taskId"],
            "sequence": data["sequence"],
        })

    app = web.Application()
    app.router.add_post("/hermes-platform/activity", receive)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    a = adapter()
    a.base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    a._client = httpx.AsyncClient(
        base_url=a.base_url, headers={"Authorization": "Bearer isolated-test-bot"}
    )
    gateway = object.__new__(GatewayRunner)
    gateway.adapters = {Platform.THECHAT: a}
    gateway._profile_adapters = {}
    gateway._served_profile_homes = {"default": tmp_path}
    gateway._running = True
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    turn.event("subagent.start", subagent_id="actual-child")
    turn.finish()
    acknowledge = turn.store.acknowledge

    def after_ack(snapshot):
        acknowledge(snapshot)
        if len(received) >= 2:
            observed.set()

    turn.store.acknowledge = after_ack
    watcher = None
    try:
        assert callable(getattr(gateway, "_background_activity_watcher", None)), (
            "Owning messaging gateway must publish after parent turn"
        )
        watcher = asyncio.create_task(
            gateway._background_activity_watcher(interval=0.01, heartbeat_seconds=0.03)
        )
        await asyncio.wait_for(observed.wait(), 5)
        assert received[0]["parent"]["state"] == "waiting_worker"
        assert received[-1]["sequence"] > received[0]["sequence"]
        assert received[-1]["workers"][0]["lastActionAt"] is None
        assert turn.store.pending() == []
    finally:
        gateway._running = False
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await a._client.aclose()
        await server.cleanup()


@pytest.mark.asyncio
async def test_actual_gateway_turn_reply_and_verification_delivery(
    tmp_path, monkeypatch
):
    import asyncio
    import weakref
    import gateway.run as gateway_run
    import run_agent
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from gateway.platforms.base import ProcessingOutcome
    from tests.gateway.test_thechat_auto_title_path import _ImmediateAgent, _make_runner

    instances = []

    class OfflineAgent(_ImmediateAgent):
        def __init__(self, **kw):
            super().__init__(**kw)
            instances.append(self)

        def run_conversation(self, *args, **kwargs):
            if len(instances) == 1 and not getattr(self, "launched", False):
                self.launched = True
                self.tool_progress_callback("subagent.start", subagent_id="held-worker")
            return {
                "final_response": "Offline deterministic model reply",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            }

    monkeypatch.setattr(run_agent, "AIAgent", OfflineAgent)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"tool_progress": "off", "streaming": False}},
    )
    runner = _make_runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._resolve_session_agent_runtime = lambda **kw: (
        "offline",
        {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "api_key": "isolated-test",
            "base_url": "http://127.0.0.1:1",
        },
    )
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    a = adapter()
    runner.adapters = {Platform.THECHAT: a}
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="chat",
        chat_type="dm",
        user_id="u",
        message_id="message",
    )
    source._transport_adapter_ref = weakref.ref(a)
    key = a._source_session_key(source)
    a._event_contexts["message"] = {
        "invocation_id": ORIGIN,
        "conversation_id": "chat",
        "delivered": True,
    }
    result = await runner._run_agent(
        message="offline",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=key,
        event_message_id="message",
    )
    assert result["completed"] is True
    store = a._activity_stores[str(tmp_path)]
    snapshot = store.pending()[0]
    assert snapshot["parent"]["state"] == "waiting_worker", (
        "Real parent turn completion is not task completion"
    )
    event = MessageEvent(source=source, text="offline", message_id="message")
    await a.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    instances[0].tool_progress_callback(
        "subagent.complete", subagent_id="held-worker", status="completed"
    )
    assert store.snapshot(snapshot["taskId"])["parent"]["state"] == "verifying"
    result = await runner._run_agent(
        message="verification",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=key,
        persist_user_display_kind="async_delegation_complete",
    )
    assert result["completed"] is True
    assert store.snapshot(snapshot["taskId"])["parent"]["state"] == "verifying"
    followup_event = MessageEvent(
        source=source,
        text="verification",
        message_id="internal-completion",
        internal=True,
    )
    await a.on_processing_complete(followup_event, ProcessingOutcome.SUCCESS)
    assert store.snapshot(snapshot["taskId"])["parent"]["state"] == "finished"


@pytest.mark.asyncio
async def test_idle_stop_owns_processes_and_detached_children_only(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import asyncio
    import threading
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from tools import async_delegation as ad
    from tools.process_registry import process_registry
    from tools.process_activity import bind_process_activity, observe_process_start
    from tools.delegate_tool_progress import _build_child_progress_callback

    ad._reset_for_tests()
    a = adapter()
    gateway = GatewayRunner(
        config=GatewayConfig(platforms={Platform.THECHAT: a.config})
    )
    gateway.adapters = {Platform.THECHAT: a}
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned-chat", chat_type="dm", user_id="u"
    )
    entry = await gateway.async_session_store.get_or_create_session(source)
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key=entry.session_key,
        session_id=entry.session_id,
        origin_invocation_id=ORIGIN,
    )
    runner = TurnRunner(gateway, context(turn))
    callback = lambda event, name=None, preview=None, args=None, **kw: (
        runner._record_structured_tool_progress(event, name, preview, **kw)
    )
    relay = _build_child_progress_callback(
        0,
        "SECRET",
        SimpleNamespace(tool_progress_callback=callback),
        subagent_id="detached-child",
    )
    started, requested, release, finished = (threading.Event() for _ in range(4))

    def child():
        relay("subagent.start")
        started.set()
        release.wait(10)
        relay("subagent.complete", status="interrupted")
        finished.set()
        return {"status": "interrupted", "summary": "offline controlled child"}

    handle = ad.dispatch_async_delegation(
        goal="offline child",
        context=None,
        toolsets=None,
        role="leaf",
        model="offline",
        session_key=entry.session_key,
        parent_session_id=entry.session_id,
        runner=child,
        interrupt_fn=requested.set,
    )
    assert handle["status"] == "dispatched"
    assert await asyncio.to_thread(started.wait, 5)
    bind_process_activity(callback)
    owned = process_registry.spawn_local(
        command="sleep 30", task_id=entry.session_key, session_key=entry.session_key
    )
    observe_process_start(process_registry, owned)
    other = process_registry.spawn_local(
        command="sleep 30", task_id="other-task", session_key="other-session"
    )
    turn.finish()
    try:
        reply = await gateway._handle_stop_command(
            MessageEvent(text="/stop", source=source)
        )
        assert requested.is_set(), (
            "Actual idle stop must reach detached-unit interrupt_for_session"
        )
        assert owned._completion_event.wait(5), (
            "Idle task stop must kill its owned process"
        )
        assert owned.completion_reason == "killed"
        assert not other.exited, "Other session's process must survive"
        assert turn.snapshot()["parent"]["state"] != "stopped", (
            "Request acknowledgement must not fabricate worker completion"
        )
        assert (
            a.begin_activity_turn(
                home=tmp_path,
                session_key=entry.session_key,
                session_id=entry.session_id,
                verification=True,
                task_ids=(turn.task_id,),
            )
            is None
        ), "A stop completion must not start a new execution of the stopped goal"
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        assert turn.snapshot()["parent"]["state"] == "stopped"
        assert all(w["state"] == "stopped" for w in turn.snapshot()["workers"])
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 5)
        process_registry.kill_process(
            owned.id, source="test_cleanup", consume_output=True
        )
        process_registry.kill_process(
            other.id, source="test_cleanup", consume_output=True
        )
        bind_process_activity(None)
        ad._reset_for_tests()


def test_profile_scoped_nested_process_callback_keeps_original_binding(
    tmp_path, monkeypatch
):
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from tools.process_registry import process_registry
    from tools.process_activity import bind_process_activity, observe_process_start
    from tools.delegate_tool_progress import _build_child_progress_callback

    homes = [tmp_path / "profile-A", tmp_path / "profile-B"]
    for home in homes:
        home.mkdir()
    set_multiplex_active(True)
    a = adapter()
    processes = []
    try:
        with _profile_runtime_scope(homes[0], {}):
            assert get_hermes_home() == homes[0]
            first = a.begin_activity_turn(
                home=get_hermes_home(),
                session_key="profile-A:owned",
                session_id="session-A",
                origin_invocation_id=ORIGIN,
            )
            callback = lambda event, name=None, preview=None, args=None, **kw: (
                first.event(event, name, **kw)
            )
            relay = _build_child_progress_callback(
                0,
                "SECRET goal",
                SimpleNamespace(tool_progress_callback=callback),
                subagent_id="child-A",
            )
            relay("subagent.start")
            bind_process_activity(relay)
            process = process_registry.spawn_local(
                command="sleep 30", task_id="child-A", session_key="profile-A:owned"
            )
            processes.append(process)
            observe_process_start(process_registry, process)
            first.finish()
        with _profile_runtime_scope(homes[1], {}):
            second = a.begin_activity_turn(
                home=get_hermes_home(),
                session_key="profile-B:owned",
                session_id="session-B",
                origin_invocation_id="22222222-2222-4222-8222-222222222222",
            )
            before = second.snapshot()
            relay("tool.started", "read_file", "SECRET path", tool_call_id="quiet-A")
            process_registry.kill_process(
                process.id, source="profile_scope_test", consume_output=True
            )
            assert process._completion_event.wait(5)
            assert second.snapshot() == before
        with _profile_runtime_scope(homes[0], {}):
            snapshot = first.snapshot()
            assert snapshot["originInvocationId"] == ORIGIN
            workers = {worker["id"]: worker for worker in snapshot["workers"]}
            assert process.id in workers, (
                "Nested real process must relay to owning task, not disappear"
            )
            assert workers[process.id]["parentId"] == "child-A"
            assert workers[process.id]["state"] == "stopped"
            assert first.store.home == homes[0]
            assert second.store.home == homes[1]
            assert "SECRET" not in json.dumps(snapshot)
    finally:
        for process in processes:
            process_registry.kill_process(
                process.id, source="test_cleanup", consume_output=True
            )
        bind_process_activity(None)
        set_multiplex_active(False)


def test_old_parent_execution_cannot_mutate_new_verification(tmp_path):
    a = adapter()
    first = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    first.event("subagent.start", subagent_id="worker")
    first.finish()
    first.event("subagent.complete", subagent_id="worker", status="completed")
    followup = a.begin_activity_turn(
        home=tmp_path, session_key="owned", session_id="session-A", verification=True
    )
    before = followup.snapshot()
    first.event("tool.started", "terminal", tool_call_id="late-old-parent")
    assert followup.snapshot() == before, (
        "Root callbacks must be fenced by execution identity, not session alone"
    )
    first.parent_returned()
    first.finish()
    assert followup.snapshot() == before
    followup.event("tool.started", "read_file", tool_call_id="verify-call")
    followup.event("tool.completed", "read_file", tool_call_id="verify-call")
    followup.finish()
    terminal = followup.snapshot()
    first.event("subagent.start", subagent_id="late-new-child")
    first.event("subagent.complete", subagent_id="worker", status="failed")
    first.store.heartbeat()
    assert followup.snapshot() == terminal


@pytest.mark.asyncio
async def test_stop_command_ack_cannot_finish_an_unrelated_parent_turn(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent, ProcessingOutcome

    a = adapter()
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    key = a._source_session_key(source)
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id=ORIGIN,
        event_message_id="original",
    )
    turn.event("tool.started", "terminal", tool_call_id="quiet-parent")
    before = turn.snapshot()
    await a.on_processing_complete(
        MessageEvent(text="/stop", source=source, message_id="control"),
        ProcessingOutcome.SUCCESS,
    )
    assert turn.snapshot() == before, (
        "Command acknowledgement is not delivery of the owning parent turn"
    )
    await a.on_processing_complete(
        MessageEvent(text="parent", source=source, message_id="original"),
        ProcessingOutcome.FAILURE,
    )
    assert turn.snapshot()["parent"]["state"] == "failed"


@pytest.mark.asyncio
async def test_actual_internal_completion_verifies_original_task_not_newer_goal(
    tmp_path,
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    a = adapter()
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="same-chat", chat_type="dm", user_id="u"
    )
    key = a._source_session_key(source)
    first = a.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    first.event("subagent.start", subagent_id="child-A", delegation_id="deleg-A")
    first.finish()
    first.event("subagent.complete", subagent_id="child-A", status="completed")
    newer = a.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="session-A",
        origin_invocation_id="22222222-2222-4222-8222-222222222222",
    )
    newer.event("subagent.start", subagent_id="child-B", delegation_id="deleg-B")
    newer.finish()
    before = newer.snapshot()
    gateway = object.__new__(GatewayRunner)
    gateway._build_process_event_source = lambda evt: source
    gateway._resolve_injection_adapter = lambda *args: a
    gateway._delivery_adapter_for = lambda source: a
    gateway._resolve_profile_home_for_source = lambda source: tmp_path
    gateway._run_agent_progress_threading = lambda *args: (None, None, None)
    gateway.hooks = SimpleNamespace(loaded_hooks=False)
    received = []

    async def accept(event):
        received.append(event)
        event._gateway_accepted = True

    a.handle_message = accept
    assert (
        await gateway._inject_watch_notification(
            "SECRET child result",
            {
                "type": "async_delegation",
                "delegation_id": "deleg-A",
                "session_key": key,
                "parent_session_id": "session-A",
            },
        )
        is True
    )
    event = received[0]
    assert getattr(event, "_background_activity_task_ids", None) == (first.task_id,), (
        "Actual internal notification must carry original task binding"
    )
    ctx = TurnContext(
        source=source,
        session_key=key,
        session_id="session-A",
        persist_user_display_kind="internal_notification",
        background_activity_task_ids=event._background_activity_task_ids,
    )
    gateway._run_agent_bind_turn_wiring(
        ctx, TurnRunner(gateway, ctx), source, None, False
    )
    assert ctx._background_activity.task_id == first.task_id
    assert ctx._background_activity.execution_id != first.execution_id
    ctx._background_activity.finish()
    assert first.snapshot()["parent"]["state"] == "finished"
    assert newer.snapshot() == before


def test_retention_bounds_terminal_cache_without_dropping_live_tasks(
    tmp_path, monkeypatch
):
    import gateway.background_activity as activity

    monkeypatch.setattr(activity, "MAX_TERMINAL_TASKS", 3, raising=False)
    a = adapter()
    live = a.begin_activity_turn(
        home=tmp_path,
        session_key="live",
        session_id="live",
        origin_invocation_id=ORIGIN,
    )
    terminal = []
    for index in range(6):
        turn = a.begin_activity_turn(
            home=tmp_path,
            session_key=f"terminal-{index}",
            session_id=f"session-{index}",
            origin_invocation_id=ORIGIN,
        )
        turn.finish()
        terminal.append(turn)
    live.store.heartbeat()
    retained = [turn.snapshot() for turn in terminal if turn.snapshot() is not None]
    assert len(retained) == 3, "Derived producer history must be bounded"
    assert [snapshot["taskId"] for snapshot in retained] == [
        turn.task_id for turn in terminal[-3:]
    ]
    assert live.snapshot()["parent"]["state"] == "working"
    assert len(live.store.pending()) == 4
    for _ in range(60):
        live.event("tool.started", "read_file", tool_call_id=f"call-{_}")
        live.event("tool.completed", "read_file", tool_call_id=f"call-{_}")
    assert len(live.snapshot()["activities"]) == 40
    assert len(json.dumps(live.snapshot()).encode()) <= 65536


def test_queued_real_delegate_does_not_finish_with_parent_reply(tmp_path):
    from tools.delegate_tool_progress import _build_child_progress_callback

    a = adapter()
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    relay = _build_child_progress_callback(
        0,
        "SECRET",
        SimpleNamespace(
            tool_progress_callback=lambda event, name=None, preview=None, args=None, **kw: (
                turn.event(event, name, **kw)
            )
        ),
        subagent_id="queued-child",
    )
    relay("subagent.spawn_requested")
    turn.parent_returned()
    turn.finish()
    assert turn.snapshot()["parent"]["state"] == "waiting_worker", (
        "A dispatched child may start only after parent reply"
    )
    assert turn.snapshot()["workers"][0]["state"] == "waiting"
    relay("subagent.start")
    assert turn.snapshot()["workers"][0]["state"] == "running"
    relay("subagent.complete", status="completed")
    assert turn.snapshot()["workers"][0]["state"] == "finished"
    assert turn.snapshot()["parent"]["state"] == "verifying"


def test_muted_verification_keeps_private_activity_observer(tmp_path):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext
    from tests.gateway.test_thechat_auto_title_path import _make_runner

    a = adapter()
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    ctx = TurnContext(
        source=source,
        session_key="owned",
        session_id="session-A",
        mute_notification_reply=True,
        user_config={},
        _run_still_current=lambda: False,
    )
    ctx._background_activity = turn
    ctx._hooks_ref = SimpleNamespace(loaded_hooks=False)
    runner = TurnRunner(_make_runner(), ctx)
    ctx.progress_callback = runner.progress_callback
    agent = SimpleNamespace(tools=[], model="offline", session_id="session-A")
    runner._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert callable(agent.tool_progress_callback), (
        "Muted presentation cannot disable canonical activity"
    )
    agent.tool_progress_callback(
        "tool.started",
        "terminal",
        "SECRET",
        {"command": "SECRET"},
        tool_call_id="quiet",
    )
    agent.tool_progress_callback("tool.completed", "terminal", tool_call_id="quiet")
    assert turn.snapshot()["activities"][0]["state"] == "finished"
    agent.tool_start_callback("native-quiet", "read_file", {"path": "SECRET"})
    agent.tool_complete_callback("native-quiet", "read_file", {}, "real output")
    assert turn.snapshot()["activities"][1]["state"] == "finished"
    assert "SECRET" not in json.dumps(turn.snapshot())


def test_worker_bound_evicts_history_not_new_live_work(tmp_path):
    a = adapter()
    turn = a.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id=ORIGIN,
    )
    for index in range(32):
        turn.event("subagent.start", subagent_id=f"finished-{index}")
        turn.event(
            "subagent.complete", subagent_id=f"finished-{index}", status="completed"
        )
    turn.event("subagent.start", subagent_id="new-live-worker")
    turn.finish()
    assert turn.snapshot()["parent"]["state"] == "waiting_worker", (
        "Bounded history must not hide newly live work"
    )
    assert len(turn.snapshot()["workers"]) == 32
    assert turn.snapshot()["workers"][-1]["id"] == "new-live-worker"
    turn.event("subagent.complete", subagent_id="new-live-worker", status="completed")
    assert turn.snapshot()["parent"]["state"] == "verifying"


@pytest.mark.asyncio
async def test_binding_uses_inbound_invocation_not_reply_anchor(tmp_path):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    a = adapter()
    source = SessionSource(
        platform=Platform.THECHAT, chat_id="owned", chat_type="dm", user_id="u"
    )
    a._event_contexts["incoming"] = {"invocation_id": ORIGIN}
    a._event_contexts["historical-reply-anchor"] = {
        "invocation_id": "22222222-2222-4222-8222-222222222222"
    }
    gateway = object.__new__(GatewayRunner)
    gateway.hooks = SimpleNamespace(loaded_hooks=False)
    gateway._delivery_adapter_for = lambda source: a
    gateway._resolve_profile_home_for_source = lambda source: tmp_path
    gateway._run_agent_progress_threading = lambda *args: (None, None, None)
    ctx = TurnContext(
        source=source,
        session_key="owned",
        session_id="session-A",
        inbound_message_id="incoming",
    )
    gateway._run_agent_bind_turn_wiring(
        ctx, TurnRunner(gateway, ctx), source, "historical-reply-anchor", False
    )
    assert ctx._background_activity.snapshot()["originInvocationId"] == ORIGIN, (
        "A reply anchor must never replace admitted invocation provenance"
    )
