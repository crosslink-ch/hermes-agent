"""Task-bound server control cannot stop a newer goal after session rotation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from tools.process_activity import bind_process_activity, observe_process_start
from tools.process_registry import process_registry
from gateway.session import SessionSource


def test_retention_prunes_immutable_control_routes_with_their_task(
    tmp_path, monkeypatch
):
    from gateway import background_activity as activity

    monkeypatch.setattr(activity, "MAX_TERMINAL_TASKS", 1)
    store = activity.ActivityStore(tmp_path, "control-transport")
    for origin in ("old", "new"):
        turn = store.begin(
            "owned-conversation", "same-session", origin=str(uuid.uuid4())
        )
        store.bind_route(turn.task_id, str(uuid.uuid4()))
        turn.finish()
    store.prune()
    assert store.db.execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 1, (
        "Control metadata must share bounded task retention"
    )


@pytest.mark.asyncio
async def test_owner_advertises_task_control_only_on_authenticated_command_registration():
    import httpx
    import json

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="isolated-control",
            extra={"base_url": "http://127.0.0.1"},
        )
    )
    received = []

    def receiver(request):
        received.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    adapter._client = httpx.AsyncClient(
        base_url=adapter.base_url, transport=httpx.MockTransport(receiver)
    )
    try:
        await adapter._register_commands()
        assert received[0][0] == "/bots/me/commands"
        assert received[0][1].get("backgroundTaskControlVersion") == 1, (
            "Receiver must not dispatch task control to legacy owners"
        )
    finally:
        await adapter._client.aclose()


@pytest.mark.asyncio
async def test_task_stop_interrupts_only_owned_parent_and_detached_delegation(
    tmp_path, monkeypatch
):
    import threading
    from run_agent import AIAgent
    from tools import async_delegation as ad
    from gateway.background_activity_control import stop_bound_task
    from unittest.mock import patch

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    chat = str(uuid.uuid4())
    key = f"agent:main:thechat:dm:{chat}"
    config = PlatformConfig(
        enabled=True, token="isolated-control", extra={"base_url": "http://127.0.0.1"}
    )
    adapter = TheChatAdapter(config)
    gateway = GatewayRunner(config=GatewayConfig(platforms={Platform.THECHAT: config}))
    gateway.adapters = {Platform.THECHAT: adapter}
    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id=chat,
        chat_type="dm",
        user_id="isolated-human",
    )
    turns, handles = [], []
    ready = [threading.Event(), threading.Event()]
    stop = [threading.Event(), threading.Event()]
    start, release = threading.Event(), threading.Event()
    ad._reset_for_tests()
    with patch("agent.process_bootstrap.OpenAI"):
        parent = AIAgent(
            api_key="isolated",
            model="offline",
            base_url="http://127.0.0.1:1",
            enabled_toolsets=["terminal"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="same-session",
        )
    try:
        for index in range(2):
            adapter._event_contexts[str(index)] = {
                "conversation_id": chat,
                "thread_id": None,
            }
            turn = adapter.begin_activity_turn(
                home=tmp_path,
                session_key=key,
                session_id="same-session",
                origin_invocation_id=str(uuid.uuid4()),
                event_message_id=str(index),
            )
            turns.append(turn)

            def child(index=index, turn=turn):
                assert start.wait(10)
                turn.event(
                    "subagent.start",
                    subagent_id=f"child-{index}",
                    delegation_id=handles[index]["delegation_id"],
                )
                ready[index].set()
                release.wait(10)
                status = "interrupted" if stop[index].is_set() else "completed"
                turn.event(
                    "subagent.complete", subagent_id=f"child-{index}", status=status
                )
                return {"status": status, "summary": "Offline scoped child"}

            handles.append(
                ad.dispatch_async_delegation(
                    goal="offline scoped unit",
                    context=None,
                    toolsets=None,
                    role="leaf",
                    model="offline",
                    session_key=key,
                    parent_session_id="same-session",
                    runner=child,
                    interrupt_fn=stop[index].set,
                )
            )
        start.set()
        assert all(
            await asyncio.gather(*(asyncio.to_thread(event.wait, 5) for event in ready))
        )
        turns[1].finish()
        parent._background_activity = turns[0]
        gateway._running_agents[key] = parent
        await stop_bound_task(
            gateway,
            source,
            {
                "version": 1,
                "action": "stop",
                "taskId": turns[0].task_id,
                "sessionId": "same-session",
            },
        )
        assert parent._interrupt_requested, (
            "Task-wide Stop includes its actual native parent"
        )
        assert stop[0].is_set(), "Detached unit must be selected by task binding"
        assert not stop[1].is_set(), "Same-key unrelated goal must not be interrupted"
        turns[0].parent_returned(stopped=True)
        assert turns[0].snapshot()["parent"]["state"] == "waiting_worker"
    finally:
        start.set()
        release.set()
        parent.close()
        from tools.process_registry import process_registry

        observed = set()
        while len(observed) < len(handles):
            event = await asyncio.to_thread(
                process_registry.completion_queue.get, True, 10
            )
            if event.get("delegation_id") in {
                handle["delegation_id"] for handle in handles
            }:
                observed.add(event["delegation_id"])
        ad._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_signed_control_stops_original_process_not_newer_same_conversation(
    tmp_path, monkeypatch, busy
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    chat, actor, bot_id, bot_user = (str(uuid.uuid4()) for _ in range(4))
    key = f"agent:main:thechat:dm:{chat}"
    config = PlatformConfig(
        enabled=True, token="isolated-control", extra={"base_url": "http://127.0.0.1"}
    )
    adapter = TheChatAdapter(config)
    gateway = GatewayRunner(config=GatewayConfig(platforms={Platform.THECHAT: config}))
    gateway.adapters = {Platform.THECHAT: adapter}
    gateway._hm_admit_event = AsyncMock(
        side_effect=lambda event: (event, event.source, False)
    )
    gateway._hm_estop_gate = lambda *args: None
    gateway._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    gateway._hm_evict_idle_stale_agent = MagicMock()
    gateway._hm_evict_reaped_agent = MagicMock()
    gateway._is_session_running = lambda *args: True
    gateway._hm_handle_running_session_message = AsyncMock(
        return_value="Legacy conversation stop"
    )
    adapter.set_message_handler(gateway._handle_message)
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="control-reply")
    )
    adapter._event_contexts["old"] = {"conversation_id": chat, "thread_id": None}
    old = adapter.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="old-session",
        origin_invocation_id=str(uuid.uuid4()),
        event_message_id="old",
    )
    bind_process_activity(
        lambda event, name=None, preview=None, args=None, **kw: old.event(
            event, name, **kw
        )
    )
    owned = process_registry.spawn_local(
        command="sleep 30", task_id=key, session_key=key
    )
    observe_process_start(process_registry, owned)
    old.finish()
    adapter._event_contexts["new"] = {"conversation_id": chat, "thread_id": None}
    newer = adapter.begin_activity_turn(
        home=tmp_path,
        session_key=key,
        session_id="new-session",
        origin_invocation_id=str(uuid.uuid4()),
        event_message_id="new",
    )
    bind_process_activity(
        lambda event, name=None, preview=None, args=None, **kw: newer.event(
            event, name, **kw
        )
    )
    other = process_registry.spawn_local(
        command="sleep 30", task_id=key, session_key=key
    )
    observe_process_start(process_registry, other)
    parent = SimpleNamespace(_background_activity=newer, interrupt=MagicMock())
    gateway._running_agents[key] = parent
    control_id = str(uuid.uuid4())
    item = {
        "id": control_id,
        "invocationId": control_id,
        "chatId": chat,
        "chatType": "dm",
        "threadId": None,
        "text": "/stop",
        "attachments": [],
        "messageId": str(uuid.uuid4()),
        "sender": {"id": actor, "name": "Isolated requester"},
        "bot": {"id": bot_id, "userId": bot_user, "name": "Isolated bot"},
        "conversation": {
            "id": chat,
            "type": "direct",
            "name": None,
            "workspaceId": None,
        },
        "taskControl": {
            "version": 1,
            "action": "stop",
            "taskId": old.task_id,
            "sessionId": "old-session",
        },
    }
    blocker = None
    guard = asyncio.Event()
    if busy:
        blocker = asyncio.create_task(asyncio.Event().wait())
        adapter._active_sessions[key] = guard
        adapter._session_tasks[key] = blocker
    try:
        await adapter._handle_platform_event(item)
        if busy:
            assert not blocker.done(), (
                "Task control must not cancel newer adapter processing"
            )
            assert adapter._active_sessions.get(key) is guard
        else:
            task = adapter._session_tasks.get(key)
            if task is not None:
                await asyncio.wait_for(asyncio.shield(task), 15)
        assert await asyncio.to_thread(owned._completion_event.wait, 5), (
            "Task control must reach the original idle parent's process"
        )
        assert old.snapshot()["parent"]["state"] == "stopped"
        assert not other.exited
        assert newer.snapshot()["workers"][0]["state"] == "running"
        parent.interrupt.assert_not_called()
        gateway._hm_handle_running_session_message.assert_not_called()
    finally:
        if blocker is not None:
            blocker.cancel()
            await asyncio.gather(blocker, return_exceptions=True)
        for proc in (owned, other):
            process_registry.kill_process(
                proc.id, source="isolated_control_cleanup", consume_output=True
            )
        bind_process_activity(None)
