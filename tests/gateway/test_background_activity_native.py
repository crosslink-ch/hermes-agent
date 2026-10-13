"""Native-ID callback path (including Codex bridges) is independently durable."""

from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


def test_native_callbacks_track_quiet_tool_without_progress_text(tmp_path):
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
    ctx = TurnContext(_run_still_current=lambda: False)
    ctx._background_activity = turn
    runner = TurnRunner(None, ctx)
    runner._structured_tool_start_callback_sync(
        "native-id", "terminal", {"command": "private-command"}
    )
    assert turn.snapshot()["activities"], (
        "Native bridge must feed durable activity without text callbacks"
    )
    turn.store.heartbeat()
    assert turn.snapshot()["activities"][0]["state"] == "running"
    runner._structured_tool_complete_callback_sync(
        "native-id", "terminal", {}, '{"exit_code":0,"output":"private-result"}'
    )
    snapshot = turn.snapshot()
    assert snapshot["activities"][0]["state"] == "finished"
    assert "private" not in str(snapshot)


def test_codex_bridge_preserves_explicit_failed_completion(tmp_path):
    from types import SimpleNamespace
    from agent.codex_runtime import make_codex_app_server_event_bridge

    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    turn = adapter.begin_activity_turn(
        home=tmp_path,
        session_key="owned",
        session_id="session-A",
        origin_invocation_id="11111111-1111-4111-8111-111111111111",
    )
    ctx = TurnContext(_run_still_current=lambda: False)
    ctx._background_activity = turn
    observer = TurnRunner(None, ctx)
    agent = SimpleNamespace(
        tool_progress_callback=observer.activity_only_progress,
        tool_start_callback=observer.activity_only_start,
        tool_complete_callback=observer.activity_only_complete,
    )
    bridge = make_codex_app_server_event_bridge(agent)
    item = {
        "type": "commandExecution",
        "id": "native-event",
        "command": "private-command",
    }
    bridge({"method": "item/started", "params": {"item": item}})
    assert turn.snapshot()["activities"][0]["state"] == "running"
    bridge({
        "method": "item/completed",
        "params": {
            "item": {**item, "exitCode": 1, "aggregatedOutput": "private-result"}
        },
    })
    snapshot = turn.snapshot()
    assert snapshot["activities"][0]["state"] == "failed", (
        "Native bridge's explicit is_error must survive the ID-bearing callback"
    )
    assert "private" not in str(snapshot)
