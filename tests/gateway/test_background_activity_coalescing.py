"""Coalesced results keep every original task's immutable binding."""

import json
from types import SimpleNamespace
import pytest
from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner
from tools import async_delegation as ad


@pytest.mark.asyncio
async def test_coalesced_delegation_delivery_keeps_all_task_bindings(tmp_path):
    from tools.process_registry import process_registry
    from unittest.mock import AsyncMock

    ad._reset_for_tests()
    a = TheChatAdapter(
        PlatformConfig(
            enabled=True, token="isolated", extra={"base_url": "http://127.0.0.1"}
        )
    )
    turns, events = [], []
    try:
        for index in range(2):
            handle = ad.dispatch_async_delegation(
                goal="offline grouped acceptance",
                context=None,
                toolsets=None,
                role="leaf",
                model="offline",
                session_key="owned",
                parent_session_id="session-A",
                runner=lambda: {"status": "completed", "summary": "offline unit"},
            )
            turn = a.begin_activity_turn(
                home=tmp_path,
                session_key="owned",
                session_id="session-A",
                origin_invocation_id=f"{index + 1:08d}-1111-4111-8111-111111111111",
            )
            turn.event(
                "subagent.start",
                subagent_id=f"worker-{index}",
                delegation_id=handle["delegation_id"],
            )
            turn.finish()
            turn.event(
                "subagent.complete", subagent_id=f"worker-{index}", status="completed"
            )
            turns.append(turn)
            while True:
                event = process_registry.completion_queue.get(timeout=5)
                if event.get("delegation_id") == handle["delegation_id"]:
                    events.append(event)
                    break
        gateway = object.__new__(GatewayRunner)
        gateway._completion_delivery_identity = lambda event: None
        gateway._completion_delivery_ready = AsyncMock(return_value=True)
        delivered = []

        async def deliver(text, event, **kwargs):
            delivered.append(event)
            return True

        gateway._deliver_completion_notification = deliver
        assert await gateway._deliver_async_delegation_group_scoped(events) is True
        refs = a.activity_tasks_for_completion(delivered[0])
        assert refs == tuple(turn.task_id for turn in turns), (
            "Coalescing must not orphan sibling goals"
        )
        group = a.begin_activity_turn(
            home=tmp_path,
            session_key="owned",
            session_id="session-A",
            verification=True,
            task_ids=refs,
        )
        group.parent_returned()
        group.finish()
        assert all(turn.snapshot()["parent"]["state"] == "finished" for turn in turns)
    finally:
        ad._reset_for_tests()
