"""Server-bound task control, admitted before conversation-wide busy commands."""

import asyncio
from pathlib import Path

from gateway.platforms.base import EphemeralReply


async def stop_bound_task(runner, source, control):
    from gateway.run import _async_profile_runtime_scope
    from tools.process_registry import process_registry

    adapter = runner._delivery_adapter_for(source)
    home = Path(runner._resolve_profile_home_for_source(source)).resolve()
    for store in list(getattr(adapter, "_activity_stores", {}).values()):
        if store.home != home:
            continue
        key = store.control_target(
            control["taskId"], control["sessionId"], source.chat_id, source.thread_id
        )
        if key is None:
            continue
        requested, process_ids = store.request_stop(
            key, target_task_id=control["taskId"]
        )
        if not requested:
            return EphemeralReply("Task has no live execution to stop.")
        async with _async_profile_runtime_scope(store.home):
            from tools.async_delegation import interrupt_for_bindings

            interrupt_for_bindings(
                store.control_links(control["taskId"]), owner_home=store.home
            )
            parent = runner._running_agents.get(key)
            activity = getattr(parent, "_background_activity", None)
            turns = (
                tuple(getattr(activity, "turns", (activity,)))
                if activity is not None
                else ()
            )
            owned = tuple(
                turn
                for turn in turns
                if getattr(turn, "store", None) is store
                and getattr(turn, "task_id", None) == control["taskId"]
            )
            # A shared verifier drains for un-stopped siblings; never cancel another goal.
            if (
                owned
                and store.control_runtime(control["taskId"])[0]
                and all(turn.store.control_runtime(turn.task_id)[1] for turn in turns)
            ):
                from agent.interrupt_compat import request_hard_interrupt

                request_hard_interrupt(parent, tool_reason="background_task_control")
            for process_id in process_ids:
                process = process_registry.get(process_id)
                if (
                    process is not None
                    and process.session_key == key
                    and not process.exited
                ):
                    await asyncio.to_thread(
                        process_registry.kill_process,
                        process_id,
                        source="background_task_control",
                        consume_output=True,
                    )
        return EphemeralReply(
            "Stop requested. Waiting for this task's owned workers to settle."
        )
    return EphemeralReply("Task binding is unavailable; no other session was stopped.")
