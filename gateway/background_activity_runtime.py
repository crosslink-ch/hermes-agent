"""Messaging gateway's supervised idle producer; no model or serve observer."""

import asyncio
import time


async def watch_activity(runner, *, interval=1.0, heartbeat_seconds=15.0):
    last_heartbeat = time.monotonic()
    retries = {}
    while runner._running:
        adapters = list(getattr(runner, "adapters", {}).values())
        for mapping in (getattr(runner, "_profile_adapters", None) or {}).values():
            adapters.extend(mapping.values())
        homes = set((getattr(runner, "_served_profile_homes", None) or {}).values())
        for adapter in dict.fromkeys(adapters):
            if getattr(type(adapter), "publish_activity_once", None) is None:
                continue
            for home in homes | {adapter._activity_owner_home}:
                adapter.restore_activity_home(home)
            if time.monotonic() - last_heartbeat >= heartbeat_seconds:
                for store in list(adapter._activity_stores.values()):
                    store.heartbeat()
            failures, next_attempt = retries.get(adapter, (0, 0.0))
            if time.monotonic() >= next_attempt:
                if await adapter.publish_activity_once():
                    retries.pop(adapter, None)
                else:
                    failures = min(failures + 1, 6)
                    delay = (
                        60
                        if getattr(adapter, "_activity_endpoint_unsupported", False)
                        else min(2**failures, 60)
                    )
                    retries[adapter] = (failures, time.monotonic() + delay)
        retries = {
            adapter: state for adapter, state in retries.items() if adapter in adapters
        }
        if time.monotonic() - last_heartbeat >= heartbeat_seconds:
            last_heartbeat = time.monotonic()
        await asyncio.sleep(interval)


class ActivityGroup:
    """One verification turn can settle several coalesced original tasks."""

    def __init__(self, turns):
        self.turns = tuple(turns)

    def event(self, event_type, tool_name=None, **metadata):
        for turn in self.turns:
            turn.event(event_type, tool_name, **metadata)

    def parent_returned(self, **outcome):
        for turn in self.turns:
            turn.parent_returned(**outcome)

    def finish(self, **outcome):
        for turn in self.turns:
            turn.finish(**outcome)
