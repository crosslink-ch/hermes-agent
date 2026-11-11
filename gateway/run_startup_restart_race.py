"""Startup that aborts into an inline stop must not race the delayed restart helper."""
import asyncio


async def cancel_pending_restart(runner, stop_task, current_task) -> None:
    restart_task = getattr(runner, "_restart_task", None)
    if (
        stop_task is None
        and restart_task is not None
        and restart_task is not current_task
        and not restart_task.done()
    ):
        # Startup noticed the restart request and is about to run the
        # shutdown path inline; the delayed restart helper would only race
        # this same stop() call. Cancel it before it can create/await a
        # second stop waiter.
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass
