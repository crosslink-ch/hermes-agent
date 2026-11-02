"""Per-turn environment and browser cleanup; lifecycle leases outlive tool dispatch."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

def cleanup_task_resources(agent, task_id: str) -> None:
    """Release this logical turn before cleaning its non-persistent environments.

    A sibling turn may still be using a shared environment. Defer collapsed
    cleanup until that sibling finishes; the idle reaper owns persistent ones.
    Headed browser sessions likewise remain open until their idle reaper runs.
    """
    from agent.chat_completion_helpers import _ra
    try:
        from tools.terminal_tool_target_lifecycle import (
            active_environment_turns,
            defer_environment_turn_cleanup,
            release_logical_environment_turn_for_cleanup,
        )

        release_logical_environment_turn_for_cleanup(task_id)
        include_collapsed = active_environment_turns(task_id) == 0
        if not include_collapsed:
            defer_environment_turn_cleanup(task_id)
        _ra().cleanup_vm(
            task_id, preserve_persistent=True, include_collapsed=include_collapsed,
        )
    except Exception as e:
        logger.warning("Failed to cleanup VM for task %s: %s", task_id, e, exc_info=True)

    from tools.browser_session_cloud import is_headed_mode
    headed = is_headed_mode()
    if headed:
        if agent.verbose_logging:
            logging.debug(
                "Skipping per-turn cleanup_browser for headed session %s; "
                "idle reaper will handle it.", task_id,
            )
    else:
        try:
            _ra().cleanup_browser(task_id)
        except Exception as e:
            logger.warning("Failed to cleanup browser for task %s: %s", task_id, e, exc_info=True)
