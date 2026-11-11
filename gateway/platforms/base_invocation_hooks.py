"""Adapter surfaces the fork adds to every platform: public HTTP routes, structured
invocation progress, and closing the invocation of an event that never runs its own turn."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome

if TYPE_CHECKING:  # annotation only; base imports this module
    from gateway.platforms.base import SendResult

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.platforms.base")


class InvocationHooksMixin:
    """Defaults mixed into ``BasePlatformAdapter``; adapters override what they support."""

    def public_http_routes(self) -> list[dict]:
        """Return public HTTP routes exposed by this adapter, if any.

        Managed deployments can publish these routes to a local reverse proxy
        sidecar. Most adapters do not run an HTTP listener, so the default is
        an empty list.
        """
        return []

    async def send_invocation_progress(
        self,
        chat_id: str,
        event: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a structured in-progress event for a platform invocation.

        This is for first-party platforms that can render tool progress outside
        the chat message history. Most chat platforms only support messages and
        edits, so the default is unsupported.
        """
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not supported")

    async def _dispatch_inline_with_processing_lifecycle(self, event: MessageEvent, session_key: str, cmd: str | None = None) -> None:
        """Close the bypass event's invocation without releasing the active turn's guard."""
        if not self._message_handler:
            return
        await self._run_processing_hook("on_processing_start", event)
        try:
            delivered = await self._dispatch_inline_reply(event, log_cmd=cmd)
        except Exception:
            await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            raise
        await self._run_processing_hook(
            "on_processing_complete", event,
            ProcessingOutcome.SUCCESS if delivered else ProcessingOutcome.FAILURE,
        )

    async def _complete_absorbed_active_session_event(
        self,
        event: MessageEvent,
        session_key: str,
    ) -> None:
        """Close an active-session event that will not run as its own turn."""
        pending_event = self._pending_messages.get(session_key)
        is_command = bool(event.get_command()) or event.message_type == MessageType.COMMAND
        if pending_event is event and not is_command:
            return
        try:
            await self._run_processing_hook("on_processing_start", event)
            await self._run_processing_hook(
                "on_processing_complete",
                event,
                ProcessingOutcome.SUCCESS,
            )
        except Exception:
            logger.debug(
                "[%s] Failed to complete absorbed active-session event",
                self.name,
                exc_info=True,
            )
