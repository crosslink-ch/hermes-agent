"""TheChat session intent and scoped asynchronous title publishing."""
from __future__ import annotations
import asyncio
import dataclasses
import inspect
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Coroutine, cast
from agent.async_utils import safe_schedule_threadsafe
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
logger = logging.getLogger("gateway.run")

class GatewayTheChatMixin:
    """Apply trusted platform branch intents and publish scoped titles."""

    def _thechat_session_intent_for_event(self, event: MessageEvent) -> Dict[str, Any]:
        raw_message = getattr(event, "raw_message", None)
        if not isinstance(raw_message, dict):
            return {}
        session_intent = raw_message.get("sessionIntent")
        return session_intent if isinstance(session_intent, dict) else {}

    def _thechat_session_intent_text(
        self,
        session_intent: Dict[str, Any],
        key: str,
    ) -> Optional[str]:
        value = session_intent.get(key)
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    def _session_entry_is_fresh(self, session_entry: Any) -> bool:
        created_at = getattr(session_entry, "created_at", None)
        updated_at = getattr(session_entry, "updated_at", None)
        if not created_at or not updated_at:
            return False
        try:
            return abs((updated_at - created_at).total_seconds()) < 0.001
        except Exception:
            return False

    def _thechat_branch_parent_source(
        self,
        source: SessionSource,
        branch_from_thread_id: Optional[str],
    ) -> SessionSource:
        return dataclasses.replace(source, thread_id=branch_from_thread_id or None)

    async def _session_db_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a SessionDB method through either the async facade or a sync test double."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        method = getattr(session_db, method_name)
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _apply_thechat_session_intent_async(
        self,
        event: MessageEvent,
        source: SessionSource,
        session_entry: Any,
    ) -> Any:
        if getattr(source, "platform", None) != Platform.THECHAT:
            return session_entry
        session_intent = self._thechat_session_intent_for_event(event)
        if session_intent.get("type") != "branch":
            return session_entry

        session_key = getattr(session_entry, "session_key", None) or self._session_key_for_source(source)
        if self._session_entry_is_fresh(session_entry):
            branch_title = self._thechat_session_intent_text(session_intent, "title")
            branch_from_thread_id = self._thechat_session_intent_text(
                session_intent,
                "fromThreadId",
            )
            parent_source = self._thechat_branch_parent_source(source, branch_from_thread_id)
            try:
                parent_entry = await self.async_session_store.get_session_for_source(parent_source)
            except Exception:
                parent_entry = None
                logger.debug(
                    "Failed to resolve TheChat branch parent source",
                    exc_info=True,
                )
            branch_parent_session_id = (
                str(getattr(parent_entry, "session_id", "") or "").strip()
                if parent_entry is not None
                else None
            )
            if branch_parent_session_id:
                branched_entry = await self._branch_session_from_parent(
                    source=source,
                    session_key=session_key,
                    parent_session_id=branch_parent_session_id,
                    branch_title=branch_title,
                )
                if branched_entry is not None:
                    return branched_entry
        else:
            logger.debug(
                "Ignoring persisted TheChat branch sessionIntent for existing session key %s",
                session_key,
            )

        return session_entry

    async def _branch_session_from_parent(
        self,
        *,
        source: SessionSource,
        session_key: str,
        parent_session_id: str,
        branch_title: Optional[str] = None,
    ) -> Any:
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        parent_session_id = str(parent_session_id or "").strip()
        if not parent_session_id:
            return None
        try:
            parent_session_id = await self._session_db_call("resolve_resume_session_id", parent_session_id)
            parent_session_id = str(parent_session_id or "").strip()
            parent_session = await self._session_db_call("get_session", parent_session_id)
        except Exception:
            logger.debug(
                "Failed to resolve TheChat branch parent %s",
                parent_session_id,
                exc_info=True,
            )
            return None
        if not parent_session:
            logger.debug("TheChat branch parent session not found: %s", parent_session_id)
            return None

        history = await self.async_session_store.load_transcript(parent_session_id)
        if not history:
            try:
                history = await self._session_db_call(
                    "get_messages_as_conversation",
                    parent_session_id,
                    include_ancestors=True,
                )
            except Exception:
                logger.debug(
                    "Failed to load TheChat branch parent messages for %s",
                    parent_session_id,
                    exc_info=True,
                )
                history = []
        if not history:
            return None

        import uuid as _uuid

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_session_id = f"{timestamp_str}_{_uuid.uuid4().hex[:6]}"
        title = (branch_title or "").strip()
        if not title:
            current_title = await self._session_db_call("get_session_title", parent_session_id)
            title = await self._session_db_call("get_next_title_in_lineage", current_title or "branch")

        try:
            await self._session_db_call(
                "create_session",
                session_id=new_session_id,
                source=source.platform.value if source.platform else "gateway",
                model=(self.config.get("model", {}) or {}).get("default") if isinstance(self.config, dict) else None,
                model_config={"_branched_from": parent_session_id},
                parent_session_id=parent_session_id,
            )
        except Exception as exc:
            logger.error("Failed to create TheChat branch session: %s", exc)
            return None

        for msg in history:
            try:
                await self._session_db_call(
                    "append_message",
                    session_id=new_session_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning"),
                    reasoning_content=msg.get("reasoning_content"),
                    reasoning_details=msg.get("reasoning_details"),
                    codex_reasoning_items=msg.get("codex_reasoning_items"),
                    codex_message_items=msg.get("codex_message_items"),
                )
            except Exception:
                pass

        try:
            await self._session_db_call("set_session_title", new_session_id, title)
        except Exception:
            pass

        switched = await self.async_session_store.switch_session(session_key, new_session_id)
        if switched is None:
            return None
        self._clear_session_boundary_security_state(session_key)
        self._evict_cached_agent(session_key)
        return switched

    def _make_thechat_session_title_callback(
        self,
        source: SessionSource,
        *,
        event_message_id: Optional[str] = None,
    ):
        """Build a title callback that propagates generated task titles to TheChat."""
        if (
            getattr(source, "platform", None) != Platform.THECHAT
            or not getattr(source, "thread_id", None)
        ):
            return None

        adapter = self._adapter_for_source(source)
        sender = getattr(adapter, "send_session_title_update", None) if adapter else None
        if not callable(sender):
            return None

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return None

        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source

        metadata = self._thread_metadata_for_source(
            copied_source,
            event_message_id or getattr(copied_source, "message_id", None),
        ) or {}
        metadata = dict(metadata)

        context_snapshot: Optional[Dict[str, Any]] = None
        context_provider = getattr(adapter, "_context_for_send", None)
        if callable(context_provider):
            try:
                context = context_provider(copied_source.chat_id, metadata=metadata)
                if isinstance(context, dict):
                    context_snapshot = dict(context)
            except Exception:
                logger.debug(
                    "Failed to snapshot TheChat context for session title update",
                    exc_info=True,
                )

        def _callback(title: str) -> None:
            self._schedule_thechat_session_title_update(
                copied_source,
                title,
                metadata=metadata,
                context=context_snapshot,
                loop=loop,
                adapter=adapter,
            )

        return _callback

    def _schedule_thechat_session_title_update(
        self,
        source: SessionSource,
        title: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        adapter: Optional[Any] = None,
    ) -> None:
        """Schedule a structured TheChat session.title event from auto-title."""
        title = str(title or "").strip()
        if (
            not title
            or getattr(source, "platform", None) != Platform.THECHAT
            or not getattr(source, "thread_id", None)
        ):
            return

        if adapter is None:
            adapter = self._adapter_for_source(source)
        sender = getattr(adapter, "send_session_title_update", None) if adapter else None
        if not callable(sender):
            return

        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return

        metadata = dict(metadata or self._thread_metadata_for_source(source) or {})

        future = safe_schedule_threadsafe(
            cast(
                Coroutine[Any, Any, Any],
                sender(
                    source.chat_id,
                    title,
                    metadata=metadata,
                    context=context,
                ),
            ),
            loop,
            logger=logger,
            log_message="TheChat session title update failed to schedule",
        )
        if future is None:
            return

        def _log_title_update_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("TheChat session title update failed", exc_info=True)

        future.add_done_callback(_log_title_update_failure)









