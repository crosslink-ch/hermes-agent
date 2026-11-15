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

    def _thechat_session_intent_for_event(self, event: MessageEvent) -> dict[str, Any]:
        raw_message = getattr(event, "raw_message", None)
        if not isinstance(raw_message, dict):
            return {}
        session_intent = raw_message.get("sessionIntent")
        return session_intent if isinstance(session_intent, dict) else {}

    def _thechat_session_intent_text(
        self,
        session_intent: dict[str, Any],
        key: str,
    ) -> Optional[str]:
        value = session_intent.get(key)
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

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

    async def _prepare_thechat_branch_route(self, event: MessageEvent, source: SessionSource) -> Any:
        """Start a new TheChat branch thread on a copy of its parent's conversation.

        Runs on the inbound message before admission: the session authority reserves the thread's
        route (and its SQLite row) when it admits the message, and executes the turn from the stored
        payload, which does not keep the platform's raw event. Applied any later, the branch never
        happens and the thread starts empty.

        TheChat marks every message of a branch thread with ``sessionIntent`` and never clears it,
        so the branch is made only while the thread has no route yet, i.e. on its first message.
        Returns the branch's routing entry, or None when nothing was branched.
        """
        if getattr(source, "platform", None) != Platform.THECHAT or getattr(event, "internal", False):
            return None
        session_intent = self._thechat_session_intent_for_event(event)
        if session_intent.get("type") != "branch":
            return None
        from gateway.session_authority import keyed_turn
        session_key = self._session_key_for_source(source)
        # Two first messages of one new thread must not both branch.
        async with keyed_turn(self.__dict__.setdefault("_thechat_branch_locks", {}), session_key):
            if await self.async_session_store.lookup_by_session_key(session_key) is not None:
                return None  # a later message of the thread: its route (the branch) exists
            return await self._branch_new_thechat_thread(source, session_key, session_intent)

    async def _branch_new_thechat_thread(
        self,
        source: SessionSource,
        session_key: str,
        session_intent: dict[str, Any],
    ) -> Any:
        from_thread_id = self._thechat_session_intent_text(session_intent, "fromThreadId")
        parent_source = self._thechat_branch_parent_source(source, from_thread_id)
        try:
            parent_entry = await self.async_session_store.get_session_for_source(parent_source)
        except Exception:
            parent_entry = None
            logger.debug("Failed to resolve TheChat branch parent source", exc_info=True)
        parent_session_id = str(getattr(parent_entry, "session_id", "") or "").strip()
        if not parent_session_id:
            logger.warning(
                "TheChat branch thread %s starts empty: its source thread %s has no Hermes session",
                source.thread_id, from_thread_id or "(main lane)",
            )
            return None
        branched = await self._branch_session_from_parent(
            source=source,
            session_key=session_key,
            parent_session_id=parent_session_id,
            branch_title=self._thechat_session_intent_text(session_intent, "title"),
        )
        if branched is None:
            logger.warning(
                "TheChat branch thread %s starts empty: copying parent session %s failed",
                source.thread_id, parent_session_id,
            )
        else:
            logger.info(
                "TheChat branch thread %s starts from parent session %s as %s",
                source.thread_id, parent_session_id, getattr(branched, "session_id", None),
            )
        return branched

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
            logger.error("Failed to create TheChat branch session: %s", exc, exc_info=True)
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
                logger.debug(
                    "Failed to copy a message into TheChat branch session %s",
                    new_session_id,
                    exc_info=True,
                )

        try:
            await self._session_db_call("set_session_title", new_session_id, title)
        except ValueError:
            # Conflict or validation failure. The message quotes the title, so it stays out.
            logger.debug("TheChat branch session %s keeps no title: it was rejected", new_session_id)
        except Exception:
            logger.debug(
                "Failed to set the title of TheChat branch session %s",
                new_session_id,
                exc_info=True,
            )

        if await self.async_session_store.lookup_by_session_key(session_key) is None:
            # A new thread has no route to repoint yet; materialize it, as /branch does for a thread.
            await self.async_session_store.get_or_create_session(source)
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

        adapter = self._delivery_adapter_for(source)
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
        except (TypeError, ValueError):  # not a dataclass instance, or one replace() cannot rebuild
            copied_source = source

        metadata = self._thread_metadata_for_source(
            copied_source,
            event_message_id or getattr(copied_source, "message_id", None),
        ) or {}
        metadata = dict(metadata)

        context_snapshot: Optional[dict[str, Any]] = None
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
        metadata: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
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
            adapter = self._delivery_adapter_for(source)
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
