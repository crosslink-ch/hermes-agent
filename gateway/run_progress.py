"""Invocation-scoped structured tool progress for gateway transports."""
from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict
from agent.async_utils import safe_schedule_threadsafe
logger = logging.getLogger("gateway.run")

class StructuredProgressMixin:
    """Publish authoritative ID-bearing activity without chat-history noise."""

    @staticmethod
    def _thechat_notice_event_shape(event_type: str) -> tuple[str, str]:
        """Map generic status callbacks onto TheChat's structured notice rail."""
        normalized = re.sub(
            r"[^a-z0-9._-]+",
            "-",
            str(event_type or "lifecycle").strip().lower(),
        ).strip("-") or "lifecycle"
        if normalized == "error":
            return "notice.error", "failed"
        if normalized in {"warn", "warning"}:
            return "notice.warning", "warning"
        return f"notice.{normalized}", "info"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return str(value)

    def _schedule_structured_progress(self, event: Dict[str, Any]) -> None:
        ctx = self._ctx
        if (
            not ctx._structured_progress_adapter
            or not ctx._structured_progress_supported
            or not ctx._run_still_current()
        ):
            return
        safe_schedule_threadsafe(
            ctx._structured_progress_adapter.send_invocation_progress(
                ctx._status_chat_id,
                event,
                metadata=ctx._status_thread_metadata,
            ),
            ctx._loop_for_step,
            logger=logger,
            log_message="structured progress scheduling error",
        )

    def _record_structured_tool_progress(
        self,
        event_type: str,
        tool_name: str = None,
        preview: str = None,
        **kwargs,
    ) -> None:
        ctx = self._ctx
        if not ctx._structured_progress_supported or not ctx._run_still_current():
            return
        if event_type == "tool.completed" and tool_name:
            meta = {
                "duration": kwargs.get("duration"),
                "isError": bool(kwargs.get("is_error", False)),
            }
            ctx._structured_completion_meta.setdefault(tool_name, []).append(meta)
            return
        if event_type == "reasoning.available":
            self._schedule_structured_progress(
                {
                    "type": "reasoning.available",
                    "status": "running",
                    "preview": preview or "",
                    "payload": {"text": preview or ""},
                }
            )

    def _structured_tool_start_callback_sync(
        self,
        tool_call_id: str,
        function_name: str,
        function_args: dict,
    ) -> None:
        if not tool_call_id or not function_name or function_name.startswith("_"):
            return
        try:
            from agent.display import build_tool_preview

            label = build_tool_preview(function_name, function_args) or function_name
        except Exception:
            label = function_name
        self._schedule_structured_progress(
            {
                "type": "tool.started",
                "status": "running",
                "toolCallId": tool_call_id,
                "toolName": function_name,
                "label": label,
                "preview": label,
                "payload": {"args": self._json_safe(function_args or {})},
            }
        )

    def _structured_tool_complete_callback_sync(
        self,
        tool_call_id: str,
        function_name: str,
        function_args: dict,
        _function_result: Any,
    ) -> None:
        if not tool_call_id or not function_name or function_name.startswith("_"):
            return
        ctx = self._ctx
        meta_items = ctx._structured_completion_meta.get(function_name) or []
        meta = meta_items.pop(0) if meta_items else {}
        duration = meta.get("duration")
        is_error = bool(meta.get("isError", False))
        payload = {
            "args": self._json_safe(function_args or {}),
            "isError": is_error,
        }
        if duration is not None:
            payload["duration"] = duration
        self._schedule_structured_progress(
            {
                "type": "tool.completed",
                "status": "failed" if is_error else "completed",
                "toolCallId": tool_call_id,
                "toolName": function_name,
                "payload": payload,
            }
        )

    def tool_start_callback(self, call_id, tool_name, args) -> None:
        """Fan one tool-start event out to structured progress and voice ack."""
        if self._ctx._structured_progress_supported:
            self._structured_tool_start_callback_sync(call_id, tool_name, args)
        self.voice_ack_callback(call_id, tool_name, args)

    def combined_tool_complete_callback(self, call_id, tool_name, args, result):
        """Compose structured and native task-card completion consumers."""
        ctx = self._ctx
        if ctx._structured_progress_supported:
            self._structured_tool_complete_callback_sync(
                call_id, tool_name, args, result
            )
        if ctx._native_slack_task_cards:
            self.native_tool_complete_callback(call_id, tool_name, args, result)








