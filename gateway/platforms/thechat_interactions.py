"""TheChat approval and clarify interactions.

Legacy progress-event cards (used when the server does not advertise bot requests) and the
signed direct-interaction callbacks that answer them: the request ledger, the wire contract and
its bounds, exact-waiter resolution and the card transition published afterwards.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("gateway.platforms.thechat")

_INTERACTION_REQUEST_MAX_COUNT = 512
_INTERACTION_ID_MAX_CHARS = 255
_INTERACTION_REQUEST_ID_MAX_CHARS = 255
_INTERACTION_SESSION_KEY_MAX_CHARS = 1000
_INTERACTION_THREAD_ID_MAX_CHARS = 512
_INTERACTION_RESPONSE_MAX_CHARS = 4000
_INTERACTION_RESPONSE_MAX_COUNT = 20
_INTERACTION_RESPONSE_ITEM_MAX_CHARS = 500
_INTERACTION_QUESTION_MAX_CHARS = 10_000
_INTERACTION_COMMAND_MAX_CHARS = 100_000
_INTERACTION_DESCRIPTION_MAX_CHARS = 10_000
_INTERACTION_CHOICE_MAX_CHARS = 500
_INTERACTION_CHOICE_MAX_COUNT = 20
_INTERACTION_RECORD_MAX_LIFETIME_SECONDS = 24 * 60 * 60
_INTERACTION_PROGRESS_MAX_ATTEMPTS = 3
_INTERACTION_PROGRESS_RETRY_BASE_SECONDS = 0.1


class _DirectInteractionError(ValueError):
    """Safe HTTP error raised while resolving a direct interaction."""

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class TheChatInteractionsMixin:
    """Approval/clarify request cards and direct-interaction resolution for ``TheChatAdapter``."""

    @staticmethod
    def _redact_interaction_value(value: Any) -> Any:
        """Redact strings before reflecting request/response data to TheChat."""
        from agent.redact import redact_sensitive_text

        if isinstance(value, str):
            return redact_sensitive_text(value, force=True)
        if isinstance(value, list):
            return [TheChatInteractionsMixin._redact_interaction_value(item) for item in value]
        return value

    def _forget_interaction_request(self, request_id: str) -> None:
        record = self._interaction_requests.pop(request_id, None)
        if not record or record.get("request_type") != "approval.request":
            return
        session_key = str(record["session_key"])
        request_ids = self._approval_request_ids.get(session_key)
        if request_ids and request_id in request_ids:
            request_ids.remove(request_id)
            if not request_ids:
                self._approval_request_ids.pop(session_key, None)
                self._approval_contexts.pop(session_key, None)

    def _prune_expired_interaction_requests(self) -> None:
        now = time.monotonic()
        expired = [
            request_id
            for request_id, record in self._interaction_requests.items()
            if float(record.get("expires_at") or 0.0) <= now
        ]
        for request_id in expired:
            self._forget_interaction_request(request_id)

    def _remember_interaction_request(
        self,
        *,
        request_type: str,
        request_id: str,
        session_key: str,
        context: dict[str, Any],
        choices: Optional[list[str]],
        multi_select: bool = False,
    ) -> dict[str, Any]:
        self._prune_expired_interaction_requests()
        if request_id in self._interaction_requests:
            raise ValueError(f"Duplicate pending interaction request id: {request_id}")
        if len(self._interaction_requests) >= _INTERACTION_REQUEST_MAX_COUNT:
            raise RuntimeError("Too many pending TheChat interaction requests")
        record: dict[str, Any] = {
            "request_type": request_type,
            "request_id": request_id,
            "session_key": session_key,
            "invocation_id": str(context["invocation_id"]),
            "conversation_id": str(context["conversation_id"]),
            "thread_id": str(context.get("thread_id") or "") or None,
            "context": dict(context),
            "choices": list(choices) if choices else None,
            "multi_select": bool(multi_select),
            "expires_at": (
                time.monotonic() + _INTERACTION_RECORD_MAX_LIFETIME_SECONDS
            ),
        }
        self._interaction_requests[request_id] = record
        if request_type == "approval.request":
            self._approval_request_ids.setdefault(session_key, []).append(request_id)
            self._approval_contexts[session_key] = record["context"]
        return record

    async def _send_interaction_request_progress(
        self,
        chat_id: str,
        event: dict[str, Any],
        *,
        metadata: Optional[dict[str, Any]],
        context: Optional[dict[str, Any]],
    ) -> SendResult:
        """Retry ambiguous request publication without changing its stable id."""
        result = SendResult(success=False, error="Interaction progress was not sent")
        for attempt in range(_INTERACTION_PROGRESS_MAX_ATTEMPTS):
            result = await self.send_invocation_progress(
                chat_id,
                event,
                metadata=metadata,
                context=context,
            )
            if result.success or not result.retryable:
                return result
            if attempt + 1 < _INTERACTION_PROGRESS_MAX_ATTEMPTS:
                await asyncio.sleep(
                    _INTERACTION_PROGRESS_RETRY_BASE_SECONDS * (2**attempt)
                )
        return result

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Publish a structured clarify request on the original invocation."""
        from tools import clarify_gateway

        pending = clarify_gateway.get_pending_request(clarify_id)
        multi_select = bool(pending and pending.get("multi_select"))
        safe_question = str(self._redact_interaction_value(str(question)))
        safe_choices = (
            [str(self._redact_interaction_value(str(choice))) for choice in choices]
            if choices
            else None
        )
        if (
            not clarify_id
            or len(clarify_id) > _INTERACTION_REQUEST_ID_MAX_CHARS
            or len(session_key) > _INTERACTION_SESSION_KEY_MAX_CHARS
            or len(safe_question) > _INTERACTION_QUESTION_MAX_CHARS
            or (
                safe_choices is not None
                and (
                    len(safe_choices) > _INTERACTION_CHOICE_MAX_COUNT
                    or any(
                        not choice.strip()
                        or len(choice) > _INTERACTION_CHOICE_MAX_CHARS
                        for choice in safe_choices
                    )
                )
            )
        ):
            return SendResult(success=False, error="TheChat clarify request is too large")
        context = self._context_for_send(chat_id, metadata=metadata)
        if self._bot_requests.enabled:
            return await self._bot_requests.send_clarify(
                chat_id=chat_id,
                question=question,
                choices=choices,
                clarify_id=clarify_id,
                session_key=session_key,
                multi_select=multi_select,
                context=context,
                thread_id=self._metadata_thread_id(metadata),
            )
        if not context:
            return SendResult(
                success=False,
                error=f"No TheChat clarify context for session {session_key}",
            )
        self._remember_interaction_request(
            request_type="clarify.request",
            request_id=clarify_id,
            session_key=session_key,
            context=context,
            choices=safe_choices,
            multi_select=multi_select,
        )
        progress_event = {
            "type": "clarify.request",
            "status": "waiting",
            "label": "Input required",
            "preview": safe_question[:4000],
            "payload": {
                "requestId": clarify_id,
                "sessionKey": session_key,
                "question": safe_question,
                "choices": safe_choices,
                "multiSelect": multi_select,
                "allowOther": True,
            },
        }
        result = await self._send_interaction_request_progress(
            chat_id,
            progress_event,
            metadata=metadata,
            context=context,
        )
        if result.success:
            # Preserve the normal gateway typed-answer fallback alongside the
            # direct card interaction. Slash commands still bypass capture.
            clarify_gateway.mark_awaiting_text(clarify_id)
        elif not result.retryable:
            self._forget_interaction_request(clarify_id)
        return result

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
        request_id: Optional[str] = None,
    ) -> SendResult:
        """Send command approval as structured TheChat invocation progress."""
        request_id = request_id or uuid.uuid4().hex
        safe_command = str(self._redact_interaction_value(str(command)))
        safe_description = str(self._redact_interaction_value(str(description)))
        if (
            not request_id
            or len(request_id) > _INTERACTION_REQUEST_ID_MAX_CHARS
            or len(session_key) > _INTERACTION_SESSION_KEY_MAX_CHARS
            or len(safe_command) > _INTERACTION_COMMAND_MAX_CHARS
            or len(safe_description) > _INTERACTION_DESCRIPTION_MAX_CHARS
        ):
            return SendResult(success=False, error="TheChat approval request is too large")
        command_preview = (
            safe_command[:4000] + "..."
            if len(safe_command) > 4000
            else safe_command
        )
        if smart_denied or not allow_session:
            choices = ["once", "deny"]
        else:
            choices = ["once", "session"]
            if allow_permanent:
                choices.append("always")
            choices.append("deny")
        context = self._context_for_send(chat_id, metadata=metadata)
        if self._bot_requests.enabled:
            return await self._bot_requests.send_exec_approval(
                chat_id=chat_id,
                command=command,
                description=description,
                session_key=session_key,
                request_id=request_id,
                choices=choices,
                context=context,
                thread_id=self._metadata_thread_id(metadata),
            )
        if context:
            self._remember_interaction_request(
                request_type="approval.request",
                request_id=request_id,
                session_key=session_key,
                context=context,
                choices=choices,
            )
        progress_event = {
            "type": "approval.request",
            "status": "waiting",
            "label": "Command approval required",
            "preview": command_preview,
            "payload": {
                "requestId": request_id,
                "command": safe_command,
                "description": safe_description,
                "sessionKey": session_key,
                "choices": choices,
            },
        }
        result = await self._send_interaction_request_progress(
            chat_id,
            progress_event,
            metadata=metadata,
            context=context,
        )
        if not result.success and not result.retryable:
            self._forget_interaction_request(request_id)
        return result

    async def send_approval_resolution(
        self,
        chat_id: str,
        session_key: str,
        choice: str,
        resolved_count: int = 1,
        resolve_all: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
        request_ids: Optional[list[str]] = None,
        _forget_request: bool = True,
    ) -> SendResult:
        """Publish approval.resolved progress so clients dismiss the card.

        TheChat resolves approval cards oldest-first per ``sessionKey``,
        mirroring ``resolve_gateway_approval``; ``resolveAll`` collapses every
        pending card for the session at once.

        With bot requests the card is withdrawn instead ("Allowed once in the
        conversation."); without exact ids the approval's settle hook closes it.
        """
        if self._bot_requests.enabled:
            ids = [request_id] if request_id is not None else list(request_ids or [])
            await self._bot_requests.approvals_resolved_in_chat(ids, choice)
            return SendResult(success=True)
        records: list[dict[str, Any]] = []
        if request_id is not None:
            target_request_ids = [request_id]
        elif request_ids is not None:
            target_request_ids = list(request_ids)
        else:
            target_request_ids = []

        if target_request_ids:
            if len(set(target_request_ids)) != len(target_request_ids):
                return SendResult(success=False, error="Duplicate approval request ids")
            for target_request_id in target_request_ids:
                record = self._interaction_requests.get(target_request_id)
                if (
                    record is None
                    or record.get("request_type") != "approval.request"
                    or record.get("session_key") != session_key
                ):
                    return SendResult(
                        success=False,
                        error=(
                            "No TheChat approval request context for "
                            f"{target_request_id}"
                        ),
                    )
                records.append(record)
        elif request_id is not None or request_ids is not None:
            return SendResult(success=False, error="No approval requests were resolved")
        else:
            legacy_request_ids = list(
                self._approval_request_ids.get(session_key) or []
            )
            if not resolve_all:
                legacy_request_ids = legacy_request_ids[:1]
            records = [
                self._interaction_requests[candidate]
                for candidate in legacy_request_ids
                if candidate in self._interaction_requests
            ]

        if records:
            last_result: Optional[SendResult] = None
            for record in records:
                current_id = str(record["request_id"])
                try:
                    last_result = await self.send_invocation_progress(
                        str(record["conversation_id"]),
                        {
                            "type": "approval.resolved",
                            "status": "completed",
                            "label": "Approval resolved",
                            "payload": {
                                "requestId": current_id,
                                "choice": choice,
                                "sessionKey": session_key,
                                "resolveAll": resolve_all,
                                "resolvedCount": resolved_count,
                            },
                        },
                        context=record["context"],
                    )
                finally:
                    if _forget_request:
                        self._forget_interaction_request(current_id)
                if not last_result.success:
                    return last_result
            assert last_result is not None
            return last_result

        context = self._approval_contexts.get(session_key) or self._context_for_send(
            chat_id, metadata=metadata
        )
        if not context:
            return SendResult(
                success=False,
                error=f"No TheChat approval context for session {session_key}",
            )
        return await self.send_invocation_progress(
            chat_id,
            {
                "type": "approval.resolved",
                "status": "completed",
                "label": "Approval resolved",
                "payload": {
                    "choice": choice,
                    "sessionKey": session_key,
                    "resolveAll": resolve_all,
                    "resolvedCount": resolved_count,
                },
            },
            metadata=metadata,
            context=context,
        )

    async def send_clarify_resolution(
        self,
        chat_id: str,
        session_key: str,
        request_id: str,
        response: Any,
        _forget_request: bool = True,
    ) -> SendResult:
        """Publish clarify.resolved to the invocation that asked the question."""
        if self._bot_requests.enabled:
            from gateway.platforms.thechat_requests import ANSWERED_IN_CHAT_REASON

            await self._bot_requests.clarify_closed(request_id, ANSWERED_IN_CHAT_REASON)
            return SendResult(success=True)
        record = self._interaction_requests.get(request_id)
        if (
            record is None
            or record.get("request_type") != "clarify.request"
            or record.get("session_key") != session_key
        ):
            return SendResult(
                success=False,
                error=f"No TheChat clarify request context for {request_id}",
            )
        normalized_response = response
        if record.get("multi_select") and isinstance(response, str):
            try:
                decoded = json.loads(response)
                if isinstance(decoded, list) and all(
                    isinstance(item, str) for item in decoded
                ):
                    normalized_response = decoded
            except (TypeError, ValueError):
                pass
        safe_response = self._redact_interaction_value(normalized_response)
        try:
            return await self.send_invocation_progress(
                str(record["conversation_id"]),
                {
                    "type": "clarify.resolved",
                    "status": "completed",
                    "label": "Input received",
                    "payload": {
                        "requestId": request_id,
                        "sessionKey": session_key,
                        "response": safe_response,
                    },
                },
                context=record["context"],
            )
        finally:
            if _forget_request:
                self._forget_interaction_request(request_id)

    async def retire_clarify_card(self, clarify_id: str, notice: str) -> None:
        """Withdraw a clarify request that ended without an inbox answer (timeout, /new, run end,
        superseding prose). Legacy progress cards have no persistent state to retire."""
        if not self._bot_requests.enabled:
            return
        from gateway.platforms.thechat_requests import ANSWERED_IN_CHAT_REASON

        reason = str(notice or "").lstrip("\u23f3\u21a9\ufe0f\u2705 ").strip()
        if reason.lower().startswith("answered"):
            # Never echo the typed answer onto a card other participants can see.
            reason = ANSWERED_IN_CHAT_REASON
        await self._bot_requests.clarify_closed(clarify_id, reason)

    @staticmethod
    def _metadata_thread_id(metadata: Optional[dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        thread_id = metadata.get("thread_id") or metadata.get("message_thread_id")
        return str(thread_id) if thread_id else None

    @staticmethod
    def _validate_bounded_token(value: Any, *, field: str, limit: int) -> str:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > limit
            or value != value.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError(f"TheChat interaction has an invalid {field}")
        return value

    def _extract_webhook_interaction(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object")
        if payload.get("type") != "thechat.hermes_platform.interaction":
            raise ValueError("Webhook payload has an invalid TheChat interaction type")
        if set(payload) != {"type", "interaction"}:
            raise ValueError("TheChat interaction envelope has invalid fields")
        interaction = payload.get("interaction")
        return self._validate_direct_interaction(interaction)

    def _validate_direct_interaction(self, item: Any) -> dict[str, Any]:
        """Validate the signed direct-interaction wire contract and bounds."""
        if not isinstance(item, dict):
            raise ValueError("TheChat interaction must be a JSON object")
        required = {
            "id",
            "requestType",
            "requestId",
            "invocationId",
            "conversationId",
            "threadId",
            "sessionKey",
            "response",
        }
        if set(item) != required:
            raise ValueError("TheChat interaction has invalid fields")

        self._validate_bounded_token(
            item["id"], field="id", limit=_INTERACTION_ID_MAX_CHARS
        )
        self._validate_bounded_token(
            item["requestId"],
            field="requestId",
            limit=_INTERACTION_REQUEST_ID_MAX_CHARS,
        )
        self._validate_bounded_token(
            item["sessionKey"],
            field="sessionKey",
            limit=_INTERACTION_SESSION_KEY_MAX_CHARS,
        )
        for field in ("invocationId", "conversationId"):
            value = self._validate_bounded_token(
                item[field], field=field, limit=36
            )
            if not self._is_current_chat_id(value):
                raise ValueError(f"TheChat interaction has an invalid {field}")

        thread_id = item["threadId"]
        if thread_id is not None:
            self._validate_bounded_token(
                thread_id,
                field="threadId",
                limit=_INTERACTION_THREAD_ID_MAX_CHARS,
            )

        request_type = item["requestType"]
        if not isinstance(request_type, str) or request_type not in {
            "approval.request",
            "clarify.request",
        }:
            raise ValueError("TheChat interaction has an invalid requestType")

        response = item["response"]
        if request_type == "approval.request":
            if not isinstance(response, str) or response not in {
                "once",
                "session",
                "always",
                "deny",
            }:
                raise ValueError("TheChat approval interaction has an invalid response")
        elif isinstance(response, str):
            if not response.strip() or len(response) > _INTERACTION_RESPONSE_MAX_CHARS:
                raise ValueError("TheChat clarify interaction has an invalid response")
        elif isinstance(response, list):
            if not response or len(response) > _INTERACTION_RESPONSE_MAX_COUNT:
                raise ValueError("TheChat clarify interaction has an invalid response")
            total_chars = 0
            for choice in response:
                if (
                    not isinstance(choice, str)
                    or not choice.strip()
                    or len(choice) > _INTERACTION_RESPONSE_ITEM_MAX_CHARS
                ):
                    raise ValueError(
                        "TheChat clarify interaction has an invalid response"
                    )
                total_chars += len(choice)
            if total_chars > _INTERACTION_RESPONSE_MAX_CHARS:
                raise ValueError("TheChat clarify interaction has an invalid response")
        else:
            raise ValueError("TheChat clarify interaction has an invalid response")
        return item

    def _match_interaction_request(self, interaction: dict[str, Any]) -> dict[str, Any]:
        request_id = interaction["requestId"]
        record = self._interaction_requests.get(request_id)
        if record is None:
            raise _DirectInteractionError(
                "TheChat interaction is stale or has no waiting request",
                status=409,
            )
        expected = {
            "request_type": interaction["requestType"],
            "request_id": request_id,
            "session_key": interaction["sessionKey"],
            "invocation_id": interaction["invocationId"],
            "conversation_id": interaction["conversationId"],
            "thread_id": interaction["threadId"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise _DirectInteractionError(
                "TheChat interaction does not match the original request context",
                status=409,
            )

        response = interaction["response"]
        choices = record.get("choices") or []
        if interaction["requestType"] == "approval.request":
            if response not in choices:
                raise _DirectInteractionError(
                    "TheChat approval response was not offered for this request",
                    status=400,
                )
        elif isinstance(response, list):
            if not record.get("multi_select") or any(
                choice not in choices for choice in response
            ):
                raise _DirectInteractionError(
                    "TheChat clarify selections were not offered for this request",
                    status=400,
                )
        return record

    async def _resolve_direct_interaction(
        self,
        interaction: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._match_interaction_request(interaction)
        request_id = str(record["request_id"])
        session_key = str(record["session_key"])
        response = interaction["response"]

        if interaction["requestType"] == "approval.request":
            from tools.approval import resolve_gateway_approval

            resolved = resolve_gateway_approval(
                session_key,
                str(response),
                request_id=request_id,
            )
        else:
            from tools.clarify_gateway import resolve_gateway_clarify

            clarify_response = (
                json.dumps(response, ensure_ascii=False)
                if isinstance(response, list)
                else response
            )
            resolved = int(resolve_gateway_clarify(request_id, clarify_response))

        if resolved != 1:
            if interaction["requestType"] == "approval.request":
                from tools.approval import has_gateway_approval_request

                if not has_gateway_approval_request(session_key, request_id):
                    self._forget_interaction_request(request_id)
            else:
                from tools.clarify_gateway import get_request_state

                if get_request_state(request_id) == "missing":
                    self._forget_interaction_request(request_id)
            raise _DirectInteractionError(
                "TheChat interaction is stale or has no waiting request",
                status=409,
            )

        self.resume_typing_for_chat(str(record["conversation_id"]))
        return record

    async def _publish_direct_interaction_resolution(
        self,
        interaction: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        """Publish the card transition off the webhook response critical path."""
        request_id = str(record["request_id"])
        session_key = str(record["session_key"])
        try:
            for attempt in range(_INTERACTION_PROGRESS_MAX_ATTEMPTS):
                if interaction["requestType"] == "approval.request":
                    published = await self.send_approval_resolution(
                        chat_id=str(record["conversation_id"]),
                        session_key=session_key,
                        choice=str(interaction["response"]),
                        request_id=request_id,
                        _forget_request=False,
                    )
                else:
                    published = await self.send_clarify_resolution(
                        chat_id=str(record["conversation_id"]),
                        session_key=session_key,
                        request_id=request_id,
                        response=interaction["response"],
                        _forget_request=False,
                    )
                if published.success:
                    return
                if not published.retryable:
                    break
                if attempt + 1 < _INTERACTION_PROGRESS_MAX_ATTEMPTS:
                    await asyncio.sleep(
                        _INTERACTION_PROGRESS_RETRY_BASE_SECONDS * (2**attempt)
                    )
            # Avoid transport detail here because errors may contain a signed
            # URL or credential-bearing response body.
            logger.warning(
                "TheChat: interaction resolved but resolution progress failed"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "TheChat: interaction resolution publication failed",
                exc_info=True,
            )
        finally:
            self._forget_interaction_request(request_id)

    def _schedule_direct_interaction_resolution(
        self,
        interaction: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        task = asyncio.create_task(
            self._publish_direct_interaction_resolution(interaction, record),
            name=f"thechat-interaction-resolution-{record['request_id']}",
        )
        self._webhook_tasks.add(task)
        task.add_done_callback(self._webhook_tasks.discard)
