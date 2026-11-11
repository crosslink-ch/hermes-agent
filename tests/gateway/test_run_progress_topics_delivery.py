"""Progress delivery the fork adds: status metadata, live status, edit retry, bubble
rollover, TheChat structured activity and interim commentary."""

import importlib
import pytest
import sys
import time
import types

from gateway.config import Platform
from gateway.config import StreamingConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.platforms.event import MessageType
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import (
    CommentaryAgent,
    FakeAgent,
    ManyProgressLinesAgent,
    ProgressCaptureAdapter,
    SmallLimitProgressAdapter,
    _ACTIVE_ADAPTER,
    _make_runner,
)


class LiveStatusCaptureAdapter(ProgressCaptureAdapter):
    supports_status_text = True

    def __init__(self, platform=Platform.SLACK):
        super().__init__(platform=platform)
        self.status_updates = []

    def set_status_text(self, chat_id, text):
        self.status_updates.append({"chat_id": chat_id, "text": text})


class RetryableFirstEditProgressCaptureAdapter(ProgressCaptureAdapter):
    """Fail one progress edit transiently, then accept later edits."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.edit_outcomes = []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        if not self.edit_outcomes:
            self.edit_outcomes.append(False)
            return SendResult(success=False, error="temporary network failure",
                              retryable=True, error_kind="transient")
        self.edit_outcomes.append(True)
        return SendResult(success=True, message_id=message_id)


class StructuredProgressCaptureAdapter(ProgressCaptureAdapter):
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, platform=Platform.THECHAT):
        super().__init__(platform=platform)
        self.progress_events = []

    async def send_invocation_progress(self, chat_id, event, metadata=None) -> SendResult:
        self.progress_events.append(
            {
                "chat_id": chat_id,
                "event": event,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"progress-{len(self.progress_events)}")


class RetryableEditProgressAgent:
    """Keep the turn alive long enough to retry the same progress bubble."""

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        callback = self.tool_progress_callback
        assert callback is not None
        callback("tool.started", "terminal", "first command", {})
        time.sleep(0.5)
        callback("tool.started", "terminal", "second command", {})
        time.sleep(1.7)
        callback("tool.started", "terminal", "third command", {})
        time.sleep(0.5)
        callback("tool.started", "terminal", "fourth command", {})
        time.sleep(0.6)
        return {"final_response": "done", "messages": [], "api_calls": 1}


class StatusAndToolProgressAgent:
    """Emit both status and tool progress to exercise their metadata paths."""

    def __init__(self, **kwargs):
        self.status_callback = kwargs.get("status_callback")
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        assert self.status_callback is not None
        assert self.tool_progress_callback is not None
        self.status_callback("lifecycle", "Compacting context for test")
        self.tool_progress_callback("tool.started", "terminal", "pwd", {})
        time.sleep(0.4)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class TheChatStructuredActivityAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tool_start_callback = kwargs.get("tool_start_callback")
        self.tool_complete_callback = kwargs.get("tool_complete_callback")
        self.status_callback = kwargs.get("status_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        assert self.status_callback is not None
        assert self.tool_progress_callback is not None
        assert self.tool_start_callback is not None
        assert self.tool_complete_callback is not None

        self.status_callback("lifecycle", "Codex gpt-5.5 caps context at 272K")
        self.status_callback("warn", "Compression provider is unavailable")
        self.status_callback("error", "Compression provider failed")
        self.tool_progress_callback(
            "reasoning.available",
            "_thinking",
            "Drafting a response",
            None,
        )
        self.tool_start_callback(
            "call-skill",
            "skill_view",
            {"name": "html-math-study-notes"},
        )
        self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
        self.tool_progress_callback(
            "tool.completed",
            "terminal",
            None,
            None,
            duration=0.4,
            is_error=False,
        )
        self.tool_complete_callback("call-1", "terminal", {"command": "pwd"}, "ok")
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


async def _run_with_agent(
    monkeypatch,
    tmp_path,
    agent_cls,
    *,
    session_id,
    pending_text=None,
    config_data=None,
    platform=Platform.TELEGRAM,
    chat_id="-1001",
    chat_type="group",
    thread_id="17585",
    event_message_id=None,
    adapter_extra=None,
    adapter_cls=ProgressCaptureAdapter,
    user_id=None,
    scope_id=None,
):
    if config_data:
        import hermes_yaml as yaml

        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config_data), encoding="utf-8")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = adapter_cls(platform=platform)
    if adapter_extra:
        adapter.config.extra.update(adapter_extra)
    _ACTIVE_ADAPTER["adapter"] = adapter
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    if config_data and "streaming" in config_data:
        runner.config.streaming = StreamingConfig.from_dict(config_data["streaming"])
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=thread_id,
        user_id=user_id,
        scope_id=scope_id,
    )
    session_key = f"agent:main:{platform.value}:{chat_type}:{chat_id}"
    if thread_id:
        session_key = f"{session_key}:{thread_id}"
    if pending_text is not None:
        adapter._pending_messages[session_key] = MessageEvent(
            text=pending_text,
            message_type=MessageType.TEXT,
            source=source,
            message_id="queued-1",
        )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key=session_key,
        event_message_id=event_message_id,
    )
    return adapter, result


@pytest.mark.asyncio
async def test_slack_flat_progress_and_status_do_not_recreate_synthetic_thread(
    monkeypatch, tmp_path
):
    event_message_id = "1700000000.000100"
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        StatusAndToolProgressAgent,
        session_id="sess-slack-flat-progress",
        config_data={
            "display": {"platforms": {"slack": {"tool_progress": "all"}}}
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        thread_id=event_message_id,
        event_message_id=event_message_id,
        scope_id="T_ALPHA",
        adapter_extra={"reply_in_thread": False},
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert all(call["metadata"].get("slack_team_id") == "T_ALPHA"
               and "thread_id" not in call["metadata"]
               and "message_id" not in call["metadata"] for call in adapter.sent), adapter.sent
    assert all(call["metadata"].get("slack_team_id") == "T_ALPHA"
               and "thread_id" not in call["metadata"] for call in adapter.typing)


@pytest.mark.asyncio
async def test_slack_live_status_receives_tool_events_when_chat_progress_is_off(
    monkeypatch, tmp_path
):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FakeAgent,
        session_id="sess-slack-live-status-only",
        config_data={
            "display": {
                "platforms": {
                    "slack": {"tool_progress": "off", "live_status": "full"}
                }
            }
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        thread_id=None,
        adapter_cls=LiveStatusCaptureAdapter,
    )

    assert result["final_response"] == "done"
    assert adapter.sent == []
    assert len(adapter.status_updates) == 2
    assert all(update["chat_id"] == "C123" for update in adapter.status_updates)
    assert all(update["text"] for update in adapter.status_updates)


@pytest.mark.asyncio
async def test_retryable_progress_edit_keeps_same_message_id(monkeypatch, tmp_path):
    """A transient edit failure must not create a replacement progress bubble."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        RetryableEditProgressAgent,
        session_id="sess-progress-retry-same-message",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
            }
        },
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="direct",
        thread_id="1700000000.000100",
        adapter_cls=RetryableFirstEditProgressCaptureAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, RetryableFirstEditProgressCaptureAdapter)
    assert len(adapter.sent) == 1
    assert adapter.edit_outcomes[0] is False
    assert any(adapter.edit_outcomes[1:])
    assert {call["message_id"] for call in adapter.edits} == {"progress-1"}
    assert "fourth command" in adapter.edits[-1]["content"]


@pytest.mark.asyncio
async def test_run_agent_rolls_progress_bubble_before_platform_limit(monkeypatch, tmp_path):
    """Tool progress should start a second editable bubble before Telegram's limit.

    Regression: once the first progress bubble grew past the platform limit,
    the gateway kept trying to edit that same oversized full transcript.  The
    Telegram adapter then split-and-sent a fresh continuation on every update,
    causing a noisy trail of one-line messages instead of a new editable bubble.
    """
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        ManyProgressLinesAgent,
        session_id="sess-progress-overflow-rollover",
        config_data={
            "display": {
                "tool_progress": "all",
                "interim_assistant_messages": False,
                "tool_preview_length": 60,
            }
        },
        adapter_cls=SmallLimitProgressAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, SmallLimitProgressAdapter)
    assert len(adapter.sent) >= 2, "expected a fresh progress bubble after the first filled"
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []
    all_bubbles = [call["content"] for call in adapter.sent + adapter.edits]
    assert all(len(text) <= adapter.MAX_MESSAGE_LENGTH for text in all_bubbles)


@pytest.mark.asyncio
async def test_thechat_structured_activity_keeps_notices_reasoning_and_tools_distinct(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        TheChatStructuredActivityAgent,
        session_id="sess-thechat-structured-activity",
        config_data={"display": {"tool_progress": "off"}},
        platform=Platform.THECHAT,
        chat_id="11111111-1111-4111-8111-111111111111",
        chat_type="dm",
        thread_id=None,
        event_message_id="msg-origin-1",
        adapter_cls=StructuredProgressCaptureAdapter,
    )

    assert result["final_response"] == "done"
    assert isinstance(adapter, StructuredProgressCaptureAdapter)
    assert adapter.progress_events
    assert all(
        call["metadata"] == {"message_id": "msg-origin-1"}
        for call in adapter.progress_events
    )
    event_types = [call["event"]["type"] for call in adapter.progress_events]
    assert event_types == [
        "notice.lifecycle",
        "notice.warning",
        "notice.error",
        "reasoning.available",
        "tool.started",
        "tool.started",
        "tool.completed",
    ]
    assert not any(event_type.startswith("status.") for event_type in event_types)
    assert adapter.progress_events[0]["event"]["status"] == "info"
    assert adapter.progress_events[1]["event"]["status"] == "warning"
    assert adapter.progress_events[2]["event"]["status"] == "failed"
    assert adapter.progress_events[3]["event"]["payload"] == {"text": "Drafting a response"}
    assert adapter.progress_events[4]["event"]["toolName"] == "skill_view"
    assert adapter.progress_events[4]["event"]["label"] == "html-math-study-notes"
    assert adapter.progress_events[5]["event"]["toolName"] == "terminal"
    assert adapter.progress_events[6]["event"]["payload"]["duration"] == 0.4


@pytest.mark.asyncio
async def test_run_agent_surfaces_real_interim_commentary(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary",
        config_data={"display": {"interim_assistant_messages": True}},
    )

    assert result.get("already_sent") is not True
    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_surfaces_interim_commentary_by_default(monkeypatch, tmp_path):
    adapter, _result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-default-on",
    )

    assert any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_suppresses_interim_commentary_when_disabled(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-disabled",
        config_data={"display": {"interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_tool_progress_does_not_control_interim_commentary(monkeypatch, tmp_path):
    """tool_progress=all with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-tool-progress",
        config_data={"display": {"tool_progress": "all", "interim_assistant_messages": False}},
    )

    assert result.get("already_sent") is not True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)


@pytest.mark.asyncio
async def test_run_agent_streaming_does_not_enable_completed_interim_commentary(
    monkeypatch, tmp_path
):
    """Streaming alone with interim_assistant_messages=false should not surface commentary."""
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        CommentaryAgent,
        session_id="sess-commentary-streaming",
        config_data={
            "display": {"tool_progress": "off", "interim_assistant_messages": False},
            "streaming": {"enabled": True},
        },
    )

    assert result.get("already_sent") is True
    assert not any(call["content"] == "I'll inspect the repo first." for call in adapter.sent)
