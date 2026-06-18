"""End-to-end gateway coverage for core Hermes session flows.

These tests drive real MessageEvent objects through BasePlatformAdapter.handle_message()
into GatewayRunner._handle_message(), then through the real session store, slash
command handlers, transcript persistence, and adapter send path.

The LLM call itself is stubbed at GatewayRunner._run_agent so the tests stay fast,
hermetic, and credential-free while still exercising the gateway/session behavior
that user-visible platforms depend on.
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource

pytestmark = pytest.mark.asyncio


class RecordingAdapter(BasePlatformAdapter):
    """Small in-memory adapter that still uses BasePlatformAdapter's pipeline."""

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="e2e-test-token"), platform)
        self.sent: list[dict] = []

    async def connect(self) -> bool:
        self._running = True
        return True

    async def disconnect(self) -> None:
        self._running = False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"e2e-send-{len(self.sent)}")

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id}


async def wait_for_adapter_idle(adapter: RecordingAdapter, *, timeout: float = 5.0) -> None:
    """Wait until BasePlatformAdapter's background processing has drained.

    A timeout here is intentionally a test failure: it catches gateway/session
    deadlocks like the TheChat branch-parent lookup hang.
    """

    async def _wait() -> None:
        while adapter._background_tasks or adapter._active_sessions:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_wait(), timeout=timeout)


def write_minimal_gateway_config() -> None:
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  default: e2e-test-model",
                "compression:",
                "  enabled: false",
                "onboarding:",
                "  profile_build: off",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_runner_and_adapter(monkeypatch: pytest.MonkeyPatch, platform: Platform = Platform.THECHAT):
    write_minimal_gateway_config()
    monkeypatch.setenv(f"{platform.value.upper()}_HOME_CHANNEL", "e2e-chat-1")

    runner = GatewayRunner(
        GatewayConfig(platforms={platform: PlatformConfig(enabled=True, token="e2e-test-token")})
    )
    assert runner._session_db is not None
    setattr(runner, "_is_user_authorized", lambda source: True)
    runner._post_turn_goal_continuation = AsyncMock()

    async def fake_run_agent(**kwargs):
        message = kwargs["message"]
        response = f"e2e echo: {message}"
        session_id = kwargs["session_id"]
        # In production AIAgent persists its own user/assistant rows to SQLite;
        # gateway persistence skips DB writes when a SessionDB exists to avoid
        # duplicate rows. Mirror that contract here so the e2e flow exercises
        # real transcript loading/branch copying instead of only session_meta.
        runner.session_store.append_to_transcript(session_id, {"role": "user", "content": message})
        runner.session_store.append_to_transcript(session_id, {"role": "assistant", "content": response})
        return {
            "final_response": response,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ],
            "api_calls": 1,
            "completed": True,
            "history_offset": len(kwargs.get("history") or []),
            "last_prompt_tokens": 7,
            "model": "e2e-test-model",
        }

    setattr(runner, "_run_agent", fake_run_agent)

    adapter = RecordingAdapter(platform)
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[platform] = adapter
    return runner, adapter


def make_thechat_event(
    text: str,
    *,
    chat_id: str = "e2e-chat-1",
    user_id: str = "e2e-user-1",
    thread_id: str | None = None,
    message_id: str | None = None,
    raw_message: dict | None = None,
) -> MessageEvent:
    safe_id = message_id or f"msg-{abs(hash((text, thread_id))) % 1_000_000}"
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.THECHAT,
            chat_id=chat_id,
            user_id=user_id,
            user_name="E2E Tester",
            chat_type="dm",
            thread_id=thread_id,
        ),
        message_id=safe_id,
        raw_message=raw_message or {},
    )


async def dispatch(adapter: RecordingAdapter, event: MessageEvent) -> str:
    before = len(adapter.sent)
    await adapter.handle_message(event)
    await wait_for_adapter_idle(adapter)
    assert len(adapter.sent) > before
    return adapter.sent[-1]["content"]


def transcript_contents(runner: GatewayRunner, session_id: str) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = []
    for message in runner.session_store.load_transcript(session_id):
        role = str(message.get("role") or "")
        raw_content = message.get("content")
        content = raw_content if isinstance(raw_content, str) else None
        rows.append((role, content))
    return rows


async def test_thechat_core_send_title_branch_resume_flow(monkeypatch):
    runner, adapter = make_runner_and_adapter(monkeypatch)
    main_source = make_thechat_event("source").source

    assert await dispatch(adapter, make_thechat_event("hello")) == "e2e echo: hello"
    main_session_id = runner.session_store.get_or_create_session(main_source).session_id

    title_response = await dispatch(adapter, make_thechat_event("/title Primary Session"))
    assert "Primary Session" in title_response
    assert runner._session_db.get_session_title(main_session_id) == "Primary Session"

    branch_response = await dispatch(adapter, make_thechat_event("/branch Branch A"))
    assert "Branch A" in branch_response
    branch_session_id = runner.session_store.get_or_create_session(main_source).session_id
    assert branch_session_id != main_session_id
    assert runner._session_db.get_session(branch_session_id)["parent_session_id"] == main_session_id
    assert runner._session_db.get_session_title(branch_session_id) == "Branch A"

    assert await dispatch(adapter, make_thechat_event("branch followup")) == "e2e echo: branch followup"

    resume_response = await dispatch(adapter, make_thechat_event("/resume Primary Session"))
    assert "Primary Session" in resume_response
    assert runner.session_store.get_or_create_session(main_source).session_id == main_session_id

    assert await dispatch(adapter, make_thechat_event("after resume")) == "e2e echo: after resume"

    main_transcript = transcript_contents(runner, main_session_id)
    branch_transcript = transcript_contents(runner, branch_session_id)
    assert ("user", "hello") in main_transcript
    assert ("assistant", "e2e echo: hello") in main_transcript
    assert ("user", "after resume") in main_transcript
    assert ("user", "branch followup") not in main_transcript
    assert ("user", "hello") in branch_transcript
    assert ("user", "branch followup") in branch_transcript
    assert ("user", "after resume") not in branch_transcript


async def test_thechat_branch_session_intent_creates_child_session_without_hanging(monkeypatch):
    runner, adapter = make_runner_and_adapter(monkeypatch)

    parent_event = make_thechat_event(
        "parent hello",
        thread_id="source-thread",
        message_id="parent-message-1",
    )
    assert await dispatch(adapter, parent_event) == "e2e echo: parent hello"
    parent_entry = runner.session_store.get_session_for_source(parent_event.source)
    assert parent_entry is not None

    branch_event = make_thechat_event(
        "child first message",
        thread_id="branch-thread",
        message_id="branch-message-1",
        raw_message={
            "sessionIntent": {
                "type": "branch",
                "fromThreadId": "source-thread",
                "title": "Child Branch",
            }
        },
    )
    assert await dispatch(adapter, branch_event) == "e2e echo: child first message"

    child_entry = runner.session_store.get_session_for_source(branch_event.source)
    assert child_entry is not None
    assert child_entry.session_id != parent_entry.session_id
    child_row = runner._session_db.get_session(child_entry.session_id)
    assert child_row["parent_session_id"] == parent_entry.session_id
    assert runner._session_db.get_session_title(child_entry.session_id) == "Child Branch"

    child_transcript = transcript_contents(runner, child_entry.session_id)
    assert ("user", "parent hello") in child_transcript
    assert ("assistant", "e2e echo: parent hello") in child_transcript
    assert ("user", "child first message") in child_transcript
    assert getattr(branch_event, "hermes_session")["sessionId"] == child_entry.session_id
