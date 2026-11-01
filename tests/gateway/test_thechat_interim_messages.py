"""TheChat learns which bot messages are progress updates, not finished work.

The gateway marks every mid-turn send with the internal ``_interim_send`` marker (stream safety);
progress sends (heartbeats, busy acks, stuck warnings) mark it ``"progress"``. Only those reach
TheChat as ``interim: true`` on ``POST /hermes-platform/messages``, so TheChat can notify people
only when work is done, failed or needs them. These tests drive the real gateway senders into a
real ``httpx`` client and read the wire body.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner, _interim_metadata
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
INVOCATION_ID = "22222222-2222-4222-8222-222222222222"


class _TheChatServer:
    """Records message posts the way TheChat's API receives them."""

    def __init__(self, on_message=None):
        self.messages: list[dict] = []
        self._on_message = on_message

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/hermes-platform/messages":
            body = json.loads(request.content)
            self.messages.append(body)
            if self._on_message:
                self._on_message(body)
            return httpx.Response(200, json={"messageId": f"message-{len(self.messages)}"})
        return httpx.Response(200, json={"ok": True})


def _adapter(server: _TheChatServer) -> TheChatAdapter:
    adapter = TheChatAdapter(
        PlatformConfig(
            enabled=True,
            token="bot-token",
            extra={"base_url": "http://thechat.test", "webhook_url": "http://gateway.test/thechat/webhook"},
        )
    )
    adapter._client = httpx.AsyncClient(
        base_url="http://thechat.test", transport=httpx.MockTransport(server.handler)
    )
    return adapter


def _invocation_context():
    return {
        "invocation_id": INVOCATION_ID,
        "bot_id": "bot-1",
        "bot_name": "Hermes",
        "conversation_id": CONVERSATION_ID,
        "conversation_name": "Hermes DM",
        "chat_type": "dm",
        "thread_id": None,
    }


def _source(thread_id=None):
    return SessionSource(
        platform=Platform.THECHAT, chat_id=CONVERSATION_ID, chat_type="dm",
        user_id="user-1", user_name="User", thread_id=thread_id,
    )


@pytest.mark.asyncio
async def test_interim_marker_reaches_thechat_and_final_answers_stay_unmarked():
    server = _TheChatServer()
    adapter = _adapter(server)
    adapter._contexts[CONVERSATION_ID] = _invocation_context()

    progress = await adapter.send(
        CONVERSATION_ID, "⏳ Working — 3 min", metadata=_interim_metadata({}, progress=True)
    )
    final = await adapter.send(CONVERSATION_ID, "Here is the report.")

    assert progress.success and final.success
    interim_body, final_body = server.messages
    assert interim_body["interim"] is True
    assert interim_body["invocationId"] == INVOCATION_ID
    # The gateway-internal marker never leaks onto the wire under its own name.
    assert "_interim_send" not in interim_body
    # The final answer carries no flag at all, so older TheChat servers see the same body as before.
    assert "interim" not in final_body
    assert final_body["content"] == "Here is the report."


@pytest.mark.asyncio
async def test_shutdown_notice_still_notifies_because_the_work_was_cut_short():
    """A restart notice is mid-turn (it must not seal a stream) but ends the user's work."""
    server = _TheChatServer()
    adapter = _adapter(server)
    adapter._contexts[CONVERSATION_ID] = _invocation_context()
    runner = object.__new__(GatewayRunner)

    delivered = await runner._send_shutdown_notice(
        adapter, CONVERSATION_ID, "⚠️ Gateway restarting — your current task will be interrupted.",
        "active session", "thechat",
    )

    assert delivered
    assert "interim" not in server.messages[0]


@pytest.mark.asyncio
async def test_drain_refusal_still_notifies_and_a_queued_drain_notice_is_progress():
    """While draining, "send it again later" needs the user; "queued for after restart" does not."""
    server = _TheChatServer()
    adapter = _adapter(server)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._delivery_adapter_for = lambda source: adapter
    runner._status_action_gerund = lambda: "restarting"
    runner._queue_or_replace_pending_event = lambda key, event: None
    event = MessageEvent(text="one more thing", source=_source(), message_id="message-3")

    runner._queue_during_drain_enabled = lambda mode: False
    await runner._send_busy_drain_notice(event, "sess", "interrupt")
    runner._queue_during_drain_enabled = lambda mode: True
    await runner._send_busy_drain_notice(event, "sess", "queue")

    rejected, queued = server.messages
    assert "interim" not in rejected
    assert queued["interim"] is True


@pytest.mark.asyncio
async def test_text_approval_prompt_is_not_interim_because_the_agent_waits_on_it():
    """The plain-text approval/clarify fallback is mid-turn but blocks the agent on a person."""
    server = _TheChatServer()
    adapter = _adapter(server)
    adapter._contexts[CONVERSATION_ID] = _invocation_context()

    await adapter.send(
        CONVERSATION_ID,
        "⚠️ Approve `rm -rf build`? Reply /approve or /deny.",
        metadata=_interim_metadata({"is_approval_prompt": True}),
    )

    assert "interim" not in server.messages[0]


@pytest.mark.asyncio
async def test_long_running_heartbeat_is_posted_as_interim(monkeypatch):
    """The real heartbeat loop: TheChat cannot edit messages, so every tick is a new post."""
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
    runner = object.__new__(GatewayRunner)
    runner._draining = runner._restart_requested = False
    agent = MagicMock()
    runner._running_agents = {"sess": agent}
    # The run ends right after the first heartbeat lands, which stops the loop.
    server = _TheChatServer(on_message=lambda _body: runner._running_agents.clear())
    adapter = _adapter(server)
    runner._delivery_adapter_for = lambda source: adapter
    runner._agent_activity_summary = staticmethod(lambda agent: None)
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False
    ctx = TurnContext(source=_source(), session_key="sess")
    ctx.agent_holder[0] = agent

    await runner._run_agent_notify_long_running(disp, ctx, [None])

    assert len(server.messages) == 1
    heartbeat = server.messages[0]
    assert heartbeat["content"].startswith("⏳")
    assert heartbeat["interim"] is True


@pytest.mark.asyncio
async def test_busy_reply_is_posted_as_interim():
    """"Interrupting current task…" and other busy acks are not a turn's answer."""
    server = _TheChatServer()
    adapter = _adapter(server)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    event = MessageEvent(text="actually, stop", source=_source(thread_id="thread-1"), message_id="message-9")

    await runner._send_busy_ack_reply(event, adapter, "⚡ Interrupting current task. I'll respond shortly.")

    assert len(server.messages) == 1
    ack = server.messages[0]
    assert ack["interim"] is True
    assert ack["threadId"] == "thread-1"
