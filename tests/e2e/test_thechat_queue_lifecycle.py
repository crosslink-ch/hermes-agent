"""TheChat /queue lifecycle regression tests.

Drives real TheChatAdapter events through BasePlatformAdapter and GatewayRunner.
No LLM or network is used: TheChat's HTTP client is replaced with a recorder.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.thechat import TheChatAdapter
from tests.e2e.conftest import make_runner


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class RecordingTheChatClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any] | None]] = []
        self.invocation_status: dict[str, str] = {}
        self._message_counter = 0

    async def post(self, path: str, json: dict[str, Any] | None = None) -> FakeResponse:
        self.posts.append((path, json))
        if path == "/hermes-platform/messages":
            self._message_counter += 1
            return FakeResponse({"messageId": f"thechat-message-{self._message_counter}"})
        if path.endswith("/completed"):
            invocation_id = path.split("/")[-2]
            self.invocation_status[invocation_id] = "completed"
            return FakeResponse({"ok": True})
        if path.endswith("/failed"):
            invocation_id = path.split("/")[-2]
            self.invocation_status[invocation_id] = "failed"
            return FakeResponse({"ok": True})
        if path.endswith("/cancelled"):
            invocation_id = path.split("/")[-2]
            self.invocation_status[invocation_id] = "cancelled"
            return FakeResponse({"ok": True})
        return FakeResponse({"ok": True})

    def message_payloads(self) -> list[dict[str, Any]]:
        return [payload or {} for path, payload in self.posts if path == "/hermes-platform/messages"]

    def completed_invocations(self) -> list[str]:
        return [path.split("/")[-2] for path, _ in self.posts if path.endswith("/completed")]

    def active_invocations(self) -> list[str]:
        return [
            invocation_id
            for invocation_id, status in self.invocation_status.items()
            if status in {"queued", "running"}
        ]


async def wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


async def wait_for_adapter_idle(adapter: TheChatAdapter, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        tasks = list(getattr(adapter, "_session_tasks", {}).values())
        if not getattr(adapter, "_active_sessions", {}) and all(task.done() for task in tasks):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("adapter did not become idle before timeout")


def thechat_event(*, invocation_id: str, text: str, message_id: str) -> dict[str, Any]:
    return {
        "id": invocation_id,
        "invocationId": invocation_id,
        "chatId": CONVERSATION_ID,
        "chatType": "dm",
        "threadId": "thread-1",
        "text": text,
        "messageId": message_id,
        "sender": {"id": "user-1", "name": "Bruno"},
        "bot": {"id": "bot-1", "userId": "bot-user-1", "name": "Koda"},
        "conversation": {
            "id": CONVERSATION_ID,
            "type": "direct",
            "name": "Koda DM",
            "workspaceId": "workspace-1",
        },
    }


@pytest.mark.asyncio
async def test_thechat_queue_command_completes_ack_without_second_stuck_working_state() -> None:
    runner = make_runner(Platform.THECHAT)
    client = RecordingTheChatClient()
    adapter = TheChatAdapter(PlatformConfig(enabled=True, token="thechat-test-token"))
    cast(Any, adapter)._client = client
    adapter.config.typing_indicator = False
    adapter.set_message_handler(runner._handle_message)
    runner.adapters[Platform.THECHAT] = adapter

    first_turn_started = asyncio.Event()
    release_first_turn = asyncio.Event()
    handled_prompts: list[str] = []

    async def fake_agent_turn(event, _source, _session_key, _run_generation):
        handled_prompts.append(event.text)
        if len(handled_prompts) == 1:
            first_turn_started.set()
            await asyncio.wait_for(release_first_turn.wait(), timeout=2.0)
            return "response to message 1"
        return f"response to queued: {event.text}"

    runner._handle_message_with_agent = AsyncMock(side_effect=fake_agent_turn)

    client.invocation_status["inv-1"] = "running"
    await adapter._handle_platform_event(
        thechat_event(invocation_id="inv-1", text="message 1", message_id="msg-1")
    )
    await asyncio.wait_for(first_turn_started.wait(), timeout=2.0)

    client.invocation_status["inv-2"] = "running"
    await adapter._handle_platform_event(
        thechat_event(invocation_id="inv-2", text="/queue Message 2", message_id="msg-2")
    )

    await wait_until(lambda: "inv-2" in client.completed_invocations())

    assert client.active_invocations() == ["inv-1"]
    assert "inv-1" not in client.completed_invocations()
    assert handled_prompts == ["message 1"]

    queue_ack = next(
        payload
        for payload in client.message_payloads()
        if str(payload.get("content", "")).startswith("Queued for the next turn")
    )
    assert queue_ack["threadId"] == "thread-1"

    release_first_turn.set()
    await wait_for_adapter_idle(adapter)

    assert handled_prompts == ["message 1", "Message 2"]
    assert client.invocation_status == {"inv-1": "completed", "inv-2": "completed"}
    assert client.active_invocations() == []

    messages = client.message_payloads()
    first_response = next(
        payload for payload in messages if payload.get("content") == "response to message 1"
    )
    queued_response = next(
        payload
        for payload in messages
        if payload.get("content") == "response to queued: Message 2"
    )
    assert first_response["threadId"] == "thread-1"
    assert queued_response["threadId"] == "thread-1"
