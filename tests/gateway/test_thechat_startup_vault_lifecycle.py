"""Startup admission is not completion of a TheChat unlock-capable turn."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import ProcessingOutcome
from gateway.platforms.thechat import TheChatAdapter
from gateway.run import GatewayRunner


class Client:
    def __init__(self):
        self.posts = []

    async def post(self, path, json):
        self.posts.append((path, json))


def scenario():
    adapter = TheChatAdapter(PlatformConfig(enabled=True, token="synthetic-token", extra={
        "base_url": "http://thechat.test", "webhook_url": "http://gateway.test/thechat/webhook"}))
    adapter._client = Client()
    context = {"invocation_id": "invocation", "conversation_id": "conversation", "thread_id": None}
    event = SimpleNamespace(message_id="message", source=SimpleNamespace(
        platform=Platform.THECHAT, chat_id="conversation", thread_id=None))
    adapter._event_contexts["message"] = context
    adapter._contexts["conversation"] = context
    runner = SimpleNamespace(_startup_restore_queue=[])
    GatewayRunner._queue_startup_restore_event(runner, event)
    return adapter, event, context, runner


@pytest.mark.asyncio
async def test_startup_queued_turn_keeps_unlock_context_until_real_replay_finishes():
    adapter, event, context, runner = scenario()
    assert runner._startup_restore_queue == [event]
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert adapter._client.posts == []
    assert adapter._event_contexts["message"] is context
    assert adapter._contexts["conversation"] is context
    event._hermes_startup_restore_replay = True
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert len(adapter._client.posts) == 1
    assert adapter._client.posts[0][0].endswith("/invocation/completed")
    assert "message" not in adapter._event_contexts


@pytest.mark.asyncio
async def test_startup_queued_cancellation_still_closes_the_invocation():
    adapter, event, _context, _runner = scenario()
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)
    assert len(adapter._client.posts) == 1
    assert adapter._client.posts[0][0].endswith("/invocation/cancelled")
    assert "message" not in adapter._event_contexts
