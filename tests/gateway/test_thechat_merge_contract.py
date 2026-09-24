"""Contracts spanning the upstream decomposition and the fork-only transport."""
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _instantiate_builtin_adapter
from gateway.platforms.thechat import TheChatAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, SessionStore


def test_thechat_scoped_config_constructs_adapter_and_publishes_routes(monkeypatch, tmp_path):
    from gateway.config_env import _apply_env_overrides

    monkeypatch.setenv("THECHAT_BASE_URL", "https://thechat.example/")
    monkeypatch.setenv("THECHAT_BOT_TOKEN", "test-token")
    monkeypatch.setenv("THECHAT_WEBHOOK_URL", "https://runtime.example/thechat/webhook")
    monkeypatch.setenv("HERMES_HTTP_ROUTES_PATH", str(tmp_path / "routes.json"))
    config = GatewayConfig()
    _apply_env_overrides(config)
    adapter = _instantiate_builtin_adapter(Platform.THECHAT, config.platforms[Platform.THECHAT])
    assert isinstance(adapter, TheChatAdapter)
    assert adapter.base_url == "https://thechat.example"
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.THECHAT: adapter}
    runner._publish_http_route_manifest(force=True)
    assert json.loads((tmp_path / "routes.json").read_text())["routes"] == adapter.public_http_routes()
    runner.adapters.clear()
    runner._publish_http_route_manifest()
    assert json.loads((tmp_path / "routes.json").read_text())["routes"] == []


@pytest.mark.asyncio
async def test_thechat_branch_uses_real_async_session_database(tmp_path):
    from hermes_state import AsyncSessionDB, SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_db = AsyncSessionDB(db)
    runner._clear_session_boundary_security_state = lambda key: None
    runner._evict_cached_agent = lambda key: None
    source = SessionSource(platform=Platform.THECHAT, chat_id="conversation", user_id="owner", thread_id="parent")
    parent = store.get_or_create_session(source)
    db.create_session(session_id=parent.session_id, source="thechat")
    db.append_message(session_id=parent.session_id, role="user", content="Original request")
    branch_source = SessionSource(platform=Platform.THECHAT, chat_id="conversation", user_id="owner", thread_id="branch")
    current = store.get_or_create_session(branch_source)
    event = MessageEvent(text="Try an alternative", message_type=MessageType.TEXT, source=branch_source,
                         raw_message={"sessionIntent": {"type": "branch", "fromThreadId": "parent", "title": "Alternative"}})
    try:
        branched = await runner._apply_thechat_session_intent_async(event, branch_source, current)
        assert branched.session_id != parent.session_id
        assert db.get_session(branched.session_id)["parent_session_id"] == parent.session_id
        assert db.get_session_title(branched.session_id) == "Alternative"
        assert db.get_messages_as_conversation(branched.session_id)[0]["content"] == "Original request"
        assert store.get_session_for_source(branch_source).session_id == branched.session_id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_telegram_manifest_tracks_actual_webhook_listener(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram import adapter as telegram

    adapter = object.__new__(telegram.TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig()
    adapter._drop_pending_on_cold_boot = True
    adapter._app = SimpleNamespace(updater=SimpleNamespace(start_webhook=AsyncMock()))
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-webhook-secret")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_PORT", "18765")
    # The PTB network transport is stubbed, not required to test listener metadata.
    monkeypatch.setattr(telegram, "Update", SimpleNamespace(ALL_TYPES=["message"]), raising=False)
    await adapter._start_webhook_mode("https://example.test/custom/telegram", is_reconnect=False)
    route = adapter.public_http_routes()[0]
    listener = adapter._app.updater.start_webhook.await_args.kwargs
    assert route["path"] == listener["url_path"] == "/custom/telegram"
    assert route["upstream"] == f"http://127.0.0.1:{listener['port']}"
    assert listener["secret_token"] == "test-webhook-secret"
