"""TheChat proactive routing must not depend on Telegram's optional SDK."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from gateway.config import Platform
from tools.send_message_tool import send_message_tool
from tools.send_message_targets import _parse_target_ref
from tools.send_message_thechat import _send_thechat

def _run_async_immediately(coro):
    return asyncio.run(coro)


class TestParseTargetRefTheChat:
    """_parse_target_ref recognizes explicit TheChat chat targets."""

    def test_obsolete_thechat_composite_chat_key_is_not_explicit(self):
        target = "thechat:workspace:workspace-1:conversation:conversation-1:bot:bot-1"

        _chat_id, _thread_id, is_explicit = _parse_target_ref("thechat", target)

        assert is_explicit is False

    def test_thechat_conversation_uuid_is_explicit(self):
        target = "11111111-1111-4111-8111-111111111111"

        chat_id, thread_id, is_explicit = _parse_target_ref("thechat", target)

        assert chat_id == target
        assert thread_id is None
        assert is_explicit is True

    def test_thechat_name_still_requires_directory_resolution(self):
        assert _parse_target_ref("thechat", "#general")[2] is False
        assert _parse_target_ref("discord", "thechat:workspace:ws")[2] is False

    @pytest.mark.asyncio
    async def test_send_thechat_posts_current_conversation_uuid(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"messageId": "message-1"}

        class FakeAsyncClient:
            def __init__(self, *, base_url, headers, timeout):
                captured["base_url"] = base_url
                captured["headers"] = headers
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, path, json=None):
                captured["path"] = path
                captured["payload"] = json
                return FakeResponse()

        class FakeTimeout:
            def __init__(self, *args, **kwargs):
                captured["timeout_args"] = (args, kwargs)

        monkeypatch.setitem(
            sys.modules,
            "httpx",
            SimpleNamespace(AsyncClient=FakeAsyncClient, Timeout=FakeTimeout),
        )
        pconfig = SimpleNamespace(
            token="bot-token",
            extra={"base_url": "http://thechat.test"},
        )
        chat_id = "11111111-1111-4111-8111-111111111111"

        result = await _send_thechat(pconfig, chat_id, "cron update")

        assert result == {"success": True, "message_id": "message-1"}
        assert captured["base_url"] == "http://thechat.test"
        assert captured["headers"] == {"Authorization": "Bearer bot-token"}
        assert captured["path"] == "/hermes-platform/messages"
        assert captured["payload"] == {
            "conversationId": chat_id,
            "content": "cron update",
            "attachmentIds": [],
            "platformMessageId": captured["payload"]["platformMessageId"],
            "complete": False,
        }
        assert captured["payload"]["platformMessageId"].startswith("send-message-tool:")


def test_explicit_thechat_target_routes_without_directory_resolution():
    thechat_cfg = SimpleNamespace(
        enabled=True,
        token="bot-token",
        extra={"base_url": "http://thechat.test"},
    )
    config = SimpleNamespace(
        platforms={Platform.THECHAT: thechat_cfg},
        get_home_channel=lambda _platform: None,
    )
    chat_id = "11111111-1111-4111-8111-111111111111"

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.channel_directory.resolve_channel_name") as resolve_mock, \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": f"thechat:{chat_id}",
                    "message": "cron says hello",
                }
            )
        )

    assert result["success"] is True
    resolve_mock.assert_not_called()
    send_mock.assert_awaited_once()
    assert send_mock.await_args.args[0] == Platform.THECHAT
    assert send_mock.await_args.args[2] == chat_id


def test_cron_thechat_duplicate_home_target_is_skipped():
    chat_id = "11111111-1111-4111-8111-111111111111"
    thechat_cfg = SimpleNamespace(
        enabled=True,
        token="bot-token",
        extra={"base_url": "http://thechat.test"},
    )
    home = SimpleNamespace(chat_id=chat_id)
    config = SimpleNamespace(
        platforms={Platform.THECHAT: thechat_cfg},
        get_home_channel=lambda platform: home if platform == Platform.THECHAT else None,
    )

    with patch.dict(
        os.environ,
        {
            "HERMES_CRON_AUTO_DELIVER_PLATFORM": "thechat",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID": chat_id,
        },
        clear=False,
    ), \
         patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock:
        result = json.loads(
            send_message_tool(
                {
                    "action": "send",
                    "target": "thechat",
                    "message": "cron says hello",
                }
            )
        )

    assert result["success"] is True
    assert result["skipped"] is True
    assert result["reason"] == "cron_auto_delivery_duplicate_target"
    send_mock.assert_not_awaited()
