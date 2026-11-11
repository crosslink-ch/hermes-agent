"""send_message tests this fork keeps beside upstream's ``test_send_message_tool.py``.

Upstream folded these cases into table-driven tests when it pruned its suite; the fork still
carries them. They live here so the upstream file stays byte-identical to upstream and merges
cleanly. Nothing here needs python-telegram-bot, so they also run where that file skips.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.config import Platform
from tools.send_message_targets import _parse_target_ref
from tools.send_message_tool import send_message_tool


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    telegram_cfg = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: telegram_cfg},
        get_home_channel=lambda _platform: None,
    ), telegram_cfg


class TestSendMessageTool:
    def test_cron_duplicate_target_is_skipped_and_explained(self):
        home = SimpleNamespace(chat_id="-1001")
        config, _telegram_cfg = _make_config()
        config.get_home_channel = lambda _platform: home

        with patch.dict(
            os.environ,
            {
                "HERMES_CRON_AUTO_DELIVER_PLATFORM": "telegram",
                "HERMES_CRON_AUTO_DELIVER_CHAT_ID": "-1001",
            },
            clear=False,
        ), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        assert result["skipped"] is True
        assert result["reason"] == "cron_auto_delivery_duplicate_target"
        assert "final response" in result["note"]
        send_mock.assert_not_awaited()
        mirror_mock.assert_not_called()

    def test_resolved_telegram_topic_name_preserves_thread_id(self):
        config, telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.channel_directory.resolve_channel_name", return_value="-1001:17585"), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:Coaching Chat / topic 17585",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="17585",
            media_files=[],
            force_document=False,
        )

    def test_display_label_target_resolves_via_channel_directory(self, tmp_path):
        config, telegram_cfg = _make_config()
        cache_file = tmp_path / "channel_directory.json"
        cache_file.write_text(json.dumps({
            "updated_at": "2026-01-01T00:00:00",
            "platforms": {
                "telegram": [
                    {"id": "-1001:17585", "name": "Coaching Chat / topic 17585", "type": "group"}
                ]
            },
        }))

        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file), \
             patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:Coaching Chat / topic 17585 (group)",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.TELEGRAM,
            telegram_cfg,
            "-1001",
            "hello",
            thread_id="17585",
            media_files=[],
            force_document=False,
        )

    def test_resolved_slack_thread_name_preserves_thread_id(self):
        slack_cfg = SimpleNamespace(enabled=True, token="xoxb-test", extra={})
        config = SimpleNamespace(
            platforms={Platform.SLACK: slack_cfg},
            get_home_channel=lambda _platform: None,
        )

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.channel_directory.resolve_channel_name", return_value="C123ABCDEF:171.000001"), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "slack:ops / topic 171.000001",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.SLACK,
            slack_cfg,
            "C123ABCDEF",
            "hello",
            thread_id="171.000001",
            media_files=[],
            force_document=False,
        )

    def test_resolved_matrix_thread_name_preserves_thread_id(self):
        matrix_cfg = SimpleNamespace(
            enabled=True,
            token="tok",
            extra={"homeserver": "https://matrix.example.com"},
        )
        config = SimpleNamespace(
            platforms={Platform.MATRIX: matrix_cfg},
            get_home_channel=lambda _platform: None,
        )

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch(
                 "gateway.channel_directory.resolve_channel_name",
                 return_value="!roomid:matrix.example.org:$thread123:matrix.example.org",
             ), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True):
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "matrix:Ops / topic $thread123",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        send_mock.assert_awaited_once_with(
            Platform.MATRIX,
            matrix_cfg,
            "!roomid:matrix.example.org",
            "hello",
            thread_id="$thread123:matrix.example.org",
            media_files=[],
            force_document=False,
        )

    def test_mirror_receives_current_session_user_id(self):
        config, _telegram_cfg = _make_config()

        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("model_tools._run_async", side_effect=_run_async_immediately), \
             patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
             patch("gateway.session_context.get_session_env") as get_session_env_mock, \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror_mock:
            get_session_env_mock.side_effect = lambda name, default="": {
                "HERMES_SESSION_PLATFORM": "telegram",
                "HERMES_SESSION_USER_ID": "user-123",
            }.get(name, default)
            result = json.loads(
                send_message_tool(
                    {
                        "action": "send",
                        "target": "telegram:12345",
                        "message": "hello",
                    }
                )
            )

        assert result["success"] is True
        mirror_mock.assert_called_once_with(
            "telegram",
            "12345",
            "hello",
            source_label="telegram",
            thread_id=None,
            user_id="user-123",
        )


class TestParseTargetRef:
    def test_lid_jid_is_explicit(self):
        chat_id, _, is_explicit = _parse_target_ref(
            "whatsapp", "149606612619433@lid"
        )
        assert chat_id == "149606612619433@lid"
        assert is_explicit is True

    def test_broadcast_and_newsletter_jids_are_explicit(self):
        assert _parse_target_ref("whatsapp", "status@broadcast")[2] is True
        assert _parse_target_ref("whatsapp", "120363000000000000@newsletter")[2] is True

    def test_whatsapp_e164_still_explicit_alongside_jids(self):
        """The pre-existing '+'-prefixed E.164 path must keep working."""
        chat_id, _, is_explicit = _parse_target_ref("whatsapp", "+15551234567")
        assert chat_id == "+15551234567"
        assert is_explicit is True

    def test_jid_suffix_only_matches_whatsapp(self):
        """WhatsApp JID suffixes must NOT be treated as explicit elsewhere."""
        assert _parse_target_ref("telegram", "120363408391911677@g.us")[2] is False
        assert _parse_target_ref("signal", "149606612619433@lid")[2] is False

    def test_non_jid_whatsapp_target_falls_through(self):
        """A bare friendly name is not a JID — it must fall through to
        directory resolution (returns not-explicit so the caller can resolve)."""
        assert _parse_target_ref("whatsapp", "general")[2] is False


class TestParseTargetRefSlack:
    """_parse_target_ref recognizes Slack conversation and user targets."""

    def test_thread_target_is_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "C0B0QV5434G:171.000001")
        assert chat_id == "C0B0QV5434G"
        assert thread_id == "171.000001"
        assert is_explicit is True

    def test_public_channel_id_is_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "C0B0QV5434G")
        assert chat_id == "C0B0QV5434G"
        assert thread_id is None
        assert is_explicit is True

    def test_private_channel_id_is_explicit(self):
        assert _parse_target_ref("slack", "G123ABCDEF")[2] is True

    def test_dm_id_is_explicit(self):
        assert _parse_target_ref("slack", "D123ABCDEF")[2] is True

    def test_user_id_is_explicit_dm_target(self):
        """Slack user IDs are explicit user targets that must be opened as DMs."""
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "U123ABCDEF")
        assert chat_id == "user:U123ABCDEF"
        assert thread_id is None
        assert is_explicit is True

    def test_at_username_is_explicit_dm_target(self):
        """slack:@username targets resolve to DMs through users.list + conversations.open."""
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "@alice")
        assert chat_id == "user_name:alice"
        assert thread_id is None
        assert is_explicit is True

    def test_workspace_id_is_not_explicit(self):
        """Slack workspace IDs (W...) are not sendable conversation/user targets."""
        assert _parse_target_ref("slack", "W123ABCDEF")[2] is False

    def test_whitespace_is_stripped(self):
        chat_id, _, is_explicit = _parse_target_ref("slack", "  C0B0QV5434G  ")
        assert chat_id == "C0B0QV5434G"
        assert is_explicit is True

    def test_lowercase_or_short_id_is_not_explicit(self):
        assert _parse_target_ref("slack", "c0b0qv5434g")[2] is False
        assert _parse_target_ref("slack", "C123")[2] is False
        assert _parse_target_ref("slack", "X0B0QV5434G")[2] is False

    def test_slack_id_not_explicit_for_other_platforms(self):
        assert _parse_target_ref("discord", "C0B0QV5434G")[2] is False
        assert _parse_target_ref("telegram", "C0B0QV5434G")[2] is False




class TestParseTargetRefEmail:
    """_parse_target_ref recognizes email addresses as explicit for the email platform."""

    def test_standard_email_is_explicit(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("email", "user@example.com")
        assert chat_id == "user@example.com"
        assert thread_id is None
        assert is_explicit is True

    def test_email_with_dots_in_local_part(self):
        chat_id, _, is_explicit = _parse_target_ref("email", "first.last@example.co.uk")
        assert chat_id == "first.last@example.co.uk"
        assert is_explicit is True

    def test_email_with_plus_tag(self):
        chat_id, _, is_explicit = _parse_target_ref("email", "user+tag@gmail.com")
        assert chat_id == "user+tag@gmail.com"
        assert is_explicit is True

    def test_email_strips_whitespace(self):
        chat_id, _, is_explicit = _parse_target_ref("email", "  user@example.com  ")
        assert chat_id == "user@example.com"
        assert is_explicit is True

    def test_invalid_email_not_explicit(self):
        assert _parse_target_ref("email", "not-an-email")[2] is False
        assert _parse_target_ref("email", "@example.com")[2] is False
        assert _parse_target_ref("email", "user@")[2] is False
        assert _parse_target_ref("email", "user@.com")[2] is False

    def test_email_not_explicit_for_other_platforms(self):
        assert _parse_target_ref("telegram", "user@example.com")[2] is False
        assert _parse_target_ref("discord", "user@example.com")[2] is False
        assert _parse_target_ref("slack", "user@example.com")[2] is False
