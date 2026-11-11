"""Cron auto-delivery resolves TheChat's home channel from the profile's ``.env``."""

import os
from unittest.mock import MagicMock, patch

from tests.cron.test_scheduler import _run_owned_job


class TestRunJobSessionPersistence:
    def test_run_job_sets_auto_delivery_env_from_dotenv_thechat_home_channel(
        self, tmp_path, monkeypatch
    ):
        chat_id = "11111111-1111-4111-8111-111111111111"
        job = {
            "id": "test-job",
            "name": "test",
            "prompt": "hello",
            "deliver": "thechat",
        }
        fake_db = MagicMock()
        seen = {}

        (tmp_path / ".env").write_text(f"THECHAT_HOME_CHANNEL={chat_id}\n")
        monkeypatch.delenv("THECHAT_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_PLATFORM", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID", raising=False)
        monkeypatch.delenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID", raising=False)

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_conversation(self, *args, **kwargs):
                from gateway.session_context import get_session_env

                seen["platform"] = (
                    get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM") or None
                )
                seen["chat_id"] = (
                    get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID") or None
                )
                seen["thread_id"] = (
                    get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID") or None
                )
                return {"final_response": "ok"}

        with (
            patch("cron.scheduler._hermes_home", tmp_path),
            patch("cron.scheduler._preflight_job_config", return_value=None),
            patch("hermes_state_registry.acquire", return_value=fake_db),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "***",
                    "base_url": "https://example.invalid/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                },
            ),
            patch("run_agent.AIAgent", FakeAgent),
        ):
            success, output, final_response, error = _run_owned_job(job, tmp_path, fake_db)

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert "ok" in output
        assert seen == {
            "platform": "thechat",
            "chat_id": chat_id,
            "thread_id": None,
        }
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_PLATFORM") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_CHAT_ID") is None
        assert os.getenv("HERMES_CRON_AUTO_DELIVER_THREAD_ID") is None
        fake_db.close.assert_called_once()
