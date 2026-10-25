"""TheChat's old public configuration still reaches upstream's registry-owned consumers."""
from gateway.config import GatewayConfig, Platform


def test_configured_thechat_reaches_delivery_and_authorization(monkeypatch):
    from gateway.config_env import _apply_env_overrides
    from gateway.platform_env import authz_field
    from gateway.platform_registry import platform_registry
    from cron.scheduler_delivery import _is_known_delivery_platform, _resolve_home_env_var
    from tools.send_message_targets import resolve_send_target

    conversation = "12345678-1234-4234-8234-123456789abc"
    monkeypatch.setenv("THECHAT_BASE_URL", "https://thechat.example/")
    monkeypatch.setenv("THECHAT_BOT_TOKEN", "synthetic-token")
    monkeypatch.setenv("THECHAT_HOME_CHANNEL", conversation)
    config = GatewayConfig()
    _apply_env_overrides(config)
    platform = config.platforms[Platform.THECHAT]
    assert platform.enabled and platform.token == "synthetic-token"
    assert platform.extra["base_url"] == "https://thechat.example"
    assert platform.home_channel.chat_id == conversation
    assert _is_known_delivery_platform("thechat")
    assert _resolve_home_env_var("thechat") == "THECHAT_HOME_CHANNEL"
    entry = platform_registry.get("thechat")
    assert authz_field("thechat", "allowed_users_env", entry) == "THECHAT_ALLOWED_USERS"
    assert resolve_send_target("thechat", conversation) == (conversation, None, None)
