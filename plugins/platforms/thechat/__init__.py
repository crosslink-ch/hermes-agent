"""Register the existing fork transport with upstream's platform-owned seams."""
from gateway.platforms.thechat import TheChatAdapter, check_thechat_requirements
from .gateway_env import AUTHZ, ENV_ENABLE_CREDENTIALS, ENV_OVERRIDES


def _target(ref):
    from gateway.platforms.send_targets import NOT_EXPLICIT
    from tools.send_message_targets import _THECHAT_CONVERSATION_RE
    match = _THECHAT_CONVERSATION_RE.fullmatch(ref.strip())
    return (match[1], None) if match else NOT_EXPLICIT


async def _send(config, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    from tools.send_message_thechat import _send_thechat, _send_thechat_media
    if media_files:
        return await _send_thechat_media(config, chat_id, message, [message], thread_id=thread_id,
                                        media_files=media_files, force_document=force_document)
    return await _send_thechat(config, chat_id, message, thread_id=thread_id)


def register(ctx):
    ctx.register_platform(
        name="thechat", label="TheChat", adapter_factory=TheChatAdapter,
        check_fn=check_thechat_requirements,
        is_connected=lambda config: bool(config.token and config.extra.get("base_url")),
        config_connected_fn=lambda config: bool(config.token and config.extra.get("base_url")),
        required_env=list(ENV_ENABLE_CREDENTIALS), env_enable_credentials=ENV_ENABLE_CREDENTIALS,
        env_overrides=ENV_OVERRIDES, **AUTHZ,
        cron_deliver_env_var="THECHAT_HOME_CHANNEL", target_ref_syntax_fn=_target,
        standalone_sender_fn=_send,
    )
