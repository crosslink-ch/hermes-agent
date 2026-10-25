"""TheChat's legacy configuration, declared for upstream's platform consumers."""
from gateway.config import Platform
from gateway.platform_env import EnvCred, INT

ENV_ENABLE_CREDENTIALS = ("THECHAT_BASE_URL", "THECHAT_BOT_TOKEN")
ENV_OVERRIDES = (
    EnvCred(Platform.THECHAT, ENV_ENABLE_CREDENTIALS, token="THECHAT_BOT_TOKEN",
            fixed=(("base_url", "THECHAT_BASE_URL", "", lambda value: value.strip().rstrip("/")),),
            optional=(("poll_interval", "THECHAT_POLL_INTERVAL", float),
                      ("webhook_host", "THECHAT_WEBHOOK_HOST"),
                      ("webhook_port", "THECHAT_WEBHOOK_PORT", INT),
                      ("webhook_path", "THECHAT_WEBHOOK_PATH"),
                      ("webhook_url", "THECHAT_WEBHOOK_URL")), home="THECHAT_HOME_CHANNEL"),
)
AUTHZ = {"allowed_users_env": "THECHAT_ALLOWED_USERS", "allow_all_env": "THECHAT_ALLOW_ALL_USERS"}
