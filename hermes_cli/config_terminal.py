"""Terminal config inheritance and the legacy child-environment bridge."""
from __future__ import annotations

from typing import Any, Optional

def _terminal_env_value(value: Any) -> str:
    from hermes_cli.config import json
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def _terminal_config_value_is_bridgeable(key: str, value: Any) -> bool:
    """Return whether a terminal config value owns its mirrored env var."""
    return not (key == "cwd" and str(value or "").strip() in {".", "auto", "cwd"})


def terminal_config_owned_env_vars(terminal_config: Any) -> set[str]:
    """Return env vars explicitly owned by a raw ``terminal`` config section."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, _terminal_config_value_is_bridgeable
    if not isinstance(terminal_config, dict):
        return set()
    return {
        env_var
        for key, env_var in TERMINAL_CONFIG_ENV_MAP.items()
        if key in terminal_config
        and _terminal_config_value_is_bridgeable(key, terminal_config[key])}


def effective_terminal_config(terminal_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the flat settings represented by the configured default target.

    Legacy/runtime-adjacent consumers still use ``TERMINAL_*`` variables to
    describe the environment selected when a tool call omits ``target``. In
    named-target mode those variables must mirror ``default_target`` rather
    than the top-level inheritance defaults.

    Invalid target configuration remains the strict resolver's responsibility;
    this compatibility bridge stays fail-open so config loading retains its
    historical behavior.
    """
    if not isinstance(terminal_cfg, dict):
        return {}
    targets = terminal_cfg.get("targets")
    default_target = terminal_cfg.get("default_target")
    flat = {
        key: value for key, value in terminal_cfg.items()
        if key not in {"targets", "default_target"}
    }
    if not isinstance(targets, dict) or not targets:
        return dict(terminal_cfg)
    if not isinstance(default_target, str):
        return flat
    selected = targets.get(default_target)
    if not isinstance(selected, dict):
        return flat
    effective = dict(flat)
    effective.update(selected)
    return effective


def terminal_config_env_var_for_key(key: str) -> Optional[str]:
    """Return the env var mirrored by a ``terminal.*`` config key."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    return TERMINAL_CONFIG_ENV_MAP.get(key[len("terminal."):]) if key.startswith("terminal.") else None


def _is_ssh_remote_tilde_cwd(backend: str, cwd: str) -> bool:
    """Whether the remote SSH shell must expand *cwd* itself: ``~`` expanded on the Hermes host
    would name the host/container home instead of the SSH user's."""
    return (backend or "").strip().lower() == "ssh" and (cwd == "~" or cwd.startswith("~/"))


def apply_terminal_config_to_env(
    *, env: Optional[dict[str, str]] = None, config: Optional[dict[str, Any]] = None,
    override: Optional[bool] = None) -> dict[str, str]:
    """Bridge ``terminal.*`` config into the env vars terminal tools read.
    ``tools.terminal_tool`` is environment-driven because it also runs in child processes (TUI,
    dashboard PTY, gateway workers); this gives those launch paths the same bridge as the CLI
    without importing ``cli.py``. Explicit keys in the user's raw ``terminal`` section override
    matching env values; merged defaults only backfill missing env vars."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, _is_ssh_remote_tilde_cwd, _terminal_config_value_is_bridgeable, _terminal_env_value, effective_terminal_config, load_config_readonly, os, read_raw_config
    target = os.environ if env is None else env

    raw_terminal_cfg = read_raw_config().get("terminal")
    file_has_terminal_config = isinstance(raw_terminal_cfg, dict)
    raw_terminal_cfg = raw_terminal_cfg if file_has_terminal_config else {}
    should_override = file_has_terminal_config if override is None else override

    cfg = config if config is not None else load_config_readonly()
    terminal_cfg = cfg.get("terminal", {}) if isinstance(cfg, dict) else {}
    if not isinstance(terminal_cfg, dict):
        return target
    terminal_cfg = effective_terminal_config(terminal_cfg)
    terminal_backend = str(
        terminal_cfg.get("backend") or terminal_cfg.get("env_type") or "local"
    ).strip().lower()

    # A caller-supplied config is its own source of explicit keys; otherwise only keys present
    # in raw config.yaml may override existing env values (DEFAULT_CONFIG keys are backfill-only).
    raw_effective = effective_terminal_config(raw_terminal_cfg)
    explicit_keys = terminal_cfg.keys() if config is not None else raw_effective.keys()
    backend_sources = (terminal_cfg.get("backend"), target.get("TERMINAL_ENV"))
    if not (config is not None or "backend" in raw_effective):
        backend_sources = backend_sources[::-1]  # env wins when the file did not set backend
    terminal_backend = str(backend_sources[0] or backend_sources[1] or "")
    # Whether docker_image is the user's choice (config.yaml key, or TERMINAL_DOCKER_IMAGE set before
    # any bridge ran) or the shipped default. DockerEnvironment recreates a persisted container on
    # image mismatch only for a pinned image; a default flip keeps the user's sandbox and asks.
    # Children inherit both vars, so a launcher's verdict is kept unless the file pins it.
    if should_override and "docker_image" in explicit_keys:
        target["TERMINAL_DOCKER_IMAGE_PINNED"] = "1"
    elif "TERMINAL_DOCKER_IMAGE_PINNED" not in target:
        target["TERMINAL_DOCKER_IMAGE_PINNED"] = "1" if "TERMINAL_DOCKER_IMAGE" in target else "0"

    for cfg_key, env_var in TERMINAL_CONFIG_ENV_MAP.items():
        if cfg_key not in terminal_cfg:
            continue
        value = terminal_cfg[cfg_key]
        if not _terminal_config_value_is_bridgeable(cfg_key, value):
            continue
        if cfg_key == "cwd":
            raw_cwd = str(value or "").strip()
            if isinstance(value, str) and not _is_ssh_remote_tilde_cwd(terminal_backend, raw_cwd):
                value = os.path.expanduser(value)
        if (should_override and cfg_key in explicit_keys) or env_var not in target:
            target[env_var] = _terminal_env_value(value)
    return target

