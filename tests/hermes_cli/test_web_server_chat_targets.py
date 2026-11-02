"""Behavior contracts split by scenario."""
from pathlib import Path
import hermes_cli.web_server_chat as _web_server_chat

def test_resolve_chat_argv_injects_gateway_ws_url(monkeypatch):
    import hermes_cli.main_tui_launch as tui_launch
    import hermes_cli.web_server as ws

    monkeypatch.setenv("PATH", "/run/current-system/sw/bin:/usr/bin")
    monkeypatch.setattr(
        tui_launch,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (["node", "fake-tui.js"], Path("/tmp")),
    )
    monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(ws.app.state, "bound_port", 9119, raising=False)

    _argv, _cwd, env = _web_server_chat._resolve_chat_argv()

    assert env is not None
    gateway_url = env.get("HERMES_TUI_GATEWAY_URL", "")
    assert gateway_url.startswith("ws://127.0.0.1:9119/api/ws?")
    assert "token=" in gateway_url


def test_resolve_chat_argv_scopes_terminal_bridge_to_child_profile(
    monkeypatch, tmp_path,
):
    import hermes_cli.config as config_mod
    import hermes_cli.main_tui_launch as tui_launch
    import hermes_cli.web_server_chat as ws
    import hermes_cli.web_server_profiles as profiles
    from hermes_constants import get_hermes_home

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(profiles, "_resolve_profile_dir", lambda _name: profile_home)
    monkeypatch.setattr(
        tui_launch,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (
            ["node", "fake-tui.js"],
            Path("/tmp"),
        ),
    )
    seen_homes = []

    def fake_apply(env, *, override=True):
        del override
        seen_homes.append(get_hermes_home())
        env["TERMINAL_ENV"] = "ssh"

    monkeypatch.setattr(config_mod, "apply_terminal_config_to_env", fake_apply)
    launch_home = tmp_path / "launch"
    launch_home.mkdir()
    (launch_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n  timeout: 999\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "999")

    _argv, _cwd, env = ws._resolve_chat_argv(profile="work")

    assert seen_homes == [profile_home]
    assert env["HERMES_HOME"] == str(profile_home)
    assert env["TERMINAL_ENV"] == "ssh"
    assert "TERMINAL_TIMEOUT" not in env

