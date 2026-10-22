"""Isolation/revocation tests use real turn scopes and native consumers."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.gateway.test_thechat_secure_input_consumers import prepare, run_tool, ReplyClient
from tests.gateway.test_thechat_vault_unlock import _Response, _turn, _context


@pytest.mark.asyncio
async def test_profile_a_b_a_skill_capture_uses_own_config_env_and_scope(tmp_path, monkeypatch):
    from agent.secret_scope import get_secret, is_multiplex_active, set_multiplex_active
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import atomic_config_write, load_env
    from tools import skills_tool
    monkeypatch.delenv("SYNTHETIC_PROFILE_KEY", raising=False)
    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        home.mkdir()
        atomic_config_write(home / "config.yaml", {"platforms": {"thechat": {"secure_input": {"allow_skill_secrets": True}}}})
        skill = home / "skills" / "synthetic-profile"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text('---\nname: synthetic-profile\ndescription: Test\nrequired_environment_variables:\n  - SYNTHETIC_PROFILE_KEY\n---\nTest.\n')
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", lambda *_: pytest.fail("global borrowed"))
    adapter = prepare(monkeypatch, {"value": "profile-A é"})
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        profiles = []
        for home, value in ((homes[0], "profile-A é"), (homes[1], "profile-B é"), (homes[0], "profile-A é")):
            adapter._client.values = {"value": value}
            with _profile_runtime_scope(home):
                result = await run_tool(adapter, "skill_view", {"name": "synthetic-profile"})
                assert result["readiness_status"] == "available", result
                assert load_env()["SYNTHETIC_PROFILE_KEY"] == get_secret("SYNTHETIC_PROFILE_KEY") == value
            profiles.append(home)
        requests = [p["payload"] for _, p in adapter._client.posts if p["type"] == "secure.input.request"]
        assert len(requests) == 2 and requests[0]["profileId"] != requests[1]["profileId"]
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, False, "false", "true", 1, [], {}])
async def test_optins_are_literal_profile_booleans_not_truthiness(raw, monkeypatch):
    from gateway.run_turn_runner_vault import vault_turn_scope
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    from tools.skills_tool_setup import _capture_required_environment_variables
    from tools import skills_tool, terminal_tool as terminal
    from tools.terminal_tool_sudo import _transform_sudo_command
    adapter = prepare(monkeypatch, {"value": "must-not-be-requested"})
    atomic_config_write(get_hermes_home() / "config.yaml", {"platforms": {"thechat": {"secure_input": {
        "allow_sudo": raw, "allow_skill_secrets": raw}}}})
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")  # inherited Desktop state must NOT enable a gateway prompt
    callback = lambda *_: pytest.fail("borrowed ambient callback")
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", callback)
    terminal.set_sudo_password_callback(callback)
    turn = _turn(adapter, _context(), type("Agent", (), {"is_interrupted": False})())
    try:
        with vault_turn_scope(turn):
            result = _capture_required_environment_variables("test", [{"name": "SYNTHETIC_KEY", "prompt": "ignored"}])
            assert result["missing_names"] == ["SYNTHETIC_KEY"]
            assert _transform_sudo_command("sudo synthetic") == ("sudo synthetic", None)
        assert terminal._get_sudo_password_callback() is callback
        assert not adapter._client.posts
    finally:
        terminal.set_sudo_password_callback(None)


@pytest.mark.asyncio
async def test_simultaneous_profile_turns_do_not_share_secret_callbacks(tmp_path, monkeypatch):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active, load_env_file
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import atomic_config_write
    from tools import skills_tool
    monkeypatch.delenv("SYNTHETIC_CONCURRENT_KEY", raising=False)
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", lambda *_: pytest.fail("process-global callback"))
    homes = [tmp_path / "A", tmp_path / "B"]
    adapters = [prepare(monkeypatch, {"value": value}) for value in ("synthetic-A", "synthetic-B")]
    for home in homes:
        home.mkdir()
        atomic_config_write(home / "config.yaml", {"platforms": {"thechat": {"secure_input": {"allow_skill_secrets": True}}}})
        skill = home / "skills" / "synthetic"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text('---\nname: synthetic\ndescription: Test\nrequired_environment_variables:\n  - SYNTHETIC_CONCURRENT_KEY\n---\nTest.\n')
    async def run(home, adapter):
        with _profile_runtime_scope(home):
            result = await run_tool(adapter, "skill_view", {"name": "synthetic"})
            assert result["readiness_status"] == "available"
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        await asyncio.gather(*(run(home, adapter) for home, adapter in zip(homes, adapters)))
        assert [load_env_file(home / ".env")["SYNTHETIC_CONCURRENT_KEY"] for home in homes] == ["synthetic-A", "synthetic-B"]
        assert all(not adapter._secure_input._waiters for adapter in adapters)
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_revoked_during_resolution_never_reaches_code_sink(monkeypatch):
    from tools import browser_vault_tool as tool
    adapter = prepare(monkeypatch, {"code": "synthetic-code"})
    class Client(ReplyClient):
        async def post(self, path, json):
            response = await super().post(path, json)
            if json.get("type") == "secure.input.resolved":
                self.adapter._event_contexts.clear()  # invocation finalized after decryption, before callback returns
            return response
    adapter._client = Client(adapter, {"code": "synthetic-code"})
    monkeypatch.setattr(tool, "_eval_js", lambda *_: {"success": True, "result": json.dumps([
        {"tag": "input", "type": "text", "name": "otp", "id": "otp", "autocomplete": "one-time-code", "visible": True}])})
    monkeypatch.setattr(tool, "_eval_js_secret", lambda *_: pytest.fail("retired invocation reached secret sink"))
    result = await run_tool(adapter, "browser_vault_enter_code", {})
    assert not result["success"]
    assert not adapter._secure_input._waiters


def test_thread_propagation_installs_explicit_none_on_reused_worker():
    from agent.vault_backends import unlock
    from tools import terminal_tool
    from tools.thread_context import propagate_context_to_thread
    from tools.skills_tool_setup import secret_capture_scope, _capture_required_environment_variables
    from tools import skills_tool
    callback = lambda *_: pytest.fail("stale worker prompt")
    def seed():
        unlock.set_code_prompt_callback(callback)
        unlock.set_save_login_prompt_callback(callback)
        unlock.set_unlock_prompt_callback(callback)
        terminal_tool.set_sudo_password_callback(callback)
    def check():
        assert unlock.get_code_prompt_callback() is None
        assert unlock.get_save_login_prompt_callback() is None
        assert unlock.get_unlock_prompt_callback() is None
        assert terminal_tool._get_sudo_password_callback() is None
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(seed).result()
        pool.submit(propagate_context_to_thread(check)).result()


def test_unscoped_multiplex_sudo_never_borrows_launch_secret(monkeypatch):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active, set_secret_scope, reset_secret_scope
    from tools.terminal_tool_sudo import _transform_sudo_command, sudo_prompt_scope
    monkeypatch.setenv("SUDO_PASSWORD", "launch-profile-secret")
    previous = is_multiplex_active()
    token = set_secret_scope(None)
    set_multiplex_active(True)
    try:
        with sudo_prompt_scope(None):
            assert _transform_sudo_command("sudo synthetic") == ("sudo synthetic", None)
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(token)
