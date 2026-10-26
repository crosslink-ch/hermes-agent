"""Actual executor -> tool worker -> masked relay -> native consumer paths."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.gateway.test_thechat_secure_input import relay
from tests.gateway.test_thechat_vault_unlock import _adapter, _context, _turn, _signed, _Client, _Response


class ReplyClient(_Client):
    def __init__(self, adapter, values, *, before_reply=None):
        super().__init__()
        self.adapter, self.values, self.before_reply = adapter, values, before_reply

    async def post(self, path, json):
        self.posts.append((path, json))
        if json.get("type") == "secure.input.request":
            record = self.adapter._secure_input.pending[json["payload"]["requestId"]]
            if self.before_reply:
                self.before_reply()
            response = await self.adapter._handle_webhook(_signed(self.adapter, relay(record, self.values)))
            assert response.status == 200, response.text
        return _Response({"event": {"id": "progress-event"}})


async def run_tool(adapter, name, args, *, sender="owner", check=None):
    from tools.registry import registry
    from tools.thread_context import propagate_context_to_thread
    class Agent:
        is_interrupted = False
        def run_conversation(self, *_a, **_kw):
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(propagate_context_to_thread(lambda: json.loads(registry.dispatch(name, args)))).result()
            if check:
                check(result)
            return result
    agent = Agent()
    ctx = _context()
    ctx["requester_user_id"] = sender
    turn = _turn(adapter, ctx, agent, sender=sender)
    return await asyncio.to_thread(turn._run_conversation_with_approval, agent, [], None, None, None)


def prepare(monkeypatch, values):
    from tools import browser_vault_tool as tool
    adapter = _adapter()
    adapter._owner_user_id = "owner"
    adapter._capture_secure_input_capability({"capabilities": {"secureInput": 3}})
    adapter._client = ReplyClient(adapter, values)
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.setattr(tool, "_focus_bound_origin", lambda *_: "https://example.test")
    monkeypatch.setattr(tool, "_current_page_origin", lambda *_: "https://example.test")
    return adapter


@pytest.mark.asyncio
async def test_requester_code_exact_recovery_text_reaches_origin_bound_fill(monkeypatch):
    from tools import browser_vault_tool as tool
    code = " aB-cDé 42 "
    adapter = prepare(monkeypatch, {"code": code})
    scripts = []
    monkeypatch.setattr(tool, "_eval_js", lambda *_: {"success": True, "result": json.dumps([
        {"tag": "input", "type": "text", "name": "otp", "id": "otp", "autocomplete": "one-time-code", "visible": True}])})
    monkeypatch.setattr(tool, "_eval_js_secret", lambda _, script: scripts.append(script) or {"success": True, "result": '{"filled":1}'})
    result = await run_tool(adapter, "browser_vault_enter_code", {}, sender="admitted-human")
    assert result["success"], result
    assert len(scripts) == 1 and json.dumps(code) in scripts[0]
    assert "https://example.test" in scripts[0]
    assert code not in json.dumps(result) + json.dumps(adapter._client.posts)
    assert [p["type"] for _, p in adapter._client.posts] == ["secure.input.request", "secure.input.resolved"]
    assert not adapter._secure_input.pending and not adapter._secure_input._waiters


@pytest.mark.asyncio
async def test_unsupported_v3_is_unavailable_not_user_cancellation(monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    adapter = prepare(monkeypatch, {})
    adapter._capture_secure_input_capability({"capabilities": {"secureInput": 2}})
    backend = OnePasswordLoginBackend()
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    unlock.lock("onepassword")
    result = await run_tool(adapter, "browser_vault_unlock", {"backend": "onepassword"})
    assert result["error_type"] == "unlock_unavailable"
    assert not adapter._client.posts


@pytest.mark.asyncio
@pytest.mark.linux_only
async def test_onepassword_native_child_and_owner_guard(tmp_path, monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_backends.onepassword import OnePasswordLoginBackend
    master = "  synthetic é master "
    adapter = prepare(monkeypatch, {"password": master})
    executable = tmp_path / "op"
    executable.write_text("#!/usr/bin/env python3\nimport sys\nassert sys.argv[1:] == ['signin', '--raw']\n"
                          f"assert sys.stdin.read() == {master + chr(10)!r}\nprint('synthetic-op-session')\n")
    executable.chmod(0o700)
    backend = OnePasswordLoginBackend({"binary_path": str(executable)})
    monkeypatch.setattr("agent.vault_backends.enabled_backends", lambda: [backend])
    unlock.lock("onepassword")
    try:
        result = await run_tool(adapter, "browser_vault_unlock", {"backend": "onepassword"})
        assert result["success"], result
        assert unlock.get_session_token("onepassword") == "synthetic-op-session"
        count = len(adapter._client.posts)
        denied = await run_tool(adapter, "browser_vault_unlock", {"backend": "onepassword"}, sender="not-owner")
        assert denied["error_type"] == "external_vault_access_denied"
        assert len(adapter._client.posts) == count
    finally:
        unlock.lock("onepassword")


@pytest.mark.asyncio
async def test_save_login_persists_exact_values_only_for_owner(monkeypatch):
    from agent.vault_store import get_vault_store
    from tools import browser_vault_tool as tool
    values = {"identifier": " synthetic@example.test ", "password": "  é synthetic password\n"}
    adapter = prepare(monkeypatch, values)
    monkeypatch.setattr(tool, "browser_vault_fill", lambda *_a, **_kw: '{"success":true}')
    result = await run_tool(adapter, "browser_vault_save_login", {})
    assert result["success"], result
    meta = get_vault_store().get_meta(result["handle"])
    assert meta.identifier == values["identifier"] and meta.origin == "https://example.test"
    assert get_vault_store().resolve_secret(meta.id)["password"] == values["password"]
    count = len(adapter._client.posts)
    denied = await run_tool(adapter, "browser_vault_save_login", {}, sender="not-owner")
    assert not denied["success"] and len(adapter._client.posts) == count
    assert len(get_vault_store().list_items()) == 1


def test_save_login_rechecks_owner_after_answer_before_persistence(monkeypatch):
    from agent.vault_backends import unlock
    from agent.vault_store import get_vault_store
    from tools import browser_vault_tool as tool
    allowed = [True]
    monkeypatch.setattr(tool, "_focus_bound_origin", lambda *_: "https://example.test")
    monkeypatch.setattr(tool, "_current_page_origin", lambda *_: "https://example.test")
    answer = {"identifier": "synthetic", "password": "synthetic-secret"}
    def prompt(*_):
        allowed[0] = False
        return answer
    unlock.set_unlock_prompt_callback(lambda *_: "")
    unlock.set_save_login_prompt_callback(prompt)
    try:
        with unlock.external_access_scope(lambda: allowed[0]):
            result = json.loads(tool.browser_vault_save_login())
        assert not result["success"]
        assert not get_vault_store().list_items()
        assert not answer
    finally:
        unlock.set_unlock_prompt_callback(None)
        unlock.set_save_login_prompt_callback(None)


@pytest.mark.asyncio
async def test_skill_secret_real_view_persists_profile_scoped_exact_value(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools import skills_tool
    from hermes_cli.config import load_env, atomic_config_write
    home = get_hermes_home()
    atomic_config_write(home / "config.yaml", {"platforms": {"thechat": {"secure_input": {"allow_skill_secrets": True}}}})
    skill = home / "skills" / "synthetic-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text('---\nname: synthetic-skill\ndescription: Test\nrequired_environment_variables:\n  - SYNTHETIC_MASKED_KEY\n---\nTest only.\n')
    value = "  é key\\quoted\nnext\rline'\" "
    adapter = prepare(monkeypatch, {"value": value})
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", lambda *_: pytest.fail("global callback borrowed"))
    result = await run_tool(adapter, "skill_view", {"name": "synthetic-skill"})
    assert result["readiness_status"] == "available", result
    assert load_env()["SYNTHETIC_MASKED_KEY"] == value
    assert value not in json.dumps(result) + json.dumps(adapter._client.posts)


@pytest.mark.asyncio
async def test_sudo_real_transform_ephemeral_and_opt_in(tmp_path, monkeypatch):
    import os
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    from tools import terminal_tool_sudo as sudo, terminal_tool as terminal
    value = "  synthetic é sudo "
    adapter = prepare(monkeypatch, {"password": value})
    home = get_hermes_home()
    atomic_config_write(home / "config.yaml", {"platforms": {"thechat": {"secure_input": {"allow_sudo": True}}}})
    executable = tmp_path / "sudo"
    executable.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                          "if '-v' in sys.argv:\n"
                          f" assert sys.stdin.readline() == {value + chr(10)!r}\n sys.exit(0)\n"
                          "assert sys.argv[1:3] == ['-n', '--']\n"
                          "os.execv(sys.argv[3],sys.argv[3:])\n")
    executable.chmod(0o700)
    monkeypatch.setattr(sudo, "_native_sudo_binary", lambda: str(executable))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TERMINAL_ENV", "local")
    guards, permit = [], [False]
    def approval(command, *_, **__):
        guards.append(command)
        return {"approved": permit[0], "reason": "Synthetic approval decision"}
    monkeypatch.setattr(terminal, "_check_all_guards_impl", approval)
    command = str(executable) + " /usr/bin/printf synthetic-command-complete"
    denied = await run_tool(adapter, "terminal", {"command": command, "workdir": str(tmp_path)})
    assert not adapter._client.posts and guards == [command]
    permit[0] = True
    result = await run_tool(adapter, "terminal", {"command": command, "workdir": str(tmp_path)})
    assert result["exit_code"] == 0 and "synthetic-command-complete" in result["output"], result
    assert guards == [command, command]
    assert not sudo._sudo_password_cache
    assert not (home / ".env").exists()
    count = len(adapter._client.posts)
    denied = await run_tool(adapter, "terminal", {"command": command, "workdir": str(tmp_path)}, sender="not-owner")
    assert denied["exit_code"] != 0 and len(adapter._client.posts) == count


def test_native_sudo_refuses_command_embedded_transport():
    from tools.terminal_tool_sudo import sudo_prompt_scope, _transform_sudo_command
    with sudo_prompt_scope(lambda: pytest.fail("secret would enter a command string")):
        assert _transform_sudo_command("sudo synthetic", allow_native_prompt=False) == ("sudo synthetic", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "result"])
async def test_code_sink_failures_never_echo_secret(monkeypatch, caplog, failure):
    from tools import browser_vault_tool as tool
    secret = "synthetic-sensitive-code"
    adapter = prepare(monkeypatch, {"code": secret})
    monkeypatch.setattr(tool, "_eval_js", lambda *_: {"success": True, "result": json.dumps([
        {"tag": "input", "type": "text", "name": "otp", "id": "otp", "autocomplete": "one-time-code", "visible": True}])})
    def fail(*_):
        if failure == "raise":
            raise RuntimeError(secret)
        return {"success": False, "error": secret}
    monkeypatch.setattr(tool, "_eval_js_secret", fail)
    result = await run_tool(adapter, "browser_vault_enter_code", {})
    assert not result["success"]
    assert secret not in json.dumps(result) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "expired", "failed", "disconnect"])
async def test_native_prompt_lifecycle_never_reaches_secret_sink(monkeypatch, outcome):
    from tools import browser_vault_tool as tool
    from tests.gateway.test_thechat_secure_input import relay
    adapter = prepare(monkeypatch, {})
    class Client(_Client):
        async def post(self, path, json):
            self.posts.append((path, json))
            if json.get("type") == "secure.input.request":
                record = adapter._secure_input.pending[json["payload"]["requestId"]]
                if outcome == "cancelled":
                    response = await adapter._handle_webhook(_signed(adapter, relay(record, action="cancel")))
                    assert response.status == 200
                elif outcome == "expired":
                    record.deadline = 0
                elif outcome == "failed":
                    raise RuntimeError("synthetic-private-transport-error")
                else:
                    adapter._secure_input.close()
            return _Response({"event": {"id": "progress-event"}})
    adapter._client = Client()
    monkeypatch.setattr(tool, "_eval_js", lambda *_: {"success": True, "result": json.dumps([
        {"tag": "input", "type": "text", "name": "otp", "id": "otp", "autocomplete": "one-time-code", "visible": True}])})
    monkeypatch.setattr(tool, "_eval_js_secret", lambda *_: pytest.fail("failed prompt reached browser"))
    result = await run_tool(adapter, "browser_vault_enter_code", {})
    assert not result["success"]
    assert not adapter._secure_input._waiters and not adapter._secure_input.pending
    events = [p for _, p in adapter._client.posts if p["type"] == "secure.input.resolved"]
    assert events[0]["payload"]["outcome"] == ("cancelled" if outcome == "disconnect" else outcome)


@pytest.mark.asyncio
async def test_ambiguous_publication_retries_one_immutable_prompt(monkeypatch):
    from tools import browser_vault_tool as tool
    adapter = prepare(monkeypatch, {"code": "synthetic-code"})
    class RetryClient(ReplyClient):
        first = True
        async def post(self, path, json):
            if json.get("type") == "secure.input.request" and self.first:
                self.first = False
                self.posts.append((path, json))
                raise TimeoutError("Synthetic transport timeout")
            return await super().post(path, json)
    adapter._client = RetryClient(adapter, {"code": "synthetic-code"})
    monkeypatch.setattr(tool, "_eval_js", lambda *_: {"success": True, "result": json.dumps([
        {"tag": "input", "type": "text", "name": "otp", "id": "otp", "autocomplete": "one-time-code", "visible": True}])})
    monkeypatch.setattr(tool, "_eval_js_secret", lambda *_: {"success": True, "result": '{"filled":1}'})
    result = await run_tool(adapter, "browser_vault_enter_code", {})
    assert result["success"]
    requests = [p for _, p in adapter._client.posts if p["type"] == "secure.input.request"]
    assert len(requests) == 2 and requests[0] == requests[1]
