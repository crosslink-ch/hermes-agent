"""Native masked sudo must never share credential stdin with a command.

No privilege changes: the executable models sudo's ppid timestamp and uses real
cat for the approved command, including cached authentication that reads nothing.
"""
import json
import os
import shlex

import pytest

from tools.environments.local import LocalEnvironment
from tools import terminal_tool_sudo as sudo


@pytest.fixture
def synthetic_sudo(tmp_path, monkeypatch):
    executable = tmp_path / "sudo"
    cache = tmp_path / "timestamp"
    calls = tmp_path / "calls.jsonl"
    executable.write_text(
        "#!/usr/bin/python3\n"
        "import json,os,pathlib,sys\n"
        f"cache=pathlib.Path({str(cache)!r}); calls=pathlib.Path({str(calls)!r})\n"
        "a=sys.argv[1:]\n"
        "with calls.open('a') as f: f.write(json.dumps(a)+'\\n')\n"
        "cached=cache.exists() and cache.read_text()==str(os.getppid())\n"
        "if '-v' in a:\n"
        " if not cached:\n"
        "  b=b''\n"
        "  while not b.endswith(b'\\n'):\n"
        "   c=os.read(0,1)\n"
        "   if not c: sys.exit(2)\n"
        "   b+=c\n"
        "  # Even a broken auth helper's echo is not terminal output.\n"
        "  os.write(1,b); os.write(2,b)\n"
        " cache.write_text(str(os.getppid())); sys.exit(0)\n"
        "if '-n' in a:\n"
        " if not cached: sys.exit(1)\n"
        " a=a[1:]\n"
        "elif '-S' in a:\n"
        " # Vulnerable legacy transport: cached sudo leaves all stdin to cat.\n"
        " a=a[3:]\n"
        "else: sys.exit(1)\n"
        "while a and a[0] in ('-u','--user','-g','--group'):\n"
        " a=a[2:]\n"
        "if a and a[0]=='--': a=a[1:]\n"
        "os.execv(a[0],a)\n"
    )
    executable.chmod(0o700)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(sudo, "_native_sudo_binary", lambda: str(executable), raising=False)
    monkeypatch.setattr(LocalEnvironment, "_sudo_nopasswd_works", lambda self: False)
    return executable, cache, calls


@pytest.mark.linux_only
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("options", [None, "", "-u root -g root", "--user root --group root --", "-u '#0' -g '#0'"])
def test_authentication_is_separate_from_command_stdin(tmp_path, monkeypatch, synthetic_sudo, cached, options):
    executable, cache, calls = synthetic_sudo
    if cached:
        cache.write_text(str(os.getpid()))
    secret = " synthetic unpatterned é password "
    caller_input = "caller data\nsecond line\n"
    env = LocalEnvironment(cwd=str(tmp_path))
    import subprocess
    original_popen = subprocess.Popen
    class CheckedPopen(original_popen):
        def __init__(self, argv, *args, **kwargs):
            assert secret not in json.dumps(argv, ensure_ascii=False)
            assert secret not in json.dumps(kwargs.get("env", {}), ensure_ascii=False)
            super().__init__(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", CheckedPopen)
    try:
        command = "/usr/bin/cat" if options is None else f"{shlex.quote(str(executable))} {options} /usr/bin/cat"
        with sudo.sudo_prompt_scope(lambda: secret):
            result = env.execute(command, stdin_data=caller_input)
        assert result["returncode"] == 0, result
        assert result["output"] == caller_input
        if options is None:
            assert not calls.exists()
        else:
            assert secret not in json.dumps(result) + calls.read_text()
            invocations = [json.loads(line) for line in calls.read_text().splitlines()]
            assert any("-v" in args for args in invocations)
            assert invocations[-1][0] == "-n"
        assert not sudo._sudo_password_cache
    finally:
        env.cleanup()


@pytest.mark.linux_only
@pytest.mark.parametrize("shape", [
    "{sudo} /usr/bin/true && {sudo} /usr/bin/cat",
    "/usr/bin/cat; {sudo} /usr/bin/true",
    "{sudo} -k /usr/bin/cat", "{sudo} -S /usr/bin/cat",
    "{sudo} --stdin /usr/bin/cat", "{sudo} -A /usr/bin/cat",
    "{sudo} -i", "{sudo} -u", "{sudo} -uroot /usr/bin/cat",
    "{sudo} -v", "env X=y {sudo} /usr/bin/cat",
    "{sudo} /usr/bin/cat < /dev/null",
    "{sudo} --user=root /usr/bin/cat", "{sudo} -u root -u other /usr/bin/cat",
    "{sudo} -- /usr/bin/cat $(printf injected)", "{sudo} -E /usr/bin/cat",
    "{sudo} # comment", "{sudo} -u #0 /usr/bin/cat",
])
def test_unsupported_shapes_refuse_before_capture_or_execution(tmp_path, synthetic_sudo, shape):
    executable, _, calls = synthetic_sudo
    prompted = []
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        with sudo.sudo_prompt_scope(lambda: prompted.append(True) or "synthetic-password"):
            result = env.execute(shape.format(sudo=shlex.quote(str(executable))), stdin_data="caller input\n")
        assert result["returncode"] != 0
        assert not prompted
        assert not calls.exists()
    finally:
        env.cleanup()


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", ["cancel-prompt", "cancel-auth", "timeout", "auth-failed", "expired", "multiline"])
def test_failed_or_cancelled_auth_never_hands_password_to_command(tmp_path, monkeypatch, failure):
    from tools.environments import native_sudo
    executable, started, executed = tmp_path / "sudo", tmp_path / "started", tmp_path / "executed"
    attempted = tmp_path / "attempted"
    executable.write_text(
        "#!/usr/bin/python3\nimport os,pathlib,signal,sys\n"
        "if '-v' in sys.argv:\n"
        f" pathlib.Path({str(started)!r}).touch()\n"
        f" if {failure!r} in ('cancel-auth','timeout'): signal.pause()\n"
        " b=sys.stdin.buffer.read(); os.write(1,b); os.write(2,b)\n"
        f" sys.exit(1 if {failure!r}=='auth-failed' else 0)\n"
        "assert sys.argv[1:3] == ['-n', '--']\n"
        f"pathlib.Path({str(attempted)!r}).touch()\n"
        f"if {failure!r}=='expired': sys.exit(1)\n"
        f"pathlib.Path({str(executed)!r}).touch()\n"
        "os.execv('/usr/bin/cat',['cat'])\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr(sudo, "_native_sudo_binary", lambda: str(executable))
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(native_sudo, "_AUTH_TIMEOUT", 2.0)
    secret = "synthetic-password" + ("\nextra" if failure == "multiline" else "")
    def prompt():
        return "" if failure == "cancel-prompt" else secret
    def current():
        return not (failure == "cancel-auth" and started.exists())
    env = LocalEnvironment(cwd=str(tmp_path))
    try:
        with sudo.sudo_prompt_scope(prompt, is_current=current):
            result = env.execute(f"{executable} /usr/bin/cat", stdin_data="caller data\n")
        assert result["returncode"] != 0
        assert not executed.exists()
        assert attempted.exists() == (failure == "expired")
        assert secret not in json.dumps(result)
        if failure in {"cancel-prompt", "multiline"}:
            assert not started.exists()
    finally:
        env.cleanup()


def test_legacy_configured_password_and_unsupported_native_transports(monkeypatch):
    from tools.environments.base import BaseEnvironment
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    with sudo.sudo_prompt_scope(lambda: pytest.fail("unsupported transport captured a credential")):
        with pytest.raises(sudo.NativeSudoRefused):
            sudo.prepare_native_sudo("sudo true", supported=False)
        assert sudo._transform_sudo_command("sudo true") == ("sudo true", None)
        monkeypatch.setenv("SUDO_PASSWORD", "legacy-configured")
        assert sudo.prepare_native_sudo("sudo true", supported=False) is None
        assert sudo._transform_sudo_command("sudo true") == ("sudo -S -p '' true", "legacy-configured\n")
        class Transport(BaseEnvironment):
            def cleanup(self):
                pass
        env = Transport(cwd="", timeout=10)
        assert env._prepare_command("sudo true") == ("sudo -S -p '' true", "legacy-configured\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("pty", [False, True])
async def test_native_background_is_refused_before_spawn(monkeypatch, pty):
    from tests.gateway.test_thechat_secure_input_consumers import prepare, run_tool
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    from tools import terminal_tool
    adapter = prepare(monkeypatch, {"password": "must-not-be-requested"})
    atomic_config_write(get_hermes_home() / "config.yaml", {"platforms": {"thechat": {"secure_input": {"allow_sudo": True}}}})
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(terminal_tool, "_check_all_guards_impl", lambda *_a, **_k: {"approved": True})
    spawned = []
    monkeypatch.setattr(terminal_tool, "spawn_background_process", lambda **kw: spawned.append(kw) or '{"success":true}')
    result = await run_tool(adapter, "terminal", {"command": "sudo true", "background": True, "pty": pty})
    assert not spawned and not adapter._client.posts
    assert result.get("error")
