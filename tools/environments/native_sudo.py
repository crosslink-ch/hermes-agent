"""Local POSIX native sudo: authentication-only pipe, never shell-command stdin.

Both sudo processes are direct children of Hermes, preserving ppid timestamps.
No shell, snapshot or login rc sees the password; auth output is discarded at the
OS boundary, not redacted after capture. Other transports fail before capture.
"""
import subprocess
import time

from tools.terminal_tool_sudo import NativeSudoRefused, native_sudo_is_current

_AUTH_TIMEOUT = 15.0


def spawn(env, argv, cwd, stdin_data=None, *, authentication=False):
    from tools.environments.local import _make_run_env
    from tools.environments.base_output import _pipe_stdin

    if not native_sudo_is_current():
        raise NativeSudoRefused("Native sudo was cancelled; command not started.")
    proc = subprocess.Popen(
        argv, cwd=cwd, env=_make_run_env(env.env), start_new_session=True,
        stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
        stdout=subprocess.DEVNULL if authentication else subprocess.PIPE,
        stderr=subprocess.DEVNULL if authentication else subprocess.STDOUT,
        text=True, encoding="utf-8", errors="strict", close_fds=True,
    )
    proc._hermes_pgid = proc.pid
    if stdin_data is not None:
        _pipe_stdin(proc, stdin_data)
    return proc


def authenticate(env, argv, password, cwd, timeout):
    from tools.environments.base import (
        _enter_foreground_spawn, _leave_foreground_spawn, _live_foreground, _live_foreground_cond,
    )

    if not password or any(c in password for c in "\r\n\x00"):
        raise NativeSudoRefused("Native sudo was cancelled or the password is not a single line.")
    # Validate before spawning; a UnicodeEncodeError must never contain secret text in a tool result.
    try:
        password.encode("utf-8")
    except UnicodeError:
        raise NativeSudoRefused("Native sudo password is not valid UTF-8.") from None
    if not _enter_foreground_spawn():
        raise NativeSudoRefused("Native sudo was cancelled; command not started.")
    proc = None
    try:
        try:
            proc = spawn(env, argv, cwd, password + "\n", authentication=True)
        finally:
            fenced = _leave_foreground_spawn(env, proc)
        password = ""
        deadline = time.monotonic() + min(float(timeout), _AUTH_TIMEOUT)
        while not fenced and native_sudo_is_current() and time.monotonic() < deadline:
            if proc.poll() is not None:
                if proc.returncode == 0:
                    return
                break
            time.sleep(0.05)
        raise NativeSudoRefused("Native sudo authentication failed, timed out or was cancelled.")
    except (OSError, ValueError):
        raise NativeSudoRefused("Native sudo authentication could not be completed.") from None
    finally:
        password = ""
        if proc is not None:
            if proc.poll() is None:
                env._force_kill_process(proc)
            proc.wait()
            writer = getattr(proc, "_hermes_stdin_thread", None)
            if writer is not None:
                writer.join(timeout=2)
            with _live_foreground_cond:
                _live_foreground.pop(id(proc), None)
