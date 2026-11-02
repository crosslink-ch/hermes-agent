"""Behavior contracts split by scenario."""
import json
import os
import time
from unittest.mock import patch
from tests.tools.process_registry_fixtures import _make_session

pytest_plugins = ["tests.tools.process_registry_fixtures"]

def test_wait_validation_error_preserves_execution_target_metadata(registry):
    session = _make_session(sid="proc_wait_invalid")
    session.target = "box"
    session.backend = "ssh"
    session.runtime_scope = "scope-server-1"
    registry._running[session.id] = session

    result = registry.wait(session.id, timeout=0)

    assert result == {
        "status": "error",
        "error": "timeout must be positive (got 0)",
        "target": "box",
        "backend": "ssh",
        "runtime_scope": "scope-server-1",
    }


def test_completion_notification_preserves_execution_target_metadata(registry):
    session = _make_session(sid="proc_notify_target", exited=True, exit_code=0)
    session.notify_on_complete = True
    session.target = "box"
    session.backend = "ssh"
    session.runtime_scope = "scope-server-1"
    registry._running[session.id] = session

    with patch.object(registry, "_write_checkpoint"):
        registry._move_to_finished(session)

    event = registry.completion_queue.get_nowait()
    assert event["target"] == "box"
    assert event["backend"] == "ssh"
    assert event["runtime_scope"] == "scope-server-1"


class TestCheckpoint:
    def test_write_checkpoint_includes_execution_target_metadata(self, registry, tmp_path):
        with patch("tools.process_registry.CHECKPOINT_PATH", tmp_path / "procs.json"):
            s = _make_session()
            s.target = "alpha"
            s.backend = "local"
            s.runtime_scope = "scope-v1"
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads((tmp_path / "procs.json").read_text())
            assert len(data) == 1
            assert data[0]["session_id"] == s.id
            assert data[0]["target"] == "alpha"
            assert data[0]["backend"] == "local"
            assert data[0]["runtime_scope"] == "scope-v1"

    def test_recover_dead_pid(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_dead",
            "command": "sleep 999",
            "pid": 999999999,  # almost certainly not running
            "task_id": "t1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0

    def test_recover_enqueues_watchers_with_execution_target_metadata(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        checkpoint.write_text(json.dumps([{
            "session_id": "proc_live",
            "command": "sleep 999",
            "pid": os.getpid(),  # current process — guaranteed alive
            "task_id": "t1",
            "session_key": "sk1",
            "watcher_platform": "telegram",
            "watcher_chat_id": "123",
            "watcher_user_id": "u123",
            "watcher_user_name": "alice",
            "watcher_thread_id": "42",
            "watcher_interval": 60,
            "target": "alpha",
            "backend": "local",
            "runtime_scope": "scope-v1",
        }]))
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 1
            assert len(registry.pending_watchers) == 1
            w = registry.pending_watchers[0]
            assert w["session_id"] == "proc_live"
            assert w["platform"] == "telegram"
            assert w["chat_id"] == "123"
            assert w["user_id"] == "u123"
            assert w["user_name"] == "alice"
            assert w["thread_id"] == "42"
            assert w["check_interval"] == 60
            session = registry.get("proc_live")
            assert session.target == "alpha"
            assert session.backend == "local"
            assert session.runtime_scope == "scope-v1"

    def test_recover_dead_wrapper_retries_unreaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=False
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == [entry]

    def test_recover_dead_wrapper_drops_reaped_systemd_scope(
        self, registry, tmp_path, monkeypatch
    ):
        checkpoint = tmp_path / "procs.json"
        entry = {
            "session_id": "proc_dead_scope",
            "command": "daemonize",
            "pid": 999999999,
            "pid_scope": "host",
            "host_start_time": 123.0,
            "systemd_unit": "hermes-worker-proc_dead_scope.scope",
        }
        checkpoint.write_text(json.dumps([entry]))
        monkeypatch.setattr(registry, "_host_pid_is_ours", lambda *_args: False)
        monkeypatch.setattr(registry, "_is_host_pid_alive", lambda *_args: False)

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint), patch(
            "tools.process_registry._stop_systemd_unit", return_value=True
        ) as stop_unit:
            assert registry.recover_from_checkpoint() == 0

        stop_unit.assert_called_once_with(entry["systemd_unit"])
        assert json.loads(checkpoint.read_text()) == []


    def test_recovery_skips_explicit_sandbox_backed_entries(self, registry, tmp_path):
        checkpoint = tmp_path / "procs.json"
        original = [{
            "session_id": "proc_remote",
            "command": "sleep 999",
            "pid": os.getpid(),
            "task_id": "t1",
            "pid_scope": "sandbox",
        }]
        checkpoint.write_text(json.dumps(original))

        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            recovered = registry.recover_from_checkpoint()
            assert recovered == 0
            assert registry.get("proc_remote") is None

            data = json.loads(checkpoint.read_text())
            assert data == []

    def test_checkpoint_redacts_command_with_inline_secret(self, registry, tmp_path):
        """Issue #77484: the checkpoint file persists raw commands; inline
        credentials (e.g. ``curl -H 'Authorization: Bearer sk-...'``) must be
        redacted before write. Recovery only uses command for display/logging
        (the process is already running), so masking is lossless."""
        checkpoint = tmp_path / "procs.json"
        with patch("tools.process_registry.CHECKPOINT_PATH", checkpoint):
            secret = "sk-secret1234567890"
            command = f"curl -H 'Authorization: Bearer {secret}' http://x"
            s = _make_session(sid="proc_secret", command=command)
            s.pid = 12345
            s.host_start_time = int(time.time())
            registry._running[s.id] = s
            registry._write_checkpoint()

            data = json.loads(checkpoint.read_text())
            assert data[0]["session_id"] == "proc_secret"
            assert secret not in data[0]["command"]
            assert data[0]["command"] != command
