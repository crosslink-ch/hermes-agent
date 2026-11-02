"""Shared test fixtures and builders; no collected tests."""
from io import StringIO
import subprocess
from tools.environments import docker as docker_env

def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.

    Pre-seeds the cgroup-limit probe cache to ``True`` so the throwaway probe
    container (a ``docker run ... sleep 0``) does not run and pollute the
    captured call list — these tests inspect the real sandbox-start ``run``.
    Tests that exercise the probe itself live in test_docker_cgroup_limits.py.
    """
    docker_env._cgroup_limits_ok = True
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        storage_task_id=kwargs.get("storage_task_id"),
        legacy_storage_task_id=kwargs.get("legacy_storage_task_id"),
        volumes=kwargs.get("volumes", []),
        forward_env=kwargs.get("forward_env"),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
        env=kwargs.get("env"),
        run_as_host_user=kwargs.get("run_as_host_user", False),
        extra_args=kwargs.get("extra_args", []),
        persist_across_processes=kwargs.get("persist_across_processes", True),
        shared_container_key=kwargs.get("shared_container_key", ""),
        shm_size=kwargs.get("shm_size", docker_env._DEFAULT_SHM_SIZE),
        snap_compat=kwargs.get("snap_compat", False),
    )


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode



class _FakeThread:
    """Stand-in for threading.Thread that captures target/args and calls
    target() synchronously when .start() runs, so cleanup behavior is
    observable without actually backgrounding subprocess calls."""

    def __init__(self, target=None, daemon=None, name=None):
        self._target = target
        self.daemon = daemon
        self.name = name
        self._done = False

    def start(self):
        if self._target is not None:
            self._target()
        self._done = True

    def is_alive(self):
        return not self._done

    def join(self, timeout=None):
        self._done = True
