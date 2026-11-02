"""Behavior contracts split by scenario."""
import gc
import subprocess
import weakref
import pytest
from tools.environments import docker as docker_env
from tests.tools.docker_environment_fixtures import (
    _make_dummy_env,
    _mock_subprocess_run,
    _FakeThread,
)

def test_persistent_storage_path_uses_stable_storage_task_id(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    _mock_subprocess_run(monkeypatch)

    env = _make_dummy_env(
        persistent_filesystem=True,
        task_id="runtime-config-fingerprint",
        storage_task_id="stable-target-owner",
    )

    assert env._home_dir == str(
        tmp_path / "docker" / "stable-target-owner" / "home"
    )
    assert env._workspace_dir == str(
        tmp_path / "docker" / "stable-target-owner" / "workspace"
    )


def test_legacy_named_target_storage_is_migrated(monkeypatch, tmp_path):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    _mock_subprocess_run(monkeypatch)
    legacy = tmp_path / "docker" / "legacy-runtime"
    (legacy / "home").mkdir(parents=True)
    (legacy / "home" / "marker.txt").write_text("kept", encoding="utf-8")

    env = _make_dummy_env(
        persistent_filesystem=True,
        task_id="new-runtime",
        storage_task_id="stable-owner",
        legacy_storage_task_id="legacy-runtime",
    )

    migrated = tmp_path / "docker" / "stable-owner"
    assert not legacy.exists()
    assert (migrated / "home" / "marker.txt").read_text(encoding="utf-8") == "kept"
    assert env._home_dir == str(migrated / "home")


def test_persistent_storage_retires_prior_process_runtime_on_spec_change(
    monkeypatch,
):
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "version":
            return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
        if cmd[1] == "ps" and any(
            "hermes-storage-id=" in part for part in cmd
        ):
            output = "old-container\told-runtime\tprior-process\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1] == "run":
            return subprocess.CompletedProcess(cmd, 0, stdout="new-container\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env = _make_dummy_env(
        persistent_filesystem=True,
        task_id="new-runtime",
        storage_task_id="stable-owner",
    )

    removals = [cmd for cmd in calls if cmd[1:3] == ["rm", "-f"]]
    assert removals == [["/usr/bin/docker", "rm", "-f", "old-container"]]
    run_cmd = next(cmd for cmd in calls if cmd[1] == "run")
    assert env._labels["hermes-storage-id"].startswith("storage-")
    assert f"hermes-storage-id={env._labels['hermes-storage-id']}" in run_cmd
    assert any("hermes-process-instance=" in part for part in run_cmd)


def test_persistent_storage_refuses_to_remove_live_peer_runtime(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    env = object.__new__(docker_env.DockerEnvironment)
    env._docker_exe = "/usr/bin/docker"
    env._lease_root = tmp_path / "docker" / ".runtime-leases"
    container_id = "live-peer-container"
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout=f"{container_id}\told-runtime\tpeer-process\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    docker_env._hold_container_lease(container_id, env._lease_root)
    try:
        with pytest.raises(RuntimeError, match="still used by another Hermes process"):
            env._remove_superseded_storage_containers(
                "stable-owner", "default", "new-runtime",
                allow_exact_reuse=False,
            )
    finally:
        docker_env._release_container_lease(container_id, env._lease_root)

    assert not [cmd for cmd in calls if cmd[1:3] == ["rm", "-f"]]


def test_persistent_storage_retires_exact_runtime_when_egress_changes(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    env = object.__new__(docker_env.DockerEnvironment)
    env._docker_exe = "/usr/bin/docker"
    env._lease_root = tmp_path / "leases"
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout="stale-exact\tcurrent-runtime\tdead-process\told-egress\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env._remove_superseded_storage_containers(
        "stable-owner", "default", "current-runtime",
        current_egress_label="new-egress",
        allow_exact_reuse=True,
    )

    assert [cmd for cmd in calls if cmd[1:3] == ["rm", "-f"]] == [
        ["/usr/bin/docker", "rm", "-f", "stale-exact"],
    ]


def test_runtime_lease_releases_when_environment_handle_is_dropped(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)

    class Holder:
        pass

    holder = Holder()
    ref = weakref.ref(holder)
    container_id = "gc-release-container"
    docker_env._track_environment_container(holder, container_id)
    assert any(key[1] == container_id for key in docker_env._CONTAINER_LEASES)

    del holder
    gc.collect()

    assert ref() is None
    assert not any(key[1] == container_id for key in docker_env._CONTAINER_LEASES)
    assert container_id not in docker_env._CURRENT_PROCESS_CONTAINER_IDS


def test_retracking_environment_replaces_lease_ownership_without_leak(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)

    class Holder:
        pass

    holder = Holder()
    holder._lease_root = tmp_path / "leases"
    ref = weakref.ref(holder)
    container_id = "retracked-container"
    lease_key = (str(holder._lease_root.resolve()), container_id)

    docker_env._track_environment_container(holder, container_id)
    docker_env._track_environment_container(holder, container_id)

    assert docker_env._CONTAINER_LEASES[lease_key][1] == 1
    del holder
    gc.collect()
    assert ref() is None
    assert lease_key not in docker_env._CONTAINER_LEASES
    assert container_id not in docker_env._CURRENT_PROCESS_CONTAINER_IDS


def test_storage_owner_lock_serializes_runtime_creation(monkeypatch, tmp_path):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    first = docker_env._acquire_storage_creation_lease("stable-owner")
    try:
        with pytest.raises(RuntimeError, match="Timed out waiting"):
            docker_env._acquire_storage_creation_lease(
                "stable-owner", timeout=0.01,
            )
    finally:
        docker_env._close_storage_creation_lease(first)


def test_persistent_storage_reconciles_exact_runtime_when_reuse_is_disabled(
    monkeypatch, tmp_path,
):
    from tools.environments import base as environment_base

    monkeypatch.setattr(environment_base, "get_sandbox_dir", lambda: tmp_path)
    env = object.__new__(docker_env.DockerEnvironment)
    env._docker_exe = "/usr/bin/docker"
    env._lease_root = tmp_path / "docker" / ".runtime-leases"
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "ps":
            return subprocess.CompletedProcess(
                cmd, 0,
                stdout="stale-exact\tcurrent-runtime\tdead-process\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    env._remove_superseded_storage_containers(
        "stable-owner", "default", "current-runtime",
        allow_exact_reuse=False,
    )

    assert [cmd for cmd in calls if cmd[1:3] == ["rm", "-f"]] == [
        ["/usr/bin/docker", "rm", "-f", "stale-exact"],
    ]


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When persist_across_processes=false, cleanup() must docker stop AND
    docker rm so containers don't leak across hermes processes.

    Updated for issue #20561: the previous implementation used fire-and-forget
    ``subprocess.Popen("... &", shell=True)`` which raced with parent exit;
    the new implementation uses ``subprocess.run`` on a daemon thread with
    bounded timeouts. See test_cleanup_with_persist_disabled_stops_and_rms
    for the full behavior contract.
    """
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_env, "_get_active_profile_name", lambda: "default")
    _mock_subprocess_run(monkeypatch)
    # Run the worker thread synchronously so assertions can observe its work.
    import threading
    monkeypatch.setattr(threading, "Thread", _FakeThread)

    env = docker_env.DockerEnvironment(
        image="python:3.11", cwd="/root", timeout=60,
        task_id="ephemeral-task", persistent_filesystem=False,
        persist_across_processes=False,
    )
    container_id = env._container_id
    assert container_id

    # Capture cleanup-time docker calls (everything before this was init).
    cleanup_calls = []
    real_run = docker_env.subprocess.run

    def _capture(cmd, **kw):
        cleanup_calls.append((list(cmd) if isinstance(cmd, list) else cmd, kw))
        return real_run(cmd, **kw)

    monkeypatch.setattr(docker_env.subprocess, "run", _capture)
    env.cleanup()

    stops = [c for c in cleanup_calls if isinstance(c[0], list) and c[0][1:2] == ["stop"]]
    assert stops, f"cleanup() should docker stop {container_id}; got {cleanup_calls}"

