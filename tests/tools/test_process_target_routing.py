def test_spawn_via_env_forwards_selected_cwd():
    from tools.process_registry import ProcessRegistry

    calls = []

    class FakeEnvironment:
        # Simulate another session changing mutable shared-environment cwd.
        cwd = "/srv/other-session"

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "", "returncode": 1}

    registry = ProcessRegistry()
    session = registry.spawn_via_env(
        env=FakeEnvironment(),
        command="pwd",
        cwd="/srv/named-target",
        task_id="process-cwd",
        session_key="named-session",
        target="remote",
        backend="ssh",
    )

    assert session.exited is True
    assert session.cwd == "/srv/named-target"
    assert calls[0][1]["cwd"] == session.cwd
    assert calls[0][1]["cwd"] != FakeEnvironment.cwd
    assert calls[0][1]["rewrite_compound_background"] is False


def test_named_process_metadata_survives_checkpoint_and_status(tmp_path, monkeypatch):
    import time
    from tools.process_registry import ProcessRegistry, ProcessSession
    from tools import process_registry as module

    monkeypatch.setattr(module, "_checkpoint_path", lambda: tmp_path / "processes.json")
    registry = ProcessRegistry()
    env = object()
    session = ProcessSession(
        id="proc_named", command="sleep 10", task_id="task",
        started_at=time.time(), target="remote", backend="ssh",
        cwd="/srv/project", timeout_seconds=2,
        environment_task_key="owner", runtime_scope="scope-123", env_ref=env,
    )
    with registry._lock:
        registry._running[session.id] = session
    registry._write_checkpoint()
    assert registry.has_active_environment(env)
    assert registry.has_active_processes(("owner", "remote"))
    assert not registry.has_active_processes(("owner", "other"))
    assert registry.poll(session.id)["runtime_scope"] == "scope-123"
    assert registry.read_log(session.id)["target"] == "remote"
    assert registry.wait(session.id, timeout=0)["backend"] == "ssh"
    entries = registry.list_sessions(task_id="task")
    assert entries[0]["cwd"] == "/srv/project"
    payload = (tmp_path / "processes.json").read_text()
    assert '"timeout_seconds": 2' in payload
    assert '"runtime_scope": "scope-123"' in payload
