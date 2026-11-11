"""The dashboard's Docker update guidance names this distribution's image, not upstream's."""

import pytest

import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_files as _web_server_files
import hermes_cli.web_server_gateway as _web_server_gateway


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    test_client = TestClient(app)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def test_docker_update_guidance_names_the_distribution_image(client, monkeypatch):
    def fail_spawn(*_args, **_kwargs):
        raise AssertionError("docker update guard should not spawn hermes update")

    # Bypass the managed-externally gate so we reach the docker install check.
    monkeypatch.setattr(_web_server_files, "_dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *_a, **_k: "docker")
    monkeypatch.setattr(_cfg_mod, "detect_install_method", lambda _root: "docker")
    monkeypatch.setattr(_web_server_gateway, "_spawn_hermes_action", fail_spawn)
    _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
    _web_server_gateway._ACTION_RESULTS.pop("hermes-update", None)

    resp = client.post("/api/hermes/update")

    assert resp.status_code == 200
    data = resp.json()
    assert data["error"] == "docker_update_unsupported"
    assert "docker pull crosslinkch/hermes-agent:latest" in data["message"]

    status = client.get("/api/actions/hermes-update/status")
    assert status.status_code == 200
    assert any("docker pull crosslinkch/hermes-agent:latest" in line for line in status.json()["lines"])
