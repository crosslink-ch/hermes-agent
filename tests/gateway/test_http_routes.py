from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.http_routes import loopback_route, route_manifest_path, write_route_manifest


def test_loopback_route_builds_prefix_route():
    assert loopback_route("api", path_prefix="/v1", port=8642) == {
        "id": "api",
        "upstream": "http://127.0.0.1:8642",
        "pathPrefix": "/v1",
    }


def test_loopback_route_normalizes_exact_path():
    assert loopback_route("telegram", path="telegram/webhook", port="8443") == {
        "id": "telegram",
        "upstream": "http://127.0.0.1:8443",
        "path": "/telegram/webhook",
    }


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_loopback_route_rejects_invalid_ports(port):
    with pytest.raises(ValueError):
        loopback_route("bad", path="/bad", port=port)


def test_route_manifest_path_honors_env(monkeypatch, tmp_path: Path):
    target = tmp_path / "routes.json"
    monkeypatch.setenv("HERMES_HTTP_ROUTES_PATH", str(target))

    assert route_manifest_path() == target


def test_write_route_manifest_validates_and_sorts_routes(tmp_path: Path):
    target = tmp_path / "runtime" / "http-routes.json"

    write_route_manifest(
        [
            loopback_route("telegram", path="/telegram/webhook", port=8443),
            loopback_route("api", path_prefix="/v1", port=8642),
        ],
        path=target,
    )

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert [route["id"] for route in data["routes"]] == ["api", "telegram"]
    assert data["routes"][0] == {
        "id": "api",
        "upstream": "http://127.0.0.1:8642",
        "pathPrefix": "/v1",
    }


def test_api_server_public_routes_include_v1_and_jobs():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    assert adapter.public_http_routes() == [
        {
            "id": "api-server-v1",
            "upstream": "http://127.0.0.1:8642",
            "pathPrefix": "/v1",
        },
        {
            "id": "api-server-jobs",
            "upstream": "http://127.0.0.1:8642",
            "pathPrefix": "/api/jobs",
        },
    ]


def test_write_route_manifest_rejects_non_loopback_upstream(tmp_path: Path):
    with pytest.raises(ValueError):
        write_route_manifest(
            [{"id": "bad", "path": "/bad", "upstream": "http://10.0.0.1:8642"}],
            path=tmp_path / "http-routes.json",
        )


def test_write_route_manifest_rejects_ambiguous_loopback_upstream(tmp_path: Path):
    with pytest.raises(ValueError):
        write_route_manifest(
            [{"id": "bad", "path": "/bad", "upstream": "http://127.0.0.1:8642.evil"}],
            path=tmp_path / "http-routes.json",
        )
