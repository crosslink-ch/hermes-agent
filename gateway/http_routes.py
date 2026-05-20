"""Runtime HTTP route manifest for gateway-managed listeners.

Hermes adapters own their effective bind ports and public paths. Managed
deployments can expose a single reverse-proxy sidecar by watching this manifest
instead of trying to infer routes from static deployment config.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

DEFAULT_ROUTES_PATH = "runtime/http-routes.json"


def route_manifest_path() -> Path:
    raw = os.getenv("HERMES_HTTP_ROUTES_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    return get_hermes_home() / DEFAULT_ROUTES_PATH


def loopback_route(
    route_id: str,
    *,
    port: int,
    path: str | None = None,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    """Build a route to a gateway listener on localhost."""
    port_int = int(port)
    if not (0 < port_int < 65536):
        raise ValueError(f"invalid route port for {route_id}: {port}")
    route: dict[str, Any] = {
        "id": str(route_id),
        "upstream": f"http://127.0.0.1:{port_int}",
    }
    if path_prefix:
        route["pathPrefix"] = _normalize_path(path_prefix)
    elif path:
        route["path"] = _normalize_path(path)
    else:
        raise ValueError(f"route {route_id} needs path or path_prefix")
    return route


def write_route_manifest(
    routes: Iterable[dict[str, Any]],
    *,
    path: Path | None = None,
) -> Path:
    """Validate and atomically write the current public HTTP routes."""
    target = path or route_manifest_path()
    normalized = [_normalize_route(route) for route in routes]
    normalized.sort(key=lambda route: route["id"])
    atomic_json_write(
        target,
        {
            "version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "routes": normalized,
        },
        indent=2,
    )
    return target


def _normalize_route(route: dict[str, Any]) -> dict[str, Any]:
    route_id = str(route.get("id") or "").strip()
    if not route_id:
        raise ValueError("HTTP route is missing id")

    upstream = str(route.get("upstream") or "").strip()
    parsed_upstream = urlparse(upstream)
    try:
        upstream_port = parsed_upstream.port
    except ValueError:
        upstream_port = None
    if (
        parsed_upstream.scheme != "http"
        or parsed_upstream.hostname != "127.0.0.1"
        or upstream_port is None
        or parsed_upstream.path not in ("", "/")
        or parsed_upstream.params
        or parsed_upstream.query
        or parsed_upstream.fragment
    ):
        raise ValueError(f"HTTP route {route_id} upstream must target 127.0.0.1")

    normalized: dict[str, Any] = {
        "id": route_id,
        "upstream": f"http://127.0.0.1:{upstream_port}",
    }
    if route.get("pathPrefix"):
        normalized["pathPrefix"] = _normalize_path(str(route["pathPrefix"]))
    elif route.get("path"):
        normalized["path"] = _normalize_path(str(route["path"]))
    else:
        raise ValueError(f"HTTP route {route_id} is missing path/pathPrefix")
    return normalized


def _normalize_path(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        return "/"
    return text if text.startswith("/") else f"/{text}"
