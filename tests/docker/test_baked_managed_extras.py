"""Managed-tenant features must work in the sealed image without a lazy install.

A lazy install publishes a new dependency generation that an already running
gateway cannot adopt, so the first use of an unbaked feature fails with
"restart Hermes to activate it". These extras are the features KI Hub tenants
use, so the image must already satisfy them.
"""
from __future__ import annotations

import json
import subprocess

MANAGED_EXTRAS = ("honcho", "edge-tts", "firecrawl", "exa", "stt-whisper")


def test_managed_extras_need_no_lazy_install(built_image: str) -> None:
    probe = f"""
import json
from pm import extras
print(json.dumps({{'missing': {{name: extras.missing(name) for name in {MANAGED_EXTRAS!r}}}}}))
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", "hermes",
         "--entrypoint", "/opt/hermes/.venv/bin/python", built_image, "-c", probe],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["missing"] == {name: [] for name in MANAGED_EXTRAS}, report


def test_managed_sdks_import_as_runtime_user(built_image: str) -> None:
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", "hermes",
         "--entrypoint", "/opt/hermes/.venv/bin/python", built_image, "-c",
         "from honcho import Honcho; import edge_tts, firecrawl, exa_py, faster_whisper; print('ok')"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stdout + result.stderr
