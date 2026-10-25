"""The preserved bundled provider prepares its SDK through the current PM owner."""
import sys
from types import SimpleNamespace

import pytest


def test_client_prepares_honcho_extra_before_constructing_sdk(monkeypatch):
    import pm
    from plugins.memory.honcho.client import HonchoClientConfig, _build_client

    calls = []
    monkeypatch.delitem(sys.modules, "honcho", raising=False)

    def ensure(extra):
        calls.append(extra)
        monkeypatch.setitem(sys.modules, "honcho", SimpleNamespace(Honcho=lambda **kw: kw))

    monkeypatch.setattr(pm, "ensure_import", ensure)
    result = _build_client(HonchoClientConfig(base_url="http://127.0.0.1:9", timeout=2))
    assert calls == ["honcho"]
    assert result["base_url"] == "http://127.0.0.1:9"
    assert result["api_key"] == "local"


def test_setup_uses_pm_and_surfaces_install_failure(monkeypatch, capsys):
    import pm
    from plugins.memory.honcho import cli

    monkeypatch.setitem(sys.modules, "honcho", None)
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "y")
    calls = []

    def sync(extras, *, explicit):
        calls.append((extras, explicit))
        raise pm.InstallError("venv", "synthetic offline mirror")

    monkeypatch.setattr(pm, "sync_venv", sync)
    assert cli._ensure_sdk_installed() is False
    assert calls == [(["honcho"], True)]
    assert "synthetic offline mirror" in capsys.readouterr().out
