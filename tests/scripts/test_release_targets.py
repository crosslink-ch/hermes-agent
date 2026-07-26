from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_release_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "release.py"
    spec = importlib.util.spec_from_file_location("hermes_release", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crosslink_release_target_uses_fork_defaults():
    module = _load_release_module()
    args = SimpleNamespace(
        target="crosslink",
        remote=None,
        repo=None,
        tag_prefix=None,
    )

    config = module.target_config(args)

    assert config["remote"] == "crosslink-ch"
    assert config["repo"] == "crosslink-ch/hermes-agent"
    assert config["repo_url"] == "https://github.com/crosslink-ch/hermes-agent"
    assert config["tag_prefix"] == "crosslink-v"
    assert config["tag_glob"] == "crosslink-v20*"
    assert config["push_ref"] == "HEAD:main"
    assert config["mark_latest"] is True
    assert (
        module.release_title(
            config,
            "0.17.0",
            "2026.7.1",
            "crosslink-v2026.7.1",
        )
        == "Crosslink Hermes v2026.7.1"
    )


def test_upstream_release_target_keeps_upstream_defaults():
    module = _load_release_module()
    args = SimpleNamespace(
        target="upstream",
        remote=None,
        repo=None,
        tag_prefix=None,
    )

    config = module.target_config(args)

    assert config["remote"] == "origin"
    assert config["repo"] == "NousResearch/hermes-agent"
    assert config["tag_prefix"] == "v"
    assert config["push_ref"] == "HEAD"
    assert config["mark_latest"] is False
