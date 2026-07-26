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


def test_crosslink_publish_pushes_only_fork_ref_and_selected_tag(
    monkeypatch, tmp_path
):
    module = _load_release_module()
    tag_name = "crosslink-v2026.7.26"
    git_calls: list[tuple[str, ...]] = []
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "next_available_tag",
        lambda base_tag, _prefix: (base_tag, "2026.7.26"),
    )
    monkeypatch.setattr(module, "get_current_version", lambda: "0.18.0")
    monkeypatch.setattr(module, "get_last_tag", lambda _glob: None)
    monkeypatch.setattr(
        module,
        "get_commits",
        lambda since_tag=None: [{"github_author": "ribaricplusplus"}],
    )
    monkeypatch.setattr(module, "generate_changelog", lambda *args, **kwargs: "notes")

    def fake_git_result(*args, **_kwargs):
        git_calls.append(tuple(args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_subprocess_run(command, **_kwargs):
        subprocess_calls.append(list(command))
        return SimpleNamespace(
            returncode=0,
            stdout="https://github.com/crosslink-ch/hermes-agent/releases/tag/"
            + tag_name,
            stderr="",
        )

    monkeypatch.setattr(module, "git_result", fake_git_result)
    monkeypatch.setattr(module.shutil, "which", lambda _command: "/usr/bin/gh")
    monkeypatch.setattr(module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "release.py",
            "--target",
            "crosslink",
            "--date",
            "2026.7.26",
            "--first-release",
            "--publish",
        ],
    )

    module.main()

    assert ("push", "crosslink-ch", "HEAD:main") in git_calls
    assert ("push", "crosslink-ch", tag_name) in git_calls
    assert not any("--tags" in call for call in git_calls)
    assert len(subprocess_calls) == 1
    gh_command = subprocess_calls[0]
    assert gh_command[:4] == ["gh", "release", "create", tag_name]
    assert gh_command[gh_command.index("--repo") + 1] == "crosslink-ch/hermes-agent"
    assert "--latest" in gh_command
    assert not any(argument.startswith("dist/") for argument in gh_command)
