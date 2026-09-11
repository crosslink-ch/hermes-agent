from __future__ import annotations

import pytest

from hermes_cli import banner
from hermes_cli.distribution import (
    INSTALLER_BASE_URL,
    RELEASE_URL_BASE,
    REPOSITORY_CANONICAL,
    REPOSITORY_HTTPS_URL,
    REPOSITORY_SSH_URL,
    canonical_github_remote,
)
from hermes_cli.update_cmd_git import _is_fork


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/crosslink-ch/hermes-agent.git",
        "https://github.com/crosslink-ch/hermes-agent/",
        "git@github.com:crosslink-ch/hermes-agent.git",
        "git@github.com:crosslink-ch/hermes-agent",
        "ssh://git@github.com/crosslink-ch/hermes-agent.git",
        "git@github.com:CrossLink-CH/Hermes-Agent.git",
    ],
)
def test_distribution_remote_forms_are_official(remote: str) -> None:
    assert canonical_github_remote(remote) == REPOSITORY_CANONICAL
    assert _is_fork(remote) is False


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/NousResearch/hermes-agent.git",
        "git@github.com:NousResearch/hermes-agent.git",
        "https://github.com/example/hermes-agent.git",
        "git@gitlab.com:crosslink-ch/hermes-agent.git",
    ],
)
def test_non_distribution_remotes_are_forks(remote: str) -> None:
    assert _is_fork(remote) is True


def test_missing_origin_preserves_existing_update_behavior() -> None:
    assert _is_fork(None) is False
    assert _is_fork("") is False


def test_distribution_endpoints_drive_banner_and_releases() -> None:
    assert REPOSITORY_HTTPS_URL == "https://github.com/crosslink-ch/hermes-agent.git"
    assert REPOSITORY_SSH_URL == "git@github.com:crosslink-ch/hermes-agent.git"
    assert INSTALLER_BASE_URL == "https://share.kihub.ch/hermes"
    assert (
        RELEASE_URL_BASE == "https://github.com/crosslink-ch/hermes-agent/releases/tag"
    )
    assert banner._UPSTREAM_REPO_URL == REPOSITORY_HTTPS_URL
    assert banner._OFFICIAL_REPO_CANONICAL == REPOSITORY_CANONICAL
    assert banner._RELEASE_URL_BASE == RELEASE_URL_BASE


def test_passive_compare_uses_distribution_repository(monkeypatch):
    import io
    import json
    import urllib.request

    seen = []

    def urlopen(request, **kwargs):
        seen.append(request.full_url)
        return io.BytesIO(json.dumps({"ahead_by": 2}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(banner, "_compare_payload_cache", {})
    current, target = "a" * 40, "b" * 40
    assert banner._github_compare_behind(current, target) == 2
    assert seen == [
        f"https://api.github.com/repos/crosslink-ch/hermes-agent/compare/{current}...{target}"
    ]


def test_banner_release_ignores_inherited_upstream_tags(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True,
            capture_output=True, text=True,
        )

    git("init", "-q")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "--allow-empty", "-q", "-m", "fork release")
    tag = "crosslink-v2026.7.1"
    git("tag", tag)
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "--allow-empty", "-q", "-m", "new upstream merge")
    git("tag", "v2026.9.11")
    monkeypatch.setattr(banner, "_latest_release_cache", None)
    assert banner.get_latest_release_tag(tmp_path) == (tag, f"{RELEASE_URL_BASE}/{tag}")
