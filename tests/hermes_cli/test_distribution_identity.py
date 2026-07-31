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
from hermes_cli.update_cmd import _is_fork


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
