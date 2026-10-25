"""Source subscriptions stay on the fork; release-service SemVer is a separate authority."""
import subprocess

import pytest

from hermes_cli import source_releases, update_channel


def test_unconfigured_distribution_follows_branch_without_release_lookup(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin",
                    "https://github.com/crosslink-ch/hermes-agent.git"], check=True)
    # Checkout-less fallback must own the same repository, not silently adopt Nous.
    assert source_releases.source_repository() == "crosslink-ch/hermes-agent"
    channel = update_channel.resolve_update_channel({}, root)
    assert channel == "main"
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda *a: pytest.fail("signed feed read"))
    target = source_releases.resolve_source_target(channel, ["git"], root)
    assert (target.repository, target.branch, target.commit) == ("crosslink-ch/hermes-agent", "main", None)


@pytest.mark.parametrize("channel", ["stable", "canary", "pm-preview"])
def test_source_release_migration_is_not_implicitly_enabled(channel, monkeypatch):
    calls = []
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda *a: calls.append(a))
    monkeypatch.setattr(source_releases, "_resolve_stable", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="Crosslink.*main"):
        source_releases.resolve_source_target(channel, repository="crosslink-ch/hermes-agent")
    assert calls == []
    if channel in {"stable", "canary"}:
        assert source_releases.resolve_source_release(
            channel, repository="crosslink-ch/hermes-agent") is None
    assert calls == []
