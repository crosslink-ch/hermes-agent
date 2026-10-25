"""Crosslink source installs follow main independently of published release tags.

The official fork URL is redirected to a local origin. Release lookups are forbidden:
the independent release service does not migrate existing source subscriptions.
"""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = "https://github.com/crosslink-ch/hermes-agent.git"

pytestmark = pytest.mark.platforms("posix")


def _git(cwd, *args, env=None):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env,
    ).stdout.strip()


@pytest.mark.parametrize("side,start,args,tagged,lands", [
    (False, None, [], True, "tip"),
    (False, None, ["--channel", "main"], True, "tip"),
    (False, "before", [], True, "tip"),
    (False, "tip", [], True, "tip"),
    (False, None, [], False, "tip"),
    (False, "before", [], False, "tip"),
    (True, "tip", [], True, "tip"),
    (True, "tip", ["--channel", "stable"], True, "refused"),
    (True, "before", [], True, "tip"),
])
def test_crosslink_source_install_follows_branch(tmp_path, side, start, args, tagged, lands):
    origin, home, bin_dir = tmp_path / "origin", tmp_path / "home", tmp_path / "bin"
    install = home / ".hermes" / "hermes-agent"
    for d in (origin, home, bin_dir):
        d.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "before")
    if side:
        _git(origin, "checkout", "-q", "-b", "release/1.0")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "release")
    commits = {"before": _git(origin, "rev-parse", "HEAD~1"), "release": _git(origin, "rev-parse", "HEAD")}
    if tagged:
        for tag in ("crosslink-v2026.10.12", "crosslink-v1.0.0", "v99.0.0"):
            _git(origin, "tag", "-a", tag, "-m", "release")
    _git(origin, "checkout", "-q", "main")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "newer")
    commits["tip"] = _git(origin, "rev-parse", "HEAD")
    (home / ".gitconfig").write_text(f'[url "file://{origin}"]\n  insteadOf = {OFFICIAL}\n', encoding="utf-8")
    (bin_dir / "curl").write_text('#!/bin/sh\nprintf unexpected > "$HOME/release-request"\nexit 99\n', encoding="utf-8")
    (bin_dir / "curl").chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(home), "NO_COLOR": "1",
           "GIT_CONFIG_NOSYSTEM": "1"}
    if start:
        install.parent.mkdir(parents=True)
        _git(tmp_path, "clone", "-q", OFFICIAL, str(install), env=env)
        _git(install, "checkout", "-q", "--detach", commits[start])

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "install.sh"), "--stage", "repository", "--dir", str(install),
         "--hermes-home", str(home / ".hermes"), "--non-interactive", *args],
        check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120, env=env,
    )
    assert not (home / "release-request").exists(), result.stdout + result.stderr
    if lands == "refused":
        assert result.returncode != 0
        assert "Crosslink source installs currently support main only" in result.stderr
        assert _git(install, "rev-parse", "HEAD") == commits[start]
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(install, "rev-parse", "HEAD") == commits[lands], result.stdout + result.stderr
    assert _git(install, "status", "--porcelain") == "", result.stdout + result.stderr
