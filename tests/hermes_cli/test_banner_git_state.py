from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.distribution import REPOSITORY_SLUG, REPOSITORY_SSH_URL


def test_format_banner_version_label_on_upstream_main():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3")
    assert "local" not in value


def test_get_git_banner_state_reads_origin_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}


def test_check_via_local_git_ssh_fastpath_ahead_not_behind(tmp_path):
    """SSH fast path must not report an ahead (carried) HEAD as behind.

    A carried local commit means tip SHAs differ, but the fresh upstream tip
    is an ancestor of HEAD — that is "ahead", and reporting it as behind
    nudges the user into `hermes update`, which can wipe the carried work.
    """
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return REPOSITORY_SSH_URL
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40  # carried commit, differs from upstream tip
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_github_branch_tip", return_value="a" * 40) as tip,
        # merge-base --is-ancestor exits 0: upstream tip IS an ancestor of HEAD
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=0)) as git_run,
    ):
        behind = banner._check_via_local_git(repo_dir)

    tip.assert_called_once_with(REPOSITORY_SLUG, "main")
    assert [call.args[0] for call in git_run.call_args_list] == [
        ["git", "merge-base", "--is-ancestor", "a" * 40, "HEAD"]
    ]  # Only local ancestry inspection, never fetch or ls-remote over SSH.
    assert behind == 0


def test_check_via_local_git_ssh_fastpath_genuinely_behind(tmp_path):
    """SSH fast path reports the exact count (compare API) when behind."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return REPOSITORY_SSH_URL
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_github_branch_tip", return_value="a" * 40) as tip,
        # merge-base --is-ancestor exits 1: not an ancestor -> genuinely behind
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)) as git_run,
        patch.object(banner, "_github_compare_behind", return_value=3),
    ):
        behind = banner._check_via_local_git(repo_dir)

    tip.assert_called_once_with(REPOSITORY_SLUG, "main")
    assert [call.args[0] for call in git_run.call_args_list] == [
        ["git", "merge-base", "--is-ancestor", "a" * 40, "HEAD"]
    ]  # Only local ancestry inspection, never fetch or ls-remote over SSH.
    assert behind == 3


def test_check_via_local_git_ssh_fastpath_offline_keeps_sentinel(tmp_path):
    """Behind + compare API unreachable = honest no-count sentinel, never 1."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5, network=False):
        if args == ["remote", "get-url", "origin"]:
            return REPOSITORY_SSH_URL
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_github_branch_tip", return_value="a" * 40) as tip,
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)) as git_run,
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    tip.assert_called_once_with(REPOSITORY_SLUG, "main")
    assert [call.args[0] for call in git_run.call_args_list] == [
        ["git", "merge-base", "--is-ancestor", "a" * 40, "HEAD"]
    ]  # Only local ancestry inspection, never fetch or ls-remote over SSH.
    assert behind == banner.UPDATE_AVAILABLE_NO_COUNT


@pytest.mark.parametrize("origin", [
    REPOSITORY_SSH_URL,
    "git@github.com:NousResearch/hermes-agent.git",
])
def test_check_via_local_git_insteadof_rewrite_routes_to_ssh_fastpath(tmp_path, monkeypatch, origin):
    """#104591: the origin-URL probe must run under the fetch's config-isolated env.

    A global ``url.<https>.insteadOf`` rewrite makes a plain ``git remote get-url origin``
    report HTTPS for an SSH origin, so the SSH-avoiding fast path is skipped — while the
    fetch itself drops global config (``GIT_CONFIG_GLOBAL=/dev/null``), dials the raw SSH
    origin, and its host-key prompt opens /dev/tty and steals the CLI's keystrokes. With the
    probe under the same isolated env both sides observe the raw SSH URL and the GitHub
    API path runs instead — no fetch, no ls-remote, no ssh child.
    """
    import os
    import subprocess

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    # Config-isolated setup so the developer's own global git config can't leak in.
    setup_env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    setup_cmds = [
        ["git", "init", "-q"],
        # Pinned identity: with global/system config nulled, CI runners whose bare
        # hostname makes git's auto-detected ident "user@host.(none)" reject the commit.
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--allow-empty", "-q", "-m", "init"],
        ["git", "remote", "add", "origin", origin],
        ["git", "rev-parse", "HEAD"],
    ]
    head_sha = None
    for argv in setup_cmds:
        done = subprocess.run(
            argv, cwd=repo_dir, env=setup_env, check=True, capture_output=True, text=True)
        if argv[1] == "rev-parse":
            head_sha = done.stdout.strip()
    assert head_sha

    # Global config (visible only without GIT_CONFIG_GLOBAL isolation) rewrites SSH to HTTPS.
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        '[url "https://github.com/"]\n\tinsteadOf = git@github.com:\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Git for Windows resolves global config here too

    calls = []
    real_run = banner.subprocess.run

    def spy_run(args, **kwargs):
        calls.append((list(args), kwargs))
        if args[1] in {"ls-remote", "fetch"}:
            raise AssertionError(f"a GitHub origin must be probed via the API, not git {args[1]}")
        return real_run(args, **kwargs)

    monkeypatch.setattr(banner.subprocess, "run", spy_run)
    tip = MagicMock(return_value=head_sha)
    monkeypatch.setattr(banner, "_github_branch_tip", tip)

    behind = banner._check_via_local_git(repo_dir)

    # Same upstream tip as HEAD: the SSH fast path concludes "not behind".
    assert behind == 0
    tip.assert_called_once_with(origin.split(":", 1)[1].removesuffix(".git").lower(), "main")
    assert not any(args[1] in {"fetch", "ls-remote"} for args, _ in calls), (
        "insteadOf rewrite must not cause a GitHub origin to use git network transport")
    probe = next(
        (kwargs for args, kwargs in calls if args[1:3] == ["remote", "get-url"]), None)
    assert probe is not None
    assert probe["env"]["GIT_CONFIG_GLOBAL"] == os.devnull, (
        "the origin-URL probe must observe the URL the isolated fetch will dial")
