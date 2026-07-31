from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

LEGACY_HTTPS = "https://github.com/NousResearch/hermes-agent.git"
LEGACY_SSH = "git@github.com:NousResearch/hermes-agent.git"
CROSSLINK_HTTPS = "https://github.com/crosslink-ch/hermes-agent.git"
CROSSLINK_SSH = "git@github.com:crosslink-ch/hermes-agent.git"
HOSTED_INSTALLER_BASE = "https://share.kihub.ch/hermes/install."

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("git") or "git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _managed_checkout(tmp_path: Path, origin: str) -> Path:
    repo = tmp_path / "managed-checkout"
    _git("init", "-b", "main", str(repo))
    _git("config", "user.name", "Installer Test", cwd=repo)
    _git("config", "user.email", "installer@example.invalid", cwd=repo)
    (repo / "README.md").write_text("managed\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-m", "seed", cwd=repo)
    _git("remote", "add", "origin", origin, cwd=repo)
    return repo


def _fake_git_bin(tmp_path: Path) -> Path:
    """Use real git except for network fetch/pull performed after migration."""
    real_git = shutil.which("git")
    assert real_git is not None
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"real_git = {real_git!r}\n"
        "operation = next((arg for arg in sys.argv[1:] if arg in {'fetch', 'pull'}), None)\n"
        "if operation in {'fetch', 'pull'}:\n"
        "    raise SystemExit(0)\n"
        "os.execv(real_git, [real_git, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run_repository_stage(
    tmp_path: Path,
    repo: Path,
    *,
    migrate: bool,
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    fake_bin = _fake_git_bin(tmp_path)
    args = [
        shutil.which("bash") or "bash",
        str(INSTALL_SH),
        "--stage",
        "repository",
        "--non-interactive",
        "--json",
        "--dir",
        str(repo),
        "--hermes-home",
        str(home / ".hermes"),
    ]
    if migrate:
        args.append("--migrate-legacy-origin")
    env = dict(os.environ)
    env.update({
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
    })
    return subprocess.run(args, env=env, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("legacy_origin", "expected_origin"),
    [(LEGACY_HTTPS, CROSSLINK_HTTPS), (LEGACY_SSH, CROSSLINK_SSH)],
)
def test_repository_stage_migrates_only_known_legacy_origin(
    tmp_path: Path,
    legacy_origin: str,
    expected_origin: str,
) -> None:
    repo = _managed_checkout(tmp_path, legacy_origin)

    result = _run_repository_stage(tmp_path, repo, migrate=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        _git("remote", "get-url", "origin", cwd=repo).stdout.strip() == expected_origin
    )
    assert "Managed checkout now follows crosslink-ch/hermes-agent" in result.stdout


@pytest.mark.parametrize("origin", [CROSSLINK_HTTPS, CROSSLINK_SSH])
def test_crosslink_repository_stage_is_idempotent(tmp_path: Path, origin: str) -> None:
    repo = _managed_checkout(tmp_path, origin)

    result = _run_repository_stage(tmp_path, repo, migrate=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git("remote", "get-url", "origin", cwd=repo).stdout.strip() == origin
    assert "Migrating managed checkout origin" not in result.stdout


def test_noninteractive_legacy_install_requires_explicit_migration(
    tmp_path: Path,
) -> None:
    repo = _managed_checkout(tmp_path, LEGACY_HTTPS)

    result = _run_repository_stage(tmp_path, repo, migrate=False)

    assert result.returncode != 0
    assert _git("remote", "get-url", "origin", cwd=repo).stdout.strip() == LEGACY_HTTPS
    assert "no files were modified" in result.stdout


def test_unknown_origin_is_never_retargeted(tmp_path: Path) -> None:
    custom_origin = "git@github.com:example/custom-hermes.git"
    repo = _managed_checkout(tmp_path, custom_origin)

    result = _run_repository_stage(tmp_path, repo, migrate=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git("remote", "get-url", "origin", cwd=repo).stdout.strip() == custom_origin
    assert "Leaving the custom origin unchanged" in result.stdout


def test_distribution_critical_surfaces_do_not_point_installers_upstream() -> None:
    roots = [
        REPO_ROOT / "scripts",
        REPO_ROOT / "hermes_cli",
        REPO_ROOT / "apps" / "desktop" / "electron",
        REPO_ROOT / "apps" / "desktop" / "src" / "i18n",
        REPO_ROOT / "apps" / "bootstrap-installer",
        REPO_ROOT / "website",
        REPO_ROOT / "skills" / "autonomous-ai-agents" / "hermes-agent",
    ]
    files = [
        *REPO_ROOT.glob("README*.md"),
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "hermes-already-has-routines.md",
    ]
    text_suffixes = {".cmd", ".js", ".md", ".mdx", ".ps1", ".py", ".rs", ".sh", ".ts"}
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in text_suffixes
        )

    forbidden = (
        "https://hermes-agent.nousresearch.com/install.",
        "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.",
    )
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if any(value in text for value in forbidden):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"upstream installer URLs remain in: {sorted(set(offenders))}"


def test_distribution_critical_sources_use_crosslink_and_hosted_urls() -> None:
    install_ps1 = (REPO_ROOT / "scripts" / "install.ps1").read_text(encoding="ascii")
    install_sh = INSTALL_SH.read_text(encoding="utf-8")
    install_cmd = (REPO_ROOT / "scripts" / "install.cmd").read_text(encoding="utf-8")
    dev_sandbox = (REPO_ROOT / "scripts" / "dev-sandbox.sh").read_text(
        encoding="utf-8"
    )
    update_cmd = (REPO_ROOT / "hermes_cli" / "update_cmd.py").read_text(
        encoding="utf-8"
    )
    desktop_bootstrap = (
        REPO_ROOT / "apps" / "desktop" / "electron" / "bootstrap-runner.ts"
    ).read_text(encoding="utf-8")
    desktop_about = (
        REPO_ROOT
        / "apps"
        / "desktop"
        / "src"
        / "app"
        / "settings"
        / "about-settings.tsx"
    ).read_text(encoding="utf-8")
    native_bootstrap = (
        REPO_ROOT
        / "apps"
        / "bootstrap-installer"
        / "src-tauri"
        / "src"
        / "install_script.rs"
    ).read_text(encoding="utf-8")

    assert 'RepoSlug = "crosslink-ch/hermes-agent"' in install_ps1
    assert 'REPO_SLUG="crosslink-ch/hermes-agent"' in install_sh
    assert HOSTED_INSTALLER_BASE in install_cmd
    assert "https://github.com/crosslink-ch/hermes-agent.git" in dev_sandbox
    assert "https://share.kihub.ch/hermes/install.sh" in dev_sandbox
    assert "ARCHIVE_BASE_URL" in update_cmd
    assert "raw.githubusercontent.com/crosslink-ch/hermes-agent" in desktop_bootstrap
    assert "https://github.com/crosslink-ch/hermes-agent/releases" in desktop_about
    assert "raw.githubusercontent.com/crosslink-ch/hermes-agent" in native_bootstrap
    assert "MigrateLegacyOrigin" in desktop_bootstrap
