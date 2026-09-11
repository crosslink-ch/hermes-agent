"""Behavior of the replacement installer E2E's Crosslink migration bridge."""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ["installer-script-e2e.sh", "macos-desktop-e2e.sh"]


def _function(driver, name):
    text = (ROOT / "tests/install" / driver).read_text()
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.M | re.S)
    assert match, name
    return match.group()


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("migrates", [True, False])
def test_ref_installer_flags_are_probed_without_sigpipe(tmp_path, driver, migrates):
    # Oversized source reproduces the pipefail/SIGPIPE bug in the retired driver.
    script = "#!/usr/bin/env bash\n# --skip-browser\n"
    if migrates:
        script += "# --migrate-legacy-origin\n"
    script += 'printf "%s\\n" "$@" > "$FLAGS_FILE"\n' + "# filler\n" * 32000
    source = tmp_path / "source.sh"
    source.write_text(script)
    env = {**os.environ, "SOURCE_FILE": str(source), "FLAGS_FILE": str(tmp_path / "flags"),
           "WORK_ROOT": str(tmp_path), "LOG_DIR": str(tmp_path), "REPO_ROOT": str(ROOT)}
    harness = "\n".join([
        "set -euo pipefail", 'git() { cat "$SOURCE_FILE"; }',
        "ts_prefix() { cat; }", "log_group() { :; }", "fail() { exit 77; }",
        _function(driver, "installer_supports"), _function(driver, "run_installer"),
        "run_installer HEAD probe",
    ])
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    flags = (tmp_path / "flags").read_text().splitlines()
    assert "--skip-browser" in flags
    assert ("--migrate-legacy-origin" in flags) is migrates


def test_redirect_shim_reports_real_origin_before_and_after_migration(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    legacy = "https://github.com/NousResearch/hermes-agent.git"
    crosslink = "https://github.com/crosslink-ch/hermes-agent.git"
    for args in [("init", "-q"), ("remote", "add", "origin", legacy)]:
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    env = {**os.environ, "WORK_ROOT": str(tmp_path), "REPO_ROOT": str(repo),
           "SERVE_REPO": str(tmp_path / "serve.git"), "REPO_URL_HTTPS": crosslink,
           "REPO_URL_SSH": "git@github.com:crosslink-ch/hermes-agent.git"}
    harness = "\n".join([
        "set -euo pipefail", "ok() { :; }", 'fail() { printf "%s\\n" "$*" >&2; exit 77; }',
        _function("installer-script-e2e.sh", "arm_redirect"), "arm_redirect",
        'git -C "$REPO_ROOT" remote get-url origin',
        'git -C "$REPO_ROOT" remote set-url origin "$REPO_URL_HTTPS"',
        'git -C "$REPO_ROOT" remote get-url origin',
    ])
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [legacy, crosslink]
