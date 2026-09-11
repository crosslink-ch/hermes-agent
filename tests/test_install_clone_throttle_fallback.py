"""Fresh-install clone throttle recovery and Crosslink HTTPS-first policy."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[1] / "scripts/install.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run_fresh_clone(tmp_path, *, direct_failures, partial_ok=True, reset_ok=True, ssh_ok=False):
    text = INSTALL_SH.read_text()
    branch = re.search(r'log_info "Trying HTTPS clone\.\.\..*?(?=\n    fi\n)', text, re.S)
    assert branch, "fresh clone branch missing"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
trace = Path(os.environ['TRACE'])
rows = [json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
with trace.open('a') as out:
    out.write(json.dumps(args) + '\\n')
root = Path(os.environ['INSTALL_DIR'])
if args[0] == 'clone':
    # Every retry must remove the previous incomplete tree first.
    if root.exists():
        raise SystemExit(97)
    root.mkdir()
    (root / 'partial').touch()
    if '--filter=blob:none' in args:
        success = os.environ['PARTIAL_OK'] == '1'
    elif any('git@' in a for a in args):
        success = os.environ['SSH_OK'] == '1'
    else:
        attempts = sum(r[0] == 'clone' and '--filter=blob:none' not in r and
                       not any('git@' in a for a in r) for r in rows)
        success = attempts >= int(os.environ['DIRECT_FAILURES'])
elif args[:3] == ['reset', '--hard', 'HEAD']:
    success = os.environ['RESET_OK'] == '1'
else:
    raise SystemExit('unexpected git: ' + repr(args))
raise SystemExit(0 if success else 1)
''')
    (bin_dir / "git").chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "INSTALL_DIR": str(tmp_path / "install"), "TRACE": str(tmp_path / "trace"),
           "SLEEPS": str(tmp_path / "sleeps"), "DIRECT_FAILURES": str(direct_failures),
           "PARTIAL_OK": str(int(partial_ok)), "RESET_OK": str(int(reset_ok)),
           "SSH_OK": str(int(ssh_ok)), "BRANCH": "main",
           "REPO_URL_HTTPS": "https://github.com/crosslink-ch/hermes-agent.git",
           "REPO_URL_SSH": "git@github.com:crosslink-ch/hermes-agent.git"}
    harness = '\n'.join([
        'set -euo pipefail', 'log_info() { :; }',
        'log_success() { printf "SUCCESS %s\\n" "$*"; }',
        'log_error() { printf "ERROR %s\\n" "$*"; }',
        'sleep() { printf "%s\\n" "$1" >> "$SLEEPS"; }',
        'clone_fresh() {', branch.group(), '}', 'clone_fresh',
    ])
    result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True)
    rows = [json.loads(x) for x in (tmp_path / "trace").read_text().splitlines()]
    sleeps = (tmp_path / "sleeps").read_text().splitlines() if (tmp_path / "sleeps").exists() else []
    return result, rows, sleeps


@pytest.mark.parametrize(
    "direct_failures,reset_ok,ssh_ok,expected_clones,transport",
    [(0, True, False, 1, "HTTPS"), (2, True, False, 3, "HTTPS"),
     (99, True, False, 5, "HTTPS"), (99, False, True, 6, "SSH")],
)
def test_https_retries_then_materializes_before_ssh(
    tmp_path, direct_failures, reset_ok, ssh_ok, expected_clones, transport
):
    result, rows, sleeps = _run_fresh_clone(
        tmp_path, direct_failures=direct_failures, reset_ok=reset_ok, ssh_ok=ssh_ok
    )
    assert result.returncode == 0, result.stdout + result.stderr
    clones = [r for r in rows if r[0] == "clone"]
    assert len(clones) == expected_clones
    assert "https://github.com/crosslink-ch/hermes-agent.git" in clones[0]
    assert f"SUCCESS Cloned via {transport}" in result.stdout
    if direct_failures == 2:
        assert sleeps == ["5", "10"]
    if direct_failures > 4:
        assert "--filter=blob:none" in clones[4] and "--no-checkout" in clones[4]
        resets = [r for r in rows if r[0] == "reset"]
        assert len(resets) == (1 if reset_ok else 2)
        assert sleeps[:3] == ["5", "10", "15"]
    else:
        assert not any("--filter=blob:none" in r for r in rows)


@pytest.mark.parametrize("partial_ok", [True, False])
def test_exhausted_clone_and_materialization_fail_closed(tmp_path, partial_ok):
    result, rows, _ = _run_fresh_clone(
        tmp_path, direct_failures=99, partial_ok=partial_ok, reset_ok=False, ssh_ok=False
    )
    assert result.returncode == 1
    assert "SUCCESS" not in result.stdout
    assert "ERROR Failed to clone repository" in result.stdout
    assert not (tmp_path / "install").exists()
    assert len([r for r in rows if r[0] == "clone"]) == 6
