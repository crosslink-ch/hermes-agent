"""Regression tests for install.sh browser setup.

Browser automation is optional. The installer should use the browser installer
owned by its locked ``agent-browser`` dependency and keep downloads bounded.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_honors_explicit_browser_override_only() -> None:
    """find_system_browser consults only an explicit AGENT_BROWSER_EXECUTABLE_PATH."""
    text = INSTALL_SH.read_text()

    assert 'override="${AGENT_BROWSER_EXECUTABLE_PATH:-}"' in text
    assert "Skipping bundled Chromium download" in text


def test_agent_browser_installs_are_timeout_guarded() -> None:
    text = INSTALL_SH.read_text()

    assert "run_browser_install_with_timeout()" in text
    assert "run_agent_browser_install 600 npx agent-browser install" in text
    assert "run_agent_browser_install 600 npx agent-browser install --with-deps" in text
    assert 'run_browser_install_with_timeout "$timeout_seconds" "$@"' in text
    # The root package installs agent-browser, not the raw Playwright CLI.
    assert "npx playwright install" not in text


def test_install_script_supports_skip_browser_flag() -> None:
    """--skip-browser (and --no-playwright alias) skips the browser install."""
    text = INSTALL_SH.read_text()

    assert "--skip-browser|--no-playwright)" in text
    assert "SKIP_BROWSER=true" in text
    assert 'if [ "$SKIP_BROWSER" = true ]; then' in text
    assert "--skip-browser Skip Playwright/Chromium install" in text


def test_browser_install_timeout_stays_interruptible() -> None:
    """The browser download must stay Ctrl+C-able and force-kill if wedged."""
    text = INSTALL_SH.read_text()

    assert '"$timeout_bin" --foreground -k 10 1 true' in text
    assert '"$timeout_bin" --foreground -k 10 "$timeout_seconds" "$@"' in text
    assert '"$timeout_bin" "$timeout_seconds" "$@"' in text


def _run_agent_browser_helper(tmp_path: Path, *, command_rc: int) -> dict[str, object]:
    """Extract the timeout helpers and invoke a stubbed agent-browser install."""
    fn_names = [
        "run_browser_install_with_timeout",
        "run_with_timeout",
        "run_agent_browser_install",
    ]
    source = INSTALL_SH.read_text()
    extracted: list[str] = []
    for name in fn_names:
        match = re.search(
            rf"^{re.escape(name)}\(\) \{{.*?^\}}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        assert match, f"could not extract {name}() from install.sh"
        extracted.append(match.group(0))

    runlog = tmp_path / "runs.log"
    body = "\n\n".join(extracted)
    harness = f"""
set -u
RUNLOG={str(runlog)!r}

timeout() {{
    while [ $# -gt 0 ]; do
        case "$1" in -*|[0-9]*) shift ;; *) break ;; esac
    done
    "$@"
}}

npx() {{
    printf '%s\n' "$*" >>"$RUNLOG"
    return {command_rc}
}}

{body}

run_agent_browser_install 600 npx agent-browser install --with-deps
printf 'FINAL_RC=%s\n' "$?"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
    )
    runs = runlog.read_text().splitlines()
    final_line = next(
        line for line in result.stdout.splitlines() if line.startswith("FINAL_RC=")
    )
    return {
        "runs": runs,
        "final_rc": int(final_line.split("=", 1)[1]),
        "stderr": result.stderr,
    }


def test_agent_browser_helper_invokes_locked_cli_once(tmp_path: Path) -> None:
    result = _run_agent_browser_helper(tmp_path, command_rc=0)

    assert result["runs"] == ["agent-browser install --with-deps"]
    assert result["final_rc"] == 0


def test_agent_browser_helper_propagates_failure_without_false_retry(tmp_path: Path) -> None:
    result = _run_agent_browser_helper(tmp_path, command_rc=9)

    assert result["runs"] == ["agent-browser install --with-deps"]
    assert result["final_rc"] == 9
