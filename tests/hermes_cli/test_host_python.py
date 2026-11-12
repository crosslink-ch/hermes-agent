"""``hermes_cli.host_python`` runs host scripts like the interpreter it replaces."""
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])


def _env(tmp_path, *extra_path):
    """The child imports the checkout under test (and any extra roots), like the launcher arranges."""
    import os
    return {**os.environ, "PYTHONPATH": os.pathsep.join([_REPO, *extra_path])}


def _run(tmp_path, *argv):
    return subprocess.run([sys.executable, "-m", "hermes_cli.host_python", *argv],
                          capture_output=True, text=True, cwd=tmp_path, timeout=60, check=False,
                          env=_env(tmp_path))


def test_script_path_gets_its_own_argv_and_main_name(tmp_path):
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent("""
        import sys
        if __name__ == "__main__":
            print(sys.argv)
            sys.exit(3)
    """), encoding="utf-8")

    proc = _run(tmp_path, str(script), "--apply", "x")

    assert proc.returncode == 3
    assert proc.stdout.strip() == repr([str(script), "--apply", "x"])


def test_module_and_code_forms(tmp_path):
    (tmp_path / "probe_mod.py").write_text("import sys\nprint('mod', sys.argv[1:])\n", encoding="utf-8")

    module = subprocess.run([sys.executable, "-m", "hermes_cli.host_python", "-m", "probe_mod", "a"],
                            capture_output=True, text=True, cwd=tmp_path, timeout=60, check=False,
                            env=_env(tmp_path, str(tmp_path)))
    code = _run(tmp_path, "-c", "import sys; print('code', sys.argv)", "b")

    assert module.stdout.strip() == "mod ['a']"
    assert code.stdout.strip() == "code ['-c', 'b']"


def test_missing_operand_is_a_usage_error(tmp_path):
    assert _run(tmp_path).returncode == 2
    assert _run(tmp_path, "-m").returncode == 2
