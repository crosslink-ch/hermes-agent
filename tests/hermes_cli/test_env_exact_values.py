"""Exact opt-in dotenv values without changing unmarked legacy scoped files."""
import json
import os
import subprocess
import sys

import pytest

from agent.secret_scope import get_secret, load_env_file, is_multiplex_active, set_multiplex_active
from gateway.run import _profile_runtime_scope
from hermes_cli.config import load_env, save_env_value_secure


def test_unmarked_assignments_keep_legacy_scope_semantics_across_profiles(tmp_path):
    homes = [tmp_path / "A", tmp_path / "B"]
    expected = {
        "PASSWORD": r"literal\new\return",
        "SINGLE": r"literal\n",
        "COMMENT": r"literal\n",
        "EMBEDDED": r"literal\n # hermes:exact:v1",
        "ESCAPED": r"literal\n\r" + '"',
    }
    for home in homes:
        home.mkdir()
        (home / ".env").write_text(
            'PASSWORD="literal\\new\\return"\n'
            "SINGLE='literal\\n'\n"
            'COMMENT="literal\\n" # ordinary comment\n'
            'EMBEDDED="literal\\n # hermes:exact:v1"\n'
            'ESCAPED="literal\\\\n\\\\r\\\""\n', encoding="utf-8")
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for home in (homes[0], homes[1], homes[0]):
            with _profile_runtime_scope(home):
                assert load_env() == load_env_file(home / ".env") == expected
                assert {name: get_secret(name) for name in expected} == expected
    finally:
        set_multiplex_active(previous)


@pytest.mark.parametrize("value", [
    "  edge é\nnext\rline ' \" \\n \\r ",
    "é\u2028middle\u2029end\u0085",
    "${NOT_EXPANDED} # text \\literal\\t\\a\\n\\r",
    'literal\\n # hermes:exact:v1 " trailing\\',
])
def test_exact_writer_survives_scoped_and_fresh_startup_loaders(tmp_path, monkeypatch, value):
    from dotenv import dotenv_values
    from tools.environments.local import build_subprocess_env
    key = "SYNTHETIC_EXACT_KEY"  # exercises the normal startup credential sanitizer too
    monkeypatch.delenv(key, raising=False)
    previous = is_multiplex_active()
    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
    set_multiplex_active(True)
    try:
        for index, home in enumerate(homes):
            with _profile_runtime_scope(home):
                save_env_value_secure(key, value + str(index), preserve_exact=True)
        for index in (0, 1, 0):
            home = homes[index]
            with _profile_runtime_scope(home):
                assert load_env()[key] == get_secret(key) == value + str(index)
            # Valid standard dotenv, not a proprietary encoding hidden from startup.
            assert dotenv_values(home / ".env", interpolate=False)[key] == value + str(index)
            env = build_subprocess_env()
            env["HERMES_HOME"] = str(home)
            env.pop(key, None)
            proc = subprocess.run([sys.executable, "-c", (
                "import json,os,sys; from hermes_cli.env_loader import load_hermes_dotenv; "
                "load_hermes_dotenv(load_external_secrets=False); expected=json.load(sys.stdin); "
                "assert all(os.environ.get(k)==v for k,v in expected.items()), 'startup changed exact value'; "
                "print('exact startup verified')"
            )], env=env, input=json.dumps({key: value + str(index)}),
                text=True, capture_output=True, timeout=30)
            assert proc.returncode == 0, proc.stderr
            assert "exact startup verified" in proc.stdout
        assert key not in os.environ
    finally:
        set_multiplex_active(previous)
