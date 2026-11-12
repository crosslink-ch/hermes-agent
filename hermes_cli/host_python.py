"""``python``-shaped entry for host scripts under the source launcher.

``<checkout>/.hermes/bin/hermes --run-module hermes_cli.host_python SCRIPT [ARGS...]`` runs a script the
way ``<checkout>/.venv/bin/python SCRIPT [ARGS...]`` did before PM owned the interpreter: the launcher has
already selected the store Python and the committed dependency generation, and this module only dispatches
to a script path, ``-m MODULE`` or ``-c CODE`` with ``sys.argv`` shaped as CPython would shape it.
"""
from __future__ import annotations

import runpy
import sys

_USAGE = "usage: host_python SCRIPT [ARGS...] | -m MODULE [ARGS...] | -c CODE [ARGS...]\n"


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-m", "-c"} and len(argv) < 2:
        sys.stderr.write(_USAGE)
        return 2
    if argv[0] == "-m":
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    elif argv[0] == "-c":
        sys.argv = ["-c", *argv[2:]]
        exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
    else:
        sys.argv = list(argv)
        runpy.run_path(argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
