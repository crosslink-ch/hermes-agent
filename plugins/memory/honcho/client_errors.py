"""Exception families the Honcho provider degrades on, named once for its ``except`` clauses.

Handlers catch these by name and log the message only. An SDK error can quote the request it rejected
(a 422 echoes its input) and an auth error chains the unredacted original, so these never get a
traceback; a catch-all boundary attaches one only to what falls outside them.
"""

from __future__ import annotations

import sys

# Resolving a profile: an unreadable directory, an invalid name, a symlink loop (Path.resolve() raises
# RuntimeError for one before Python 3.13).
PROFILE_ERRORS = (OSError, ValueError, RuntimeError)
# Reading user-edited config and the scoped secrets beside it. honcho.json may come from another app in
# any JSON shape, so a wrong container type surfaces as TypeError / AttributeError / LookupError.
# RuntimeError: no home directory to resolve the path from (Path.home()), or a secret read outside a
# profile scope (agent.secret_scope.UnscopedSecretError).
CONFIG_ERRORS = (OSError, ValueError, TypeError, AttributeError, LookupError, RuntimeError)


def http_errors() -> tuple[type[BaseException], ...]:
    """A raw HTTP exchange (OAuth endpoints, SDK uploads): transport and status errors, socket errors,
    and a body that does not parse or validate."""
    import httpx

    return (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError)


def call_errors() -> tuple[type[BaseException], ...]:
    """A Honcho round trip, from building the client to reading the reply: SDK errors, the HTTP family
    (its ValueError also covers the SDK's pydantic validation and a missing API key), a rejected
    credential, an unresolvable user peer, and an SDK that is absent or cannot be installed."""
    from pm import InstallError
    from plugins.memory.honcho.session_auth import HonchoAuthError
    from plugins.memory.honcho.session_peers import HonchoPeerUnresolvedError

    # Read off the loaded module: an SDK that was never imported cannot have raised.
    sdk_error = getattr(sys.modules.get("honcho"), "HonchoError", None)
    sdk = (sdk_error,) if isinstance(sdk_error, type) and issubclass(sdk_error, BaseException) else ()
    return (*sdk, *http_errors(), HonchoAuthError, HonchoPeerUnresolvedError, InstallError, ImportError)


def connect_errors() -> tuple[type[BaseException], ...]:
    """``call_errors`` for a path that also resolves the config or walks SDK objects by attribute."""
    return (*CONFIG_ERRORS, *call_errors())


def is_unexpected(exc: BaseException) -> bool:
    """Whether a catch-all boundary logs ``exc`` with its traceback: True for anything but a call error."""
    return not isinstance(exc, call_errors())
