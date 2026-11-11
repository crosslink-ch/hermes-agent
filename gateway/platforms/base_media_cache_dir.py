"""Owner-only creation of the gateway media-cache directories."""
import logging
from pathlib import Path

from hermes_dir_acl import make_user_dir

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.platforms.base")


def _secure_media_cache_dir(cache_dir: Path) -> None:
    """Create/reconcile a gateway media-cache dir owner-only (0700), except managed.

    Inbound media (photos, voice notes, documents a user sent through a
    messaging platform) is user content, not regenerable cache: the
    umask-derived 0755 these dirs inherited made them readable by every other
    local account whenever ``HERMES_HOME`` is traversable — the documented
    ``HERMES_HOME_MODE=0701`` web-server hatch. The mode is passed to ``mkdir``
    so it is set at creation, then reconciled by the house policy helper
    ``hermes_cli.config._secure_dir`` (managed/NixOS installs keep their
    group-share design: these lazily-created dirs are not covered by the
    module's ``systemd.tmpfiles`` rules, so the mode is left to the
    configured umask/setgid). Best-effort: never fails a media write.
    """
    try:
        managed = False
        try:
            from hermes_cli.config import is_managed

            managed = bool(is_managed())
        except Exception:  # pragma: no cover - defensive
            pass
        if managed:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return
        make_user_dir(cache_dir, parents=True, exist_ok=True, mode=0o700)
        try:
            from hermes_cli.config import _secure_dir

            _secure_dir(cache_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("media cache dir chmod skipped: %s", exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("media cache dir creation failed for %s: %s", cache_dir, exc)
        cache_dir.mkdir(parents=True, exist_ok=True)
