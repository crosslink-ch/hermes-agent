"""Distribution-owned repository and installer endpoints.

Crosslink Hermes installations follow ``crosslink-ch/hermes-agent`` for normal
updates. ``NousResearch/hermes-agent`` remains the source project used by
maintainers when they intentionally synchronize the fork, but it is not an
end-user update remote.
"""

from __future__ import annotations

from urllib.parse import urlparse

REPOSITORY_SLUG = "crosslink-ch/hermes-agent"
REPOSITORY_HTTPS_URL = f"https://github.com/{REPOSITORY_SLUG}.git"
REPOSITORY_SSH_URL = f"git@github.com:{REPOSITORY_SLUG}.git"
REPOSITORY_CANONICAL = f"github.com/{REPOSITORY_SLUG}"
REPOSITORY_URLS = frozenset({
    REPOSITORY_HTTPS_URL,
    REPOSITORY_HTTPS_URL.removesuffix(".git"),
    REPOSITORY_SSH_URL,
    REPOSITORY_SSH_URL.removesuffix(".git"),
})

INSTALLER_BASE_URL = "https://share.kihub.ch/hermes"
ARCHIVE_BASE_URL = f"https://github.com/{REPOSITORY_SLUG}/archive"
RELEASE_URL_BASE = f"https://github.com/{REPOSITORY_SLUG}/releases/tag"


def canonical_github_remote(url: str | None) -> str:
    """Return ``host/owner/repo`` for common GitHub remote URL forms.

    Comparison is case-insensitive and ignores a trailing slash or ``.git``.
    The helper accepts HTTPS plus both common SSH forms so public installs and
    developer checkouts are treated as the same distribution repository.
    """
    if not url:
        return ""

    value = str(url).strip()
    lowered = value.lower()
    if lowered.startswith("git@github.com:"):
        value = f"github.com/{value[len('git@github.com:') :]}"
    elif lowered.startswith("ssh://git@github.com/"):
        value = f"github.com/{value[len('ssh://git@github.com/') :]}"
    else:
        parsed = urlparse(value)
        if parsed.hostname and parsed.path:
            value = f"{parsed.hostname}{parsed.path}"

    value = value.strip().rstrip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    return value.lower()
