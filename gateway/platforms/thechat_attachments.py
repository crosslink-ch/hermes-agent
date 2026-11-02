"""TheChat attachment helpers: media-type normalization, bounded timeouts, safe errors."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("gateway.platforms.thechat")

_ATTACHMENT_TIMEOUT_MAX_SECONDS = 24 * 60 * 60
_ATTACHMENT_MEDIA_TYPE_ALIASES = {
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "image/jpg": "image/jpeg",
}


def _normalize_attachment_media_type(value: Any) -> str:
    normalized = str(value or "").split(";", 1)[0].strip().lower()
    return _ATTACHMENT_MEDIA_TYPE_ALIASES.get(normalized, normalized)


def _attachment_timeout_seconds(extra: dict, key: str, default: float) -> float:
    """Read a bounded platform timeout without letting bad config disable downloads."""
    if key not in extra:
        return default
    value = extra[key]
    try:
        seconds = float(value) if not isinstance(value, bool) else float("nan")
    except (TypeError, ValueError, OverflowError):
        seconds = float("nan")
    if math.isfinite(seconds) and 0 < seconds <= _ATTACHMENT_TIMEOUT_MAX_SECONDS:
        return seconds
    logger.warning("TheChat: invalid %s; using default timeout", key)
    return default


def _s3_error_code(response: httpx.Response) -> Optional[str]:
    """Extract only S3's bounded symbolic error code, never signed details."""
    match = re.search(r"<Code>([A-Za-z0-9_.-]{1,64})</Code>", response.text[:8192])
    return match.group(1) if match else None


class _AttachmentError(RuntimeError):
    """Safe attachment error whose text never contains an object-store URL."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        preserve_attachment: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.preserve_attachment = preserve_attachment
