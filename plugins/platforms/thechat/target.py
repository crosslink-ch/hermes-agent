"""TheChat native conversation identifiers."""
import re

CONVERSATION_RE = re.compile(
    r"^\s*([0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\s*$",
    re.IGNORECASE,
)
