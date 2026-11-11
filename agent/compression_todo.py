"""Todo continuity at an admitted compression boundary, never human intent.

Past messages may be rewritten only here: preserve structured content and invalidate
wire-content sidecars whenever a stale snapshot is removed or refreshed.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TODO_INTERNAL_NOTE_PREFIX = (
    "[Internal continuity note preserved across context compression — "
    "not a user message and not the latest request.]"
)


def _strip_todo_internal_note(content: Any) -> Any:
    """Remove an internal todo note while preserving assistant content."""
    if isinstance(content, str):
        idx = content.find(_TODO_INTERNAL_NOTE_PREFIX)
        if idx == -1:
            return content
        return content[:idx].rstrip()
    if isinstance(content, list):
        cleaned = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text") or "")
                idx = text.find(_TODO_INTERNAL_NOTE_PREFIX)
                if idx != -1:
                    retained = text[:idx].rstrip()
                    if retained:
                        updated = dict(part)
                        updated["text"] = retained
                        cleaned.append(updated)
                    continue
            cleaned.append(part)
        return cleaned
    return content


def _keep_stripped_row(message: dict, stripped: Any, flag: str) -> bool:
    """Write the stripped content back; False when nothing is left of the row."""
    from agent.conversation_compression import _message_text, _replace_message_content
    has_content = bool(
        _message_text({"content": stripped}).strip()
    ) or (isinstance(stripped, list) and bool(stripped))
    if not has_content and not message.get("tool_calls"):
        return False
    _replace_message_content(message, stripped)
    message.pop(flag, None)
    return True


def _refresh_stale_row(message: dict) -> bool:
    """Strip a stale todo note/snapshot from one row; False drops the row."""
    role = message.get("role")
    original = message.get("content")
    if role == "assistant":
        stripped = _strip_todo_internal_note(original)
        if stripped != original:
            return _keep_stripped_row(message, stripped, "_todo_snapshot_internal")
        if message.get("_todo_snapshot_internal"):
            message.pop("_todo_snapshot_internal", None)
    elif role == "user":
        from agent.conversation_compression import _strip_stale_todo_snapshot
        stripped = _strip_stale_todo_snapshot(original)
        if stripped != original:
            return _keep_stripped_row(message, stripped, "_todo_snapshot_synthetic")
    return True


def _note_insert_index(messages: list) -> int:
    """Before the latest real user turn, else before the synthetic continuation row."""
    from agent.conversation_compression import _is_real_user_message
    for idx in range(len(messages) - 1, -1, -1):
        if _is_real_user_message(messages[idx]):
            return idx
    # Zero-user sessions retain upstream's synthetic continuation row for
    # strict role templates; the internal note belongs immediately before it.
    for idx in range(len(messages) - 1, -1, -1):
        if (
            isinstance(messages[idx], dict)
            and messages[idx].get("role") == "user"
        ):
            return idx
    return len(messages)


def _fold_note_into_assistant(previous: dict, note_content: str) -> bool:
    """Append the note to the preceding assistant row; False for content it cannot extend."""
    from agent.conversation_compression import _replace_message_content
    previous_content = previous.get("content")
    if isinstance(previous_content, str):
        base = previous_content.rstrip()
        _replace_message_content(previous, f"{base}\n\n{note_content}" if base else note_content)
    elif isinstance(previous_content, list):
        _replace_message_content(previous, [*previous_content, {"type": "text", "text": note_content}])
    elif previous_content is None:
        _replace_message_content(previous, note_content)
    else:
        return False
    previous["_todo_snapshot_internal"] = True
    return True


def _repair_after_removal(messages: list, removed_rows: bool) -> None:
    # Only removal can create a new adjacency here; don't rewrite clean candidates.
    if removed_rows:
        from agent.agent_runtime_helpers import repair_message_sequence
        repair_message_sequence(None, messages)


def _inject_todo_snapshot_internal_note(messages: list, todo_snapshot: str) -> None:
    """Refresh todo continuity state without turning it into human intent."""
    snapshot = str(todo_snapshot or "").strip()
    cleaned = [
        message for message in messages
        if not isinstance(message, dict) or _refresh_stale_row(message)
    ]
    removed_rows = len(cleaned) != len(messages)
    messages[:] = cleaned
    if not snapshot:
        _repair_after_removal(messages, removed_rows)
        return

    note = {
        "role": "assistant",
        "content": f"{_TODO_INTERNAL_NOTE_PREFIX}\n{snapshot}",
        "_todo_snapshot_internal": True,
    }
    insert_at = _note_insert_index(messages)
    if insert_at > 0:
        previous = messages[insert_at - 1]
        if isinstance(previous, dict) and previous.get("role") == "assistant":
            if not _fold_note_into_assistant(previous, note["content"]):
                messages.insert(insert_at, note)
                return
            _repair_after_removal(messages, removed_rows)
            return

    messages.insert(insert_at, note)
    _repair_after_removal(messages, removed_rows)


def fold_todo_snapshot(agent: Any, compressed: list) -> None:
    """Refresh once; an unhydrated store preserves the last pending snapshot.

    A nonempty store is authoritative even when every item is complete. Only a
    truly empty/unknown store may recover continuity from the previous boundary.
    """
    from agent.conversation_compression import _pruned_skill_reload_notice
    from tools.todo_tool import TODO_INJECTION_HEADER

    snapshot = agent._todo_store.format_for_injection()
    has_items = getattr(agent._todo_store, "has_items", None)
    authoritative = bool(snapshot)
    if callable(has_items):
        try:
            authoritative = authoritative or bool(has_items())
        except Exception:
            # Third-party store: preserve continuity when authority is unknown.
            logger.debug("todo store has_items() failed; keeping the last snapshot", exc_info=True)
    if not snapshot and not authoritative:
        for message in reversed(compressed):
            if not isinstance(message, dict) or message.get("role") not in {"assistant", "user"}:
                continue
            content = message.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for part in reversed(parts):
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                body = str(part.get("text") or "")
                if TODO_INJECTION_HEADER in body:
                    snapshot = body[body.index(TODO_INJECTION_HEADER):].strip()
                    break
            if snapshot:
                break
    if snapshot:
        notice = _pruned_skill_reload_notice(compressed)
        if notice and notice not in snapshot:
            snapshot = f"{snapshot}\n\n{notice}"
    _inject_todo_snapshot_internal_note(compressed, snapshot)
