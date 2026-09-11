"""Todo continuity at an admitted compression boundary, never human intent.

Past messages may be rewritten only here: preserve structured content and invalidate
wire-content sidecars whenever a stale snapshot is removed or refreshed.
"""
from typing import Any

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


def _inject_todo_snapshot_internal_note(messages: list, todo_snapshot: str) -> None:
    """Refresh todo continuity state without turning it into human intent."""
    from agent.conversation_compression import (
        _is_real_user_message, _message_text, _replace_message_content,
        _strip_stale_todo_snapshot,
    )
    snapshot = str(todo_snapshot or "").strip()
    cleaned: list = []

    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue

        if message.get("role") == "assistant":
            original = message.get("content")
            stripped = _strip_todo_internal_note(original)
            if stripped != original:
                has_content = bool(
                    _message_text({"content": stripped}).strip()
                ) or (isinstance(stripped, list) and bool(stripped))
                if not has_content and not message.get("tool_calls"):
                    continue
                _replace_message_content(message, stripped)
                message.pop("_todo_snapshot_internal", None)
            elif message.get("_todo_snapshot_internal"):
                message.pop("_todo_snapshot_internal", None)

        if message.get("role") == "user":
            original = message.get("content")
            stripped = _strip_stale_todo_snapshot(original)
            if stripped != original:
                has_content = bool(
                    _message_text({"content": stripped}).strip()
                ) or (isinstance(stripped, list) and bool(stripped))
                if not has_content and not message.get("tool_calls"):
                    continue
                _replace_message_content(message, stripped)
                message.pop("_todo_snapshot_synthetic", None)

        cleaned.append(message)

    removed_rows = len(cleaned) != len(messages)
    messages[:] = cleaned
    if not snapshot:
        # Only removal can create a new adjacency here; don't rewrite clean
        # candidates when there is no todo state to refresh.
        if removed_rows:
            from agent.agent_runtime_helpers import repair_message_sequence
            repair_message_sequence(None, messages)
        return

    note = {
        "role": "assistant",
        "content": f"{_TODO_INTERNAL_NOTE_PREFIX}\n{snapshot}",
        "_todo_snapshot_internal": True,
    }

    insert_at = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        if _is_real_user_message(messages[idx]):
            insert_at = idx
            break
    else:
        # Zero-user sessions retain upstream's synthetic continuation row for
        # strict role templates; the internal note belongs immediately before it.
        for idx in range(len(messages) - 1, -1, -1):
            if (
                isinstance(messages[idx], dict)
                and messages[idx].get("role") == "user"
            ):
                insert_at = idx
                break

    if insert_at > 0:
        previous = messages[insert_at - 1]
        if isinstance(previous, dict) and previous.get("role") == "assistant":
            note_content = note["content"]
            previous_content = previous.get("content")
            if isinstance(previous_content, str):
                base = previous_content.rstrip()
                _replace_message_content(previous, f"{base}\n\n{note_content}" if base else note_content)
            elif isinstance(previous_content, list):
                _replace_message_content(previous, [*previous_content, {"type": "text", "text": note_content}])
            elif previous_content is None:
                _replace_message_content(previous, note_content)
            else:
                messages.insert(insert_at, note)
                return
            previous["_todo_snapshot_internal"] = True
            if removed_rows:
                from agent.agent_runtime_helpers import repair_message_sequence
                repair_message_sequence(None, messages)
            return

    messages.insert(insert_at, note)
    if removed_rows:
        from agent.agent_runtime_helpers import repair_message_sequence
        repair_message_sequence(None, messages)


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
            pass  # Third-party store: preserve continuity when authority is unknown.
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
