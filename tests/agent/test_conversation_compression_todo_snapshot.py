from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
from agent.compression_todo import _TODO_INTERNAL_NOTE_PREFIX, _inject_todo_snapshot_internal_note
from agent.conversation_compression import (
    _is_real_user_message,
    _strip_stale_todo_snapshot,
)
from tools.todo_tool import TODO_INJECTION_HEADER


SNAPSHOT = f"{TODO_INJECTION_HEADER}\n- [>] report. Summarize (in_progress)"


def test_todo_snapshot_inserted_before_latest_real_user_message():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "actual latest request"},
    ]

    _inject_todo_snapshot_internal_note(messages, SNAPSHOT)

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[-1] == {"role": "user", "content": "actual latest request"}
    assert messages[-2]["_todo_snapshot_internal"] is True
    assert messages[-2]["content"] == (
        f"old answer\n\n{_TODO_INTERNAL_NOTE_PREFIX}\n{SNAPSHOT}"
    )


def test_embedded_snapshot_survives_sequence_repair_refresh_and_completion():
    original = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "actual latest request"},
    ]
    messages = [dict(message) for message in original]

    _inject_todo_snapshot_internal_note(messages, SNAPSHOT)
    assert repair_message_sequence(None, messages) == 0

    refreshed = f"{TODO_INJECTION_HEADER}\n- [>] ship. Verify (in_progress)"
    _inject_todo_snapshot_internal_note(messages, refreshed)
    assert repair_message_sequence(None, messages) == 0
    combined = "\n".join(str(message.get("content") or "") for message in messages)
    assert combined.count(_TODO_INTERNAL_NOTE_PREFIX) == 1
    assert refreshed in combined
    assert SNAPSHOT not in combined
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]

    _inject_todo_snapshot_internal_note(messages, "")
    assert messages == original


def test_empty_todo_snapshot_noops_for_clean_messages():
    messages = [{"role": "user", "content": "hello"}]

    _inject_todo_snapshot_internal_note(messages, "")

    assert messages == [{"role": "user", "content": "hello"}]


def test_todo_snapshot_scaffolding_is_not_human_intent():
    snapshot = {
        "role": "user",
        "content": SNAPSHOT,
        "_todo_snapshot_synthetic": True,
    }

    assert _is_real_user_message(snapshot) is False


def test_stale_todo_snapshot_stripping_preserves_latest_user_request():
    content = f"actual latest request\n\n{SNAPSHOT}"

    assert _strip_stale_todo_snapshot(content) == "actual latest request"


def test_todo_snapshot_refreshes_prior_note_and_merged_user_artifact():
    messages = [
        {
            "role": "assistant",
            "content": f"{_TODO_INTERNAL_NOTE_PREFIX}\nold snapshot",
            "_todo_snapshot_internal": True,
        },
        {"role": "user", "content": f"actual latest request\n\n{SNAPSHOT}"},
    ]
    refreshed = f"{TODO_INJECTION_HEADER}\n- [>] ship. Verify (in_progress)"

    _inject_todo_snapshot_internal_note(messages, refreshed)

    notes = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("_todo_snapshot_internal")
    ]
    assert len(notes) == 1
    assert notes[0]["content"] == f"{_TODO_INTERNAL_NOTE_PREFIX}\n{refreshed}"
    assert messages[-1] == {"role": "user", "content": "actual latest request"}


def test_completed_todos_remove_persisted_stale_artifacts():
    messages = [
        {
            "role": "assistant",
            "content": f"{_TODO_INTERNAL_NOTE_PREFIX}\n{SNAPSHOT}",
            "_todo_snapshot_internal": True,
        },
        {
            "role": "user",
            "content": SNAPSHOT,
            "_todo_snapshot_synthetic": True,
        },
        {"role": "user", "content": "actual latest request"},
    ]

    _inject_todo_snapshot_internal_note(messages, "")

    assert messages == [{"role": "user", "content": "actual latest request"}]


def test_zero_user_continuation_keeps_internal_note_before_synthetic_anchor():
    messages = [
        {"role": "assistant", "content": "compressed summary"},
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
    ]

    _inject_todo_snapshot_internal_note(messages, SNAPSHOT)

    assert messages[-1]["content"] == COMPRESSION_CONTINUATION_USER_CONTENT
    assert messages[-2]["_todo_snapshot_internal"] is True
    assert not any(_is_real_user_message(message) for message in messages)


def test_multimodal_user_tail_drops_stale_snapshot_part_without_losing_content():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    messages = [
        {
            "role": "user",
            "content": [
                image,
                {"type": "text", "text": SNAPSHOT},
                {"type": "text", "text": "actual latest request"},
            ],
        }
    ]

    _inject_todo_snapshot_internal_note(messages, SNAPSHOT)

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == [
        image,
        {"type": "text", "text": "actual latest request"},
    ]
    assert TODO_INJECTION_HEADER not in str(messages[-1]["content"])
    assert messages[-2]["_todo_snapshot_internal"] is True


def test_refresh_keeps_tool_pairs_and_drops_stale_wire_sidecars():
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "todo", "arguments": "{}"}}
    messages = [
        {"role": "system", "content": "unchanged cache prefix"},
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": f"{_TODO_INTERNAL_NOTE_PREFIX}\n{SNAPSHOT}",
         "tool_calls": [tool_call], "api_content": "old assistant wire copy", "_todo_snapshot_internal": True},
        {"role": "tool", "tool_call_id": "call-1", "content": "saved"},
        {"role": "user", "content": [{"type": "text", "text": "new request"},
                                      {"type": "text", "text": SNAPSHOT}],
         "api_content": "old user wire copy"},
    ]
    _inject_todo_snapshot_internal_note(messages, SNAPSHOT)
    assert messages[0] == {"role": "system", "content": "unchanged cache prefix"}
    assert messages[2]["tool_calls"] == [tool_call]
    assert messages[3]["tool_call_id"] == "call-1"
    assert messages[-1]["content"] == [{"type": "text", "text": "new request"}]
    assert "api_content" not in messages[2] and "api_content" not in messages[-1]
    assert sum(_TODO_INTERNAL_NOTE_PREFIX in str(m.get("content")) for m in messages) == 1
    assert repair_message_sequence(None, messages) == 0


def test_unhydrated_snapshot_is_preserved_once_then_completed_authoritatively():
    from types import SimpleNamespace
    from agent.compression_todo import fold_todo_snapshot
    messages = [{"role": "user", "content": [
        {"type": "text", "text": SNAPSHOT},
        {"type": "text", "text": "new unrelated request"},
    ]}]
    store = SimpleNamespace(format_for_injection=lambda: "", has_items=lambda: False)
    agent = SimpleNamespace(_todo_store=store)
    fold_todo_snapshot(agent, messages)
    note = messages[-2]
    assert note["role"] == "assistant"
    assert SNAPSHOT in note["content"] and "new unrelated request" not in note["content"]
    assert messages[-1]["content"] == [{"type": "text", "text": "new unrelated request"}]
    fold_todo_snapshot(agent, messages)
    assert sum(_TODO_INTERNAL_NOTE_PREFIX in str(m.get("content")) for m in messages) == 1
    store.has_items = lambda: True
    fold_todo_snapshot(agent, messages)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "new unrelated request"}]}]
