from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
from agent.conversation_compression import (
    _TODO_INTERNAL_NOTE_PREFIX,
    _inject_todo_snapshot_internal_note,
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
