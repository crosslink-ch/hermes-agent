from agent.conversation_compression import _inject_todo_snapshot_internal_note


def test_todo_snapshot_inserted_before_latest_user_message():
    messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "actual latest request"},
    ]

    _inject_todo_snapshot_internal_note(
        messages,
        "[Your active task list was preserved across context compression]\n- [>] report. Summarize (in_progress)",
    )

    assert messages[-1] == {"role": "user", "content": "actual latest request"}
    assert messages[-2]["role"] == "assistant"
    assert "not a user message" in messages[-2]["content"]
    assert "[Your active task list was preserved" in messages[-2]["content"]


def test_empty_todo_snapshot_noops():
    messages = [{"role": "user", "content": "hello"}]

    _inject_todo_snapshot_internal_note(messages, "")

    assert messages == [{"role": "user", "content": "hello"}]
