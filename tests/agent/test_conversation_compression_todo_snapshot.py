from agent.conversation_compression import (
    _is_real_user_message,
    _strip_stale_todo_snapshot,
)


def test_todo_snapshot_scaffolding_is_not_human_intent():
    snapshot = {
        "role": "user",
        "content": (
            "[Your active task list was preserved across context compression]\n"
            "- [>] report. Summarize (in_progress)"
        ),
        "_todo_snapshot_synthetic": True,
    }

    assert _is_real_user_message(snapshot) is False


def test_stale_todo_snapshot_stripping_preserves_latest_user_request():
    content = (
        "actual latest request\n\n"
        "[Your active task list was preserved across context compression]\n"
        "- [>] report. Summarize (in_progress)"
    )

    assert _strip_stale_todo_snapshot(content) == "actual latest request"
