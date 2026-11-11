"""Todo continuity across compression: the snapshot is an internal assistant note.

The fork keeps the todo snapshot out of user turns (``agent/compression_todo.py``): it rides an
assistant-role internal note placed before the latest user turn, so the user's own content is
never rewritten and a summary handoff keeps its prefix. These drive the real
``_compress_context`` path against a real SessionDB, like the rotation-state suite whose
builders they share.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import tests.agent.test_compression_rotation_state as _rotation
from hermes_state import SessionDB
from tests.agent.test_compression_rotation_state import _build_agent_with_db, _msgs, _todo_notes

_agent_with_todo = _rotation.TestTodoSnapshotScaffoldingTails._agent_with_todo


class TestTodoSnapshotInternalNote:
    """Todo snapshots remain internal while preserving the latest user turn."""

    def test_multimodal_tail_is_unchanged_when_internal_note_is_inserted(
        self, tmp_path: Path
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MULTIMODAL_ROTATION"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")

        original_parts = [
            {"type": "text", "text": "tail text"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/context.png"},
            },
        ]
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": list(original_parts)},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "inspect image", "status": "pending"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] inspect image"
        )

        # Keep the fake compression beneficial after upstream began accounting
        # for multimodal image-token cost in the no-growth commit guard.
        input_msgs = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i} " + "x" * 400,
            }
            for i in range(20)
        ]
        # The stub retained the original reply and latest multimodal user turn.
        input_msgs[-1]["content"] = "acknowledged"
        input_msgs.append({"role": "user", "content": list(original_parts)})
        compressed, _ = agent._compress_context(
            input_msgs, "sys", approx_tokens=120_000
        )

        assert len(compressed) == 3
        assert [message["role"] for message in compressed] == [
            "user",
            "assistant",
            "user",
        ]
        tail = compressed[-1]
        assert tail["role"] == "user"
        assert tail["content"] == original_parts
        notes = _todo_notes(compressed)
        assert len(notes) == 1
        assert "acknowledged" in notes[0]["content"]
        assert "inspect image" in notes[0]["content"]
        assert compressed[-2] is notes[0]
        assert not any(
            previous.get("role") == current.get("role") == "user"
            for previous, current in itertools.pairwise(compressed)
        )


    def test_snapshot_note_is_persisted_in_place(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_INPLACE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")
        agent.compression_in_place = True
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last user msg"},
        ]
        agent._todo_store._todos = [
            {"id": "t1", "content": "do thing", "status": "in_progress"}
        ]
        agent._todo_store.format_for_injection = (
            lambda: "## Current Tasks\n- [ ] do thing"
        )

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        db_msgs = db.get_messages(agent.session_id)
        last_user = [message for message in db_msgs if message["role"] == "user"][-1]
        assert last_user["content"] == "last user msg"
        notes = _todo_notes(db_msgs)
        assert len(notes) == 1
        assert "do thing" in notes[0]["content"]
        assert db_msgs.index(notes[0]) < db_msgs.index(last_user)


class TestTodoSnapshotScaffoldingTails:
    """Scaffolding tails must never absorb the todo snapshot (#69292)."""

    def test_snapshot_stays_standalone_after_summary_as_user_tail(
        self, tmp_path: Path
    ):
        from agent.context_compressor import SUMMARY_PREFIX, ContextCompressor

        summary_as_user = f"{SUMMARY_PREFIX}\nzero-user summary body"
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = _agent_with_todo(
            db,
            "PARENT_TODO_SUMMARY_TAIL",
            {"role": "user", "content": summary_as_user},
        )

        input_msgs = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i} " + "x" * 400,
            }
            for i in range(20)
        ]
        compressed, _ = agent._compress_context(
            input_msgs, "sys", approx_tokens=120_000
        )

        # The summary handoff prefix must stay at the START of its own
        # message for downstream summary detection.
        summary_rows = [
            message
            for message in compressed
            if str(message.get("content") or "").startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_rows) == 1
        notes = _todo_notes(compressed)
        assert len(notes) == 1
        assert "task A" in notes[0]["content"]
        assert not any(
            "task A" in str(message.get("content") or "")
            for message in compressed
            if message.get("role") == "user"
        )
        # Zero-user provenance remains false because the summary is synthetic
        # and todo continuity now has an assistant role.
        assert not ContextCompressor._transcript_has_real_user_turn(
            [{"role": "user", "content": summary_rows[0]["content"]}]
        )

    def test_stale_snapshot_row_is_refreshed_not_stacked(self, tmp_path: Path):
        from tools.todo_tool import TODO_INJECTION_HEADER

        stale = f"{TODO_INJECTION_HEADER}\n- [ ] t0. old finished task (pending)"
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = _agent_with_todo(
            db,
            "PARENT_TODO_STALE_ROW",
            {"role": "user", "content": stale},
        )

        compressed, _ = agent._compress_context(
            _msgs(), "sys", approx_tokens=120_000
        )

        notes = _todo_notes(compressed)
        assert len(notes) == 1
        assert "task A" in notes[0]["content"]
        assert "old finished task" not in notes[0]["content"]
        assert not any(
            str(message.get("content") or "").startswith(TODO_INJECTION_HEADER)
            for message in compressed
            if message.get("role") == "user"
        )
