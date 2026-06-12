#!/usr/bin/env python3
"""Deterministic Hermes context-compaction demo for debugging.

Run from the hermes2 checkout:

    .venv/bin/python scripts/demo_context_compression.py

or save the markdown report:

    .venv/bin/python scripts/demo_context_compression.py --output /tmp/compaction-demo.md

The demo avoids network/model calls by monkeypatching only
``ContextCompressor._generate_summary`` with a deterministic summary. It still
executes the real compaction assembly code and the recent continuity fixes:

* ``ContextCompressor.compress(...)`` decides the protected head/tail and inserts
  the compaction handoff summary.
* ``_inject_todo_snapshot_internal_note(...)`` preserves the todo snapshot as an
  internal assistant note before the latest real user message.
* ``TheChatAdapter._session_payload_for_context(...)`` prefers the live
  compressed child session over a stale cached parent session.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import patch

# Let the script run directly from scripts/ without requiring PYTHONPATH=.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.conversation_compression import _inject_todo_snapshot_internal_note
from gateway.config import PlatformConfig
from gateway.platforms.thechat import TheChatAdapter

TODO_SNAPSHOT = (
    "[Your active task list was preserved across context compression]\n"
    "- [>] report: Summarize likely cause and next fix/verification steps (in_progress)\n"
    "- [ ] verify: Run focused compaction continuity tests (pending)"
)

LATEST_REAL_USER = "Before committing, show me a small compaction demo for debugging."


def _tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def build_demo_history() -> list[dict[str, Any]]:
    """A synthetic but realistic pre-compaction message window."""
    return [
        {
            "role": "system",
            "content": "System prompt: You are Hermes Agent working in /home/bruno/projects/hermes2.",
        },
        {
            "role": "user",
            "content": "Start a debugging session for Hermes/TheChat context compaction.",
        },
        {
            "role": "assistant",
            "content": "I'll inspect the session lineage and compression code before changing anything.",
        },
        {
            "role": "user",
            "content": "After compaction, TheChat seemed to answer an older clarification instead of my latest request.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call(
                    "call_logs",
                    "terminal",
                    json.dumps({"command": "grep -E 'compression|summary' ~/.hermes/logs/errors.log"}),
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_logs",
            "name": "terminal",
            "content": "errors.log: Failed to generate context summary: Connection error\nstate.db: root session had compressed children #2..#8",
        },
        {
            "role": "assistant",
            "content": "Compression did happen, but TheChat's stored continuity pointer still referenced the stale root session.",
        },
        {
            "role": "user",
            "content": "Also check whether the preserved todo/task snapshot can become the newest user message.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call(
                    "call_search",
                    "search_files",
                    json.dumps({"pattern": "format_for_injection|Your active task list", "path": "agent"}),
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "name": "search_files",
            "content": "agent/conversation_compression.py appended the todo snapshot as role='user' after compression.",
        },
        {
            "role": "assistant",
            "content": "Found bug: a synthetic todo snapshot could become the latest user-like turn after compaction.",
        },
        {
            "role": "user",
            "content": "Fix it so the latest real request stays the recency anchor.",
        },
        {
            "role": "assistant",
            "content": "Implemented an internal assistant continuity note and added regression tests.",
        },
        {
            "role": "user",
            "content": LATEST_REAL_USER,
        },
    ]


def make_demo_compressor() -> ContextCompressor:
    """Create a compressor without consulting live model metadata."""
    with patch("agent.context_compressor.get_model_context_length", return_value=64_000):
        return ContextCompressor(
            model="demo-model-no-api-call",
            threshold_percent=0.50,
            protect_first_n=1,
            protect_last_n=3,
            summary_target_ratio=0.20,
            quiet_mode=True,
        )


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def message_preview(msg: dict[str, Any], *, width: int = 120) -> str:
    role = msg.get("role", "?")
    content = _text_content(msg.get("content"))
    if role == "assistant" and msg.get("tool_calls"):
        names = [tc.get("function", {}).get("name", "?") for tc in msg.get("tool_calls") or []]
        content = f"tool_calls: {', '.join(names)}"
    elif role == "tool":
        content = f"{msg.get('name') or msg.get('tool_call_id')}: {content}"
    content = " ".join(content.split())
    return textwrap.shorten(content, width=width, placeholder=" …") if content else "∅"


def render_history(title: str, messages: Iterable[dict[str, Any]], *, full: bool = False) -> str:
    rows = [f"### {title}"]
    for idx, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = _text_content(msg.get("content"))
        flags: list[str] = []
        if content.lstrip().startswith(SUMMARY_PREFIX):
            flags.append("COMPACTION SUMMARY")
        if "Internal continuity note preserved across context compression" in content:
            flags.append("TODO SNAPSHOT AS INTERNAL NOTE")
        if role == "user" and content.startswith("[Your active task list was preserved"):
            flags.append("BUGGY SYNTHETIC USER")
        if content == LATEST_REAL_USER:
            flags.append("LATEST REAL USER")
        suffix = f"  ← {'; '.join(flags)}" if flags else ""
        if full:
            rows.append(f"{idx:02d}. **{role}**{suffix}\n\n```text\n{content or message_preview(msg, width=10_000)}\n```")
        else:
            rows.append(f"{idx:02d}. **{role}** — {message_preview(msg)}{suffix}")
    return "\n".join(rows)


def build_deterministic_summary(
    compressor: ContextCompressor,
    turns_to_summarize: list[dict[str, Any]],
    focus_topic: str | None,
) -> str:
    compacted_indices = [int(msg.get("_demo_index", -1)) for msg in turns_to_summarize]
    compacted_indices = [idx for idx in compacted_indices if idx >= 0]
    compacted_range = f"{min(compacted_indices)}..{max(compacted_indices)}" if compacted_indices else "unknown"
    user_turns = [message_preview(msg, width=180) for msg in turns_to_summarize if msg.get("role") == "user"]
    tool_turns = [message_preview(msg, width=180) for msg in turns_to_summarize if msg.get("role") == "tool"]

    body = f"""## Active Task
Continue from the latest user message after this summary. In this demo that message is: {LATEST_REAL_USER!r}

## Goal
Show what Hermes context compaction preserves when it replaces older middle turns with a handoff summary.

## Completed Actions
- Investigated a TheChat continuity issue where a stale parent/root session id could survive after compaction.
- Investigated the todo snapshot issue where a synthetic task-list marker could be appended as role='user'.
- Implemented the fix that preserves todos as an internal assistant continuity note before the latest real user request.

## Active State
Demo focus: {focus_topic or 'none'}.
The protected head and protected tail stay as normal messages; middle turns {compacted_range} are represented by this summary.

## Pending User Asks
- {LATEST_REAL_USER}

## Relevant Files
- agent/conversation_compression.py
- agent/context_compressor.py
- gateway/platforms/thechat.py
- tests/agent/test_conversation_compression_todo_snapshot.py
- tests/gateway/test_thechat_session_continuity.py

## Compacted User Turns
{chr(10).join(f'- {turn}' for turn in user_turns) or '- None'}

## Compacted Tool Evidence
{chr(10).join(f'- {turn}' for turn in tool_turns) or '- None'}

## Remaining Work
Run focused tests and inspect this demo output before deciding whether to commit.
""".strip()
    return compressor._with_summary_prefix(body)


def find_summary_message(messages: list[dict[str, Any]]) -> tuple[int | None, str]:
    for idx, msg in enumerate(messages):
        content = _text_content(msg.get("content"))
        if content.lstrip().startswith(SUMMARY_PREFIX):
            return idx, content
    return None, ""


def latest_user_content(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _text_content(msg.get("content"))
    return ""


def run_compaction_demo() -> dict[str, Any]:
    history = build_demo_history()
    for idx, msg in enumerate(history):
        msg["_demo_index"] = idx
    compressor = make_demo_compressor()
    captured_turns: list[dict[str, Any]] = []

    def fake_generate_summary(turns_to_summarize: list[dict[str, Any]], focus_topic: str | None = None) -> str:
        captured_turns[:] = list(turns_to_summarize)
        return build_deterministic_summary(compressor, captured_turns, focus_topic)

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compacted_without_todo = compressor.compress(
            history,
            current_tokens=75_000,
            focus_topic="compaction continuity + todo snapshot placement",
            force=True,
        )

    fixed_after = [dict(msg) for msg in compacted_without_todo]
    _inject_todo_snapshot_internal_note(fixed_after, TODO_SNAPSHOT)

    # Simulate the old buggy shape just for comparison. This is intentionally
    # NOT the production code path; it shows why appending the todo snapshot as a
    # synthetic user message was dangerous.
    old_buggy_after = [dict(msg) for msg in compacted_without_todo]
    old_buggy_after.append({"role": "user", "content": TODO_SNAPSHOT})

    summary_idx, summary_content = find_summary_message(fixed_after)
    summarized_indices = [int(msg.get("_demo_index", -1)) for msg in captured_turns]
    summarized_indices = [idx for idx in summarized_indices if idx >= 0]
    return {
        "before": history,
        "summarized_indices": summarized_indices,
        "summarized_message_count": len(captured_turns),
        "summarized_roles": [msg.get("role") for msg in captured_turns],
        "after_fixed": fixed_after,
        "after_buggy_comparison": old_buggy_after,
        "summary_index": summary_idx,
        "summary_content": summary_content,
        "latest_user_before": latest_user_content(history),
        "latest_user_after_fixed": latest_user_content(fixed_after),
        "latest_user_after_buggy": latest_user_content(old_buggy_after),
    }


def run_thechat_continuity_demo() -> dict[str, Any]:
    adapter = TheChatAdapter(PlatformConfig())
    context = {
        "session": {
            "sessionId": "stale-parent-before-compression",
            "sessionKey": "thechat:conversation:demo",
            "lineageRootId": "stale-parent-before-compression",
        },
        "event": SimpleNamespace(
            hermes_session={
                "sessionId": "compressed-child-after-compression",
                "sessionKey": "thechat:conversation:demo",
                "lineageRootId": "stale-parent-before-compression",
            }
        ),
    }
    emitted = adapter._session_payload_for_context(context, reason="invocation.completed")
    return {
        "cached_context_session_before": "stale-parent-before-compression",
        "live_event_session": "compressed-child-after-compression",
        "emitted_session_payload": emitted,
        "context_session_after_call": context.get("session"),
    }


def render_report(*, full: bool = False) -> str:
    demo = run_compaction_demo()
    thechat = run_thechat_continuity_demo()

    lines: list[str] = [
        "# Hermes context compaction demo",
        "",
        "This is a deterministic local demo: no LLM/API call is made. The only mocked part is summary text generation, so the demo can run offline. The surrounding compaction, todo-snapshot insertion, and TheChat session-payload code are the real project code.",
        "",
        "## Code paths exercised",
        "- `agent.context_compressor.ContextCompressor.compress(...)`",
        "- `agent.conversation_compression._inject_todo_snapshot_internal_note(...)`",
        "- `gateway.platforms.thechat.TheChatAdapter._session_payload_for_context(...)`",
        "",
        "## Safety check: latest user recency anchor",
        f"- Latest real user before compaction: `{demo['latest_user_before']}`",
        f"- Latest user after fixed compaction: `{demo['latest_user_after_fixed']}`",
        f"- Latest user in simulated old buggy shape: `{demo['latest_user_after_buggy'].splitlines()[0]}`",
        f"- Middle message indexes compacted into the summary: `{demo['summarized_indices']}` ({demo['summarized_message_count']} messages)",
        "",
        render_history("Before compaction: full conversation window", demo["before"], full=full),
        "",
        "## Compaction content inserted into the after-history",
        f"Summary message index after compaction: `{demo['summary_index']}`",
        "",
        "```text",
        demo["summary_content"],
        "```",
        "",
        render_history("After compaction: fixed/current behavior", demo["after_fixed"], full=full),
        "",
        "## Why the todo fix matters",
        "The old shape below is simulated only for comparison: it appends the todo snapshot as a synthetic `user` turn, making the task-list marker the newest user message. The fixed/current shape above inserts the same text as an internal assistant continuity note before the latest real user message.",
        "",
        render_history("Simulated old buggy after-history", demo["after_buggy_comparison"], full=full),
        "",
        "## TheChat continuity payload demo",
        "This runs the real TheChat adapter helper with a stale cached parent session in `context['session']` and a live compressed child in `event.hermes_session`.",
        "",
        "```json",
        json.dumps(thechat, indent=2, sort_keys=True),
        "```",
        "",
        "Expected: `emitted_session_payload.sessionId` is `compressed-child-after-compression`, not the stale parent.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a deterministic Hermes context-compaction debug demo.")
    parser.add_argument("--full", action="store_true", help="print full message contents in before/after histories")
    parser.add_argument("--output", type=Path, help="write the markdown report to a file instead of stdout")
    args = parser.parse_args()

    report = render_report(full=args.full)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote compaction demo report to {args.output}")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
