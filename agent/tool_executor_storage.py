"""Tool result bookkeeping, context-window budgets and wire argument parsing."""
from __future__ import annotations

from typing import Any, Optional
from tools.budget_config import BudgetConfig

def _record_persisted_path_for_stub(agent, tool_call_id: str, function_result) -> None:
    """Record the spillover file path so a later result-reference stub can't dangle (best-effort)."""
    from agent.tool_executor import _is_multimodal_tool_result, extract_persisted_path, logger
    try:
        candidates = [function_result] if isinstance(function_result, str) else [
            function_result.get("text_summary"),
            *(p.get("text") for p in function_result.get("content") or [] if isinstance(p, dict)),
        ] if _is_multimodal_tool_result(function_result) else []
        path = next((p for p in map(extract_persisted_path, candidates) if p), None)
        if path:
            agent._tool_guardrails.record_persisted_result(tool_call_id, path)
    except Exception as exc:
        logger.debug("persisted-path record for result stub failed: %s", exc, exc_info=True)


def _budget_for_agent(agent) -> BudgetConfig:
    """Tool-result BudgetConfig scaled to the agent's context window. Unknown length goes
    through ``budget_for_context_window(None)`` (not DEFAULT_BUDGET) so the MCP threshold
    override still applies.

    Large-context models keep the historical 100K/200K char defaults; small models (e.g. a 65K-token local
    model switched into mid-session) get a budget proportional to their window so a single large tool result
    can't push the request past the model's limit (#23767). Falls back to the default budget when the
    context length isn't resolvable.
    """
    from agent.tool_executor import DEFAULT_BUDGET, budget_for_context_window
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        return budget_for_context_window(int(ctx) if ctx else None)
    except (TypeError, ValueError, OverflowError):
        return DEFAULT_BUDGET


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """Parse model-emitted arguments without repairing or coercing them."""
    from agent.tool_executor import json
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {"error": "Invalid tool arguments", "message": "Tool arguments must be a valid JSON object; tool was not executed."},
        ensure_ascii=False,
    )
