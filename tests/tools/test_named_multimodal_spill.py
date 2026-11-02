"""Named producing-target authority also covers the base's multimodal spills."""
from tools import tool_result_storage as storage


def test_multimodal_budget_with_missing_target_env_never_spills_to_host(monkeypatch):
    writes = []
    monkeypatch.setattr(storage, "_write_to_spillover", lambda *a: writes.append(a) or "/host/wrong")
    messages = [{"role": "tool", "tool_call_id": "named", "content": [
        {"type": "text", "text": "x" * 250000}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]}]
    storage.enforce_turn_budget(messages, env_resolver=lambda _: None,
                                config=storage.BudgetConfig(turn_budget=200000))
    assert writes == []
    assert "Truncated" in messages[0]["content"][0]["text"]


def test_multimodal_saved_output_hints_name_the_producing_target():
    from agent.tool_executor import _persisted_target_hint
    parts = [{"type": "text", "text": "<persisted-output>saved</persisted-output>"},
             {"type": "image_url", "image_url": {"url": "fixture"}}]
    hinted = _persisted_target_hint(parts, "remote", "generation-a")
    assert 'execution_target="remote"' in hinted[0]["text"]
    assert 'runtime_scope="generation-a"' in hinted[0]["text"]
    assert hinted[1] == parts[1]
    assert "execution_target" not in parts[0]["text"]
