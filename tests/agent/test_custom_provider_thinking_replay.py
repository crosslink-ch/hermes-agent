"""Opt-in ``thinking_replay: native`` on a custom provider entry.

An unknown Anthropic Messages relay cannot verify Anthropic signatures, so Hermes strips signed
thinking there. A relay that forwards the blocks unmodified to Anthropic (a self-hosted Claude
proxy inside the cluster) can declare ``thinking_replay: native`` on its ``providers:`` entry and
get the native contract: signed blocks replay across assistant turns and tool loops, and a
signature 400 is healed by the durable suppression. Off unless set; the built-in Kimi and DeepSeek
rules still win.
"""

from __future__ import annotations

import copy
import logging

import pytest

import hermes_yaml as yaml
from agent import anthropic_thinking_policy as policy_mod
from agent.anthropic_message_convert import _manage_thinking_signatures, convert_messages_to_anthropic
from agent.anthropic_thinking_policy import anthropic_thinking_route, native_anthropic_preserves_prior_thinking
from agent.anthropic_thinking_replay import tracks_rejected_thinking
from hermes_cli import config as config_mod
from hermes_cli.config import (
    _VALID_CUSTOM_PROVIDER_FIELDS,
    _custom_provider_entry_to_provider_config,
    _entries_for_route,
    _normalize_custom_provider_entry,
    get_custom_provider_thinking_replay,
    validate_config_structure,
)

CLIPROXY = "http://cliproxyapi.cliproxy-tp.svc.cluster.local:8317"
MODELS = ("claude-opus-5-5", "claude-fable-5-2")
ALL = ["sig_prior", "sig_loop1", "sig_loop2"]


def _signed(sig):
    return {"type": "thinking", "thinking": f"thought {sig}", "signature": sig}


def _loop_step(n):
    call = {"id": f"t{n}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    return [
        {"role": "assistant", "content": "", "tool_calls": [call], "reasoning_details": [_signed(f"sig_loop{n}")],
         "anthropic_content_blocks": [_signed(f"sig_loop{n}"), {"type": "tool_use", "id": f"t{n}", "name": "read_file", "input": {}}]},
        {"role": "tool", "tool_call_id": f"t{n}", "content": "ok"},
    ]


def _history():
    """A prior signed assistant turn, then an in-flight two-step tool loop."""
    return [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_details": [_signed("sig_prior")]},
        {"role": "user", "content": "Q2"},
        *_loop_step(1),
        *_loop_step(2),
    ]


def _wire_blocks(base_url, model, history=None):
    _system, wire = convert_messages_to_anthropic(history or _history(), base_url=base_url, model=model)
    return [b for m in wire if m["role"] == "assistant" for b in m["content"]]


def _signatures(blocks):
    return [b.get("signature") for b in blocks if b.get("type") == "thinking"]


def _agent(base_url, model, api_mode="anthropic_messages"):
    return type("A", (), {"api_mode": api_mode, "base_url": base_url, "model": model})()


def _provider(**overrides):
    return {"base_url": CLIPROXY, "api_mode": "anthropic_messages", "thinking_replay": "native", **overrides}


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    """Write a real ``config.yaml`` into an isolated HERMES_HOME; the route is read from disk."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _write(config):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        return config

    return _write


@pytest.fixture
def cliproxy_native(write_config):
    return write_config({"providers": {"cliproxy": _provider()}})


# 1. Default: an unconfigured in-cluster relay is a third party and loses its signed thinking.


@pytest.mark.parametrize("model", MODELS)
def test_unconfigured_cluster_url_is_third_party_and_strips_signed_thinking(write_config, model):
    write_config({})
    assert anthropic_thinking_route(CLIPROXY, model) == "third_party"
    assert tracks_rejected_thinking(_agent(CLIPROXY, model)) is False
    assert native_anthropic_preserves_prior_thinking(CLIPROXY, model) is False

    blocks = _wire_blocks(CLIPROXY, model)
    assert _signatures(blocks) == []
    assert [b["id"] for b in blocks if b.get("type") == "tool_use"] == ["t1", "t2"]

    wire = [
        {"role": "assistant", "content": [_signed("sig_a"), {"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {"role": "assistant", "content": [_signed("sig_b")]},
    ]
    _manage_thinking_signatures(wire, CLIPROXY, model)
    assert wire[0]["content"] == [{"type": "text", "text": "A"}]
    assert wire[2]["content"] == [{"type": "text", "text": "(thinking elided)"}]


# 2. Opt-in: the same URL under providers.cliproxy with thinking_replay: native.


@pytest.mark.parametrize("model", MODELS)
def test_native_opt_in_preserves_signed_thinking_across_turns_and_tool_loops(cliproxy_native, model):
    assert get_custom_provider_thinking_replay(CLIPROXY) == "native"
    assert anthropic_thinking_route(CLIPROXY, model) == "native"
    assert tracks_rejected_thinking(_agent(CLIPROXY, model)) is True
    assert native_anthropic_preserves_prior_thinking(CLIPROXY, model) is True

    history = _history()
    before = copy.deepcopy(history)
    blocks = _wire_blocks(CLIPROXY, model, history)
    assert _signatures(blocks) == ALL
    assert [b for b in blocks if b.get("type") == "thinking"] == [_signed(sig) for sig in ALL]
    for step in (1, 2):  # each loop step's thinking precedes its tool_use
        types = [b["type"] for b in blocks if b.get("signature") == f"sig_loop{step}" or b.get("id") == f"t{step}"]
        assert types == ["thinking", "tool_use"]
    assert history == before

    # A later user turn closes the loop; every earlier signed turn still replays.
    history += [{"role": "assistant", "content": "A2", "reasoning_details": [_signed("sig_final")]},
                {"role": "user", "content": "Q3"}]
    assert _signatures(_wire_blocks(CLIPROXY, model, history)) == [*ALL, "sig_final"]

    wire = [
        {"role": "assistant", "content": [_signed("sig_a"), {"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {"role": "assistant", "content": [_signed("sig_b"), {"type": "text", "text": "B"}]},
    ]
    _manage_thinking_signatures(wire, CLIPROXY, model)
    assert _signatures([b for m in wire if m["role"] == "assistant" for b in m["content"]]) == ["sig_a", "sig_b"]


def test_native_opt_in_only_tracks_rejections_on_anthropic_messages(cliproxy_native):
    assert tracks_rejected_thinking(_agent(CLIPROXY, MODELS[0], api_mode="chat_completions")) is False


def test_native_opt_in_keeps_the_model_generation_rule(cliproxy_native):
    """``native`` is the Anthropic contract, not replay-everything: a last-turn-only model still
    gets its in-flight tool loop back and nothing older."""
    assert anthropic_thinking_route(CLIPROXY, "claude-sonnet-4-5") == "native"
    assert _signatures(_wire_blocks(CLIPROXY, "claude-sonnet-4-5")) == ["sig_loop1", "sig_loop2"]


def test_legacy_custom_providers_list_entry_opts_in(write_config):
    write_config({"custom_providers": [{"name": "cliproxy", **_provider()}]})
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "native"


@pytest.mark.parametrize("entry", [
    _provider(thinking_replay="third_party"),
    _provider(thinking_replay=""),
    _provider(thinking_replay="signed"),
    _provider(thinking_replay=True),
    _provider(thinking_replay=["native"]),
    _provider(enabled=False),
    {"base_url": CLIPROXY, "api_mode": "anthropic_messages"},
    _provider(base_url="http://other-proxy.cliproxy-tp.svc.cluster.local:8317"),
    _provider(base_url="http://cliproxyapi.cliproxy-tp.svc.cluster.local:8318"),
])
def test_anything_but_a_matching_native_declaration_keeps_the_default(write_config, entry):
    write_config({"providers": {"cliproxy": entry}})
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "third_party"
    assert tracks_rejected_thinking(_agent(CLIPROXY, MODELS[0])) is False
    assert _signatures(_wire_blocks(CLIPROXY, MODELS[0])) == []


def test_config_errors_fail_closed_to_third_party(cliproxy_native, monkeypatch):
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "native"

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "third_party"
    assert _signatures(_wire_blocks(CLIPROXY, MODELS[0])) == []


def test_a_failed_lookup_is_logged_once_without_private_exception_text(cliproxy_native, monkeypatch, caplog):
    """Failing closed must not be silent: an operator who opted a relay into native replay sees why
    signed thinking is being stripped again. The route is resolved several times per request, so one
    record per kind of failure. A config error can quote the base_url (which may embed credentials)
    or a config value, so the record names the exception type only: no message, no traceback."""
    secret_url = "http://user:hunter2@cliproxyapi.cliproxy-tp.svc.cluster.local:8317"

    def _boom():
        raise RuntimeError(f"config unreadable: {secret_url}; api_key=private-config-value")

    def _boom_differently():
        raise ValueError(f"bad value at {secret_url}; api_key=private-config-value")

    monkeypatch.setattr(policy_mod, "_LOGGED_REPLAY_LOOKUP_FAILURES", set())
    with caplog.at_level(logging.WARNING, logger=policy_mod.logger.name):
        monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
        for _ in range(3):
            assert anthropic_thinking_route(secret_url, MODELS[0]) == "third_party"
        assert _signatures(_wire_blocks(CLIPROXY, MODELS[0])) == []
        monkeypatch.setattr(config_mod, "load_config_readonly", _boom_differently)
        for _ in range(3):
            assert anthropic_thinking_route(secret_url, MODELS[0]) == "third_party"
        monkeypatch.setattr(config_mod, "load_config_readonly", _boom)
        assert anthropic_thinking_route(secret_url, MODELS[0]) == "third_party"

    records = [r for r in caplog.records if r.name == policy_mod.logger.name]
    assert [r.levelno for r in records] == [logging.WARNING, logging.WARNING]
    assert ["RuntimeError" in r.getMessage() for r in records] == [True, False]
    assert ["ValueError" in r.getMessage() for r in records] == [False, True]
    for record in records:
        assert not record.exc_info and not record.exc_text and not record.stack_info
        rendered = logging.Formatter().format(record) + repr(record.args)
        for private in ("hunter2", "cliproxyapi", "private-config-value", "config unreadable",
                        "bad value", "Traceback"):
            assert private not in rendered
            assert private not in caplog.text


def test_a_successful_lookup_logs_nothing(cliproxy_native, monkeypatch, caplog):
    monkeypatch.setattr(policy_mod, "_LOGGED_REPLAY_LOOKUP_FAILURES", set())
    with caplog.at_level(logging.DEBUG, logger=policy_mod.logger.name):
        assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "native"
    assert [r for r in caplog.records if r.name == policy_mod.logger.name] == []


def test_route_lookup_does_not_mutate_the_cached_config(cliproxy_native):
    cached = config_mod.load_config_readonly()
    before = copy.deepcopy(cached)
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "native"
    assert config_mod.load_config_readonly() == before


def test_values_are_normalized_lowercase_and_stripped(write_config):
    write_config({"providers": {"cliproxy": _provider(thinking_replay="  NATIVE ")}})
    assert get_custom_provider_thinking_replay(CLIPROXY) == "native"
    assert anthropic_thinking_route(CLIPROXY, MODELS[0]) == "native"
    entries = [{"name": "p", "base_url": CLIPROXY, "thinking_replay": " Third_Party "}]
    assert get_custom_provider_thinking_replay(CLIPROXY, custom_providers=entries) == "third_party"
    assert get_custom_provider_thinking_replay(CLIPROXY, custom_providers=[]) is None
    assert get_custom_provider_thinking_replay("", custom_providers=entries) is None


# 3. Route identity: the same normalization as _entries_for_route, so a saved /model pin matches.


@pytest.mark.parametrize("entry_url, pinned_url, matches", [
    (CLIPROXY, CLIPROXY + "/", True),
    (CLIPROXY + "/", CLIPROXY, True),
    (CLIPROXY, "HTTP://CliProxyAPI.cliproxy-tp.svc.cluster.local:8317/", True),
    (CLIPROXY + "/v1", CLIPROXY + "/v1", True),
    (CLIPROXY + "/v1", CLIPROXY + "/v1/", True),
    (CLIPROXY + "/v1/", CLIPROXY + "/v1", True),
    # /v1 is a different route identity: _entries_for_route does not equate it with the bare host.
    (CLIPROXY, CLIPROXY + "/v1", False),
    (CLIPROXY + "/v1", CLIPROXY, False),
])
def test_route_matching_has_parity_with_entries_for_route(write_config, entry_url, pinned_url, matches):
    config = write_config({"providers": {"cliproxy": _provider(base_url=entry_url)}})
    assert bool(list(_entries_for_route(pinned_url, None, config))) is matches
    expected = "native" if matches else "third_party"
    assert get_custom_provider_thinking_replay(pinned_url, config=config) == ("native" if matches else None)
    for model in MODELS:
        assert anthropic_thinking_route(pinned_url, model) == expected
        assert tracks_rejected_thinking(_agent(pinned_url, model)) is matches
        assert _signatures(_wire_blocks(pinned_url, model)) == (ALL if matches else [])


# 4. Built-in model rules win over the opt-in.


@pytest.mark.parametrize("model, route", [
    ("kimi-k2.5", "kimi"),
    ("moonshotai/kimi-k2.5", "kimi"),
    ("deepseek-v4-pro", "deepseek"),
    ("deepseek-reasoner", "deepseek"),
])
def test_kimi_and_deepseek_model_rules_win_over_the_opt_in(cliproxy_native, model, route):
    assert get_custom_provider_thinking_replay(CLIPROXY) == "native"
    assert anthropic_thinking_route(CLIPROXY, model) == route
    assert tracks_rejected_thinking(_agent(CLIPROXY, model)) is False
    assert native_anthropic_preserves_prior_thinking(CLIPROXY, model) is False


def test_deepseek_model_on_the_opted_in_route_still_strips_signed_blocks(cliproxy_native):
    unsigned = {"type": "thinking", "thinking": "plain"}
    wire = [{"role": "assistant", "content": [_signed("sig_a"), unsigned, {"type": "text", "text": "A"}]}]
    _manage_thinking_signatures(wire, CLIPROXY, "deepseek-v4-pro")
    assert wire[0]["content"] == [unsigned, {"type": "text", "text": "A"}]


# 5. Config surface: normalization and validation accept the key and ignore bad values.


def test_normalize_accepts_thinking_replay_without_unknown_key_warnings(caplog):
    with caplog.at_level(logging.WARNING):
        normalized = _normalize_custom_provider_entry(_provider(), provider_key="cliproxy-tr-accept")
        camel = _normalize_custom_provider_entry(
            {"base_url": CLIPROXY, "thinkingReplay": "native"}, provider_key="cliproxy-tr-camel")
    assert normalized["thinking_replay"] == "native"
    assert normalized["api_mode"] == "anthropic_messages"
    assert camel["thinking_replay"] == "native"
    assert "unknown config keys" not in caplog.text


@pytest.mark.parametrize("raw, stored", [
    ("native", "native"),
    (" Native\n", "native"),
    ("THIRD_PARTY", "third_party"),
    ("", None),
    ("   ", None),
    ("signed", None),
    ("anthropic", None),
    (True, None),
    (1, None),
    (None, None),
    (["native"], None),
    ({"mode": "native"}, None),
])
def test_normalize_stores_only_known_thinking_replay_values(caplog, raw, stored):
    with caplog.at_level(logging.WARNING):
        normalized = _normalize_custom_provider_entry(
            _provider(thinking_replay=raw), provider_key=f"cliproxy-tr-{raw!r}")
    assert normalized is not None and normalized["base_url"] == CLIPROXY
    assert normalized.get("thinking_replay") == stored
    assert "unknown config keys" not in caplog.text
    provider_config = _custom_provider_entry_to_provider_config(
        {"name": "cliproxy", **_provider(thinking_replay=raw)})
    assert provider_config.get("thinking_replay") == stored


def test_normalize_does_not_mutate_the_entry():
    entry = _provider(thinking_replay=" NATIVE ")
    before = copy.deepcopy(entry)
    _normalize_custom_provider_entry(entry, provider_key="cliproxy-tr-nomutate")
    assert entry == before


def test_config_validation_accepts_thinking_replay(caplog):
    assert "thinking_replay" in _VALID_CUSTOM_PROVIDER_FIELDS
    assert config_mod.get_custom_provider_thinking_replay is get_custom_provider_thinking_replay
    for config in (
        {"providers": {"cliproxy-tr-validate": _provider()}},
        {"custom_providers": [{"name": "cliproxy-tr-validate-list", **_provider()}]},
        {"providers": {"cliproxy-tr-validate-bad": _provider(thinking_replay="bogus")}},
    ):
        with caplog.at_level(logging.WARNING):
            issues = validate_config_structure(config)
        assert not [issue for issue in issues if "thinking_replay" in f"{issue.message} {issue.hint}"]
        assert not [issue for issue in issues if issue.severity == "error"]
    assert "unknown config keys" not in caplog.text
