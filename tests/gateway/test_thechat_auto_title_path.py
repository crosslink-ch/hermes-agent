"""Regression coverage for TheChat auto-title wiring through the live gateway turn path."""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.title_generator as title_generator
import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _ImmediateAgent:
    """Minimal agent that completes one exchange without tools or network I/O."""

    def __init__(self, **kwargs):
        self.tools = []
        self.model = kwargs.get("model", "test-model")
        self.provider = kwargs.get("provider", "test-provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.session_id = kwargs.get("session_id", "session-1")
        self.context_compressor = None
        self.is_interrupted = False

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        **_kwargs,
    ):
        return {
            "final_response": "The title path completed.",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


class _RecordingTheChatAdapter:
    """Minimal title transport that records which profile adapter was used."""

    def __init__(self, name):
        self.name = name
        self.title_updates = []
        self.updated = asyncio.Event()

    def _context_for_send(self, chat_id, *, metadata=None):
        return {"adapter": self.name, "chat_id": chat_id}

    def get_pending_message(self, session_key):
        return None

    async def send_session_title_update(
        self,
        chat_id,
        title,
        *,
        metadata=None,
        context=None,
    ):
        self.title_updates.append({
            "chat_id": chat_id,
            "title": title,
            "metadata": metadata,
            "context": context,
        })
        self.updated.set()


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = SimpleNamespace(_db=MagicMock())
    runner._session_db._db.get_session.return_value = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._gateway_loop = None
    return runner


def _configure_runtime(monkeypatch, tmp_path):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _ImmediateAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    (tmp_path / "config.yaml").write_text(
        "agent:\n  model: test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"model": "test-model"}},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "test-model",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
        },
    )

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(
        tools_config,
        "_get_platform_tools",
        lambda user_config, platform_key: {"core"},
    )


@pytest.mark.asyncio
async def test_run_agent_wires_thechat_title_callback_into_auto_title(
    monkeypatch,
    tmp_path,
):
    """A secondary-profile turn must publish through its own TheChat adapter.

    This drives the real gateway turn, callback builder, and scheduler. It fails
    if a refactor disconnects the helper or falls back to the primary adapter.
    """
    _configure_runtime(monkeypatch, tmp_path)
    runner = _make_runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(
        return_value={"event_message_id": "message-1", "thread_id": "thread-1"}
    )

    primary_adapter = _RecordingTheChatAdapter("primary")
    secondary_adapter = _RecordingTheChatAdapter("secondary")
    runner.adapters = {Platform.THECHAT: primary_adapter}
    runner._profile_adapters = {
        "secondary": {Platform.THECHAT: secondary_adapter},
    }
    runner._active_profile_name = lambda: "default"

    source = SessionSource(
        platform=Platform.THECHAT,
        chat_id="conversation-1",
        chat_type="dm",
        user_id="user-1",
        thread_id="thread-1",
        message_id="message-1",
        profile="secondary",
    )

    auto_title_calls = []

    def fake_maybe_auto_title(*args, **kwargs):
        auto_title_calls.append((args, kwargs))
        callback = kwargs.get("title_callback")
        if callback is not None:
            callback("Generated TheChat Title")

    monkeypatch.setattr(title_generator, "maybe_auto_title", fake_maybe_auto_title)

    result = await runner._run_agent(
        message="Why are titles stale?",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key="agent:secondary:thechat:dm:conversation-1:thread-1",
        event_message_id="message-1",
    )
    await asyncio.wait_for(secondary_adapter.updated.wait(), timeout=1)

    assert result["final_response"] == "The title path completed."
    assert len(auto_title_calls) == 1
    assert callable(auto_title_calls[0][1]["title_callback"])
    assert primary_adapter.title_updates == []
    assert secondary_adapter.title_updates == [
        {
            "chat_id": "conversation-1",
            "title": "Generated TheChat Title",
            "metadata": {
                "event_message_id": "message-1",
                "thread_id": "thread-1",
            },
            "context": {
                "adapter": "secondary",
                "chat_id": "conversation-1",
            },
        }
    ]
