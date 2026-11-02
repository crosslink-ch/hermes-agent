"""Behavior contracts split by scenario."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from tests.agent.run_agent_fixtures import _mock_assistant_msg, _mock_tool_call

pytest_plugins = ["tests.agent.run_agent_fixtures"]

class TestConcurrentToolExecution:
    """Tests for _execute_tool_calls_concurrent and dispatch logic."""

    def test_single_tool_uses_sequential_path(self, agent):
        """Single tool call should use sequential path, not concurrent."""
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()





    def test_explicit_targeted_writes_are_sequential_barriers(self, agent):
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/a.py","content":"one","execution_target":"beta"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/b.py","content":"two","execution_target":"beta"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_named_default_writes_are_sequential_barriers(self, agent):
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"relative.txt","content":"one"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"/remote/project/relative.txt","content":"two"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch(
            "agent.tool_dispatch_helpers._named_execution_targets_enabled",
            return_value=True,
        ):
            with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
                with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                    agent._execute_tool_calls(mock_msg, messages, "task-1")
                    mock_seq.assert_called_once()
                    mock_con.assert_not_called()

    def test_legacy_default_inventory_does_not_enable_named_barrier(self):
        from agent import tool_dispatch_helpers as helpers

        with patch(
            "tools.execution_targets.list_execution_targets",
            return_value=(SimpleNamespace(named=False),),
        ):
            assert helpers._named_execution_targets_enabled() is False
        with patch(
            "tools.execution_targets.list_execution_targets",
            return_value=(SimpleNamespace(named=True),),
        ):
            assert helpers._named_execution_targets_enabled() is True






    def test_concurrent_executes_all_tools(self, agent):
        """Concurrent path should execute all tools and append results in order."""
        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"beta"}', call_id="c2")
        tc3 = _mock_tool_call(name="web_search", arguments='{"q":"gamma"}', call_id="c3")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2, tc3])
        messages = []

        call_log = []

        def fake_handle(name, args, task_id, **kwargs):
            call_log.append(name)
            return json.dumps({"result": args.get("q", "")})

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 3
        # Results must be in original order
        assert messages[0]["tool_call_id"] == "c1"
        assert messages[1]["tool_call_id"] == "c2"
        assert messages[2]["tool_call_id"] == "c3"
        # All should be tool messages
        assert all(m["role"] == "tool" for m in messages)
        # Content should contain the query results
        assert "alpha" in messages[0]["content"]
        assert "beta" in messages[1]["content"]
        assert "gamma" in messages[2]["content"]

    def test_concurrent_none_args_rejected_without_crash(self, agent):
        """Concurrent executor must not crash on arguments=None. Current
        contract (_parse_tool_arguments): non-object args are rejected with
        a structured error result and the tool is not executed; the valid
        sibling still runs. One result per call, in order."""
        tc1 = _mock_tool_call(name="web_search", arguments=None, call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"ok"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        seen_args = []

        def fake_handle(name, args, task_id, **kwargs):
            seen_args.append((kwargs["tool_call_id"], args))
            return "ok"

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        # Only the valid call executed; the None-args call was rejected.
        assert seen_args == [("c2", {"q": "ok"})]
        assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]
        assert "tool was not executed" in messages[0]["content"].lower()

    def test_concurrent_preserves_order_despite_timing(self, agent):
        """Even if tools finish in different order, messages should be in original order."""
        import time as _time

        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"slow"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"fast"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            q = args.get("q", "")
            if q == "slow":
                _time.sleep(0.1)  # Slow tool
            return f"result_{q}"

        with patch("model_tools.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert messages[0]["tool_call_id"] == "c1"
        assert "result_slow" in messages[0]["content"]
        assert messages[1]["tool_call_id"] == "c2"
        assert "result_fast" in messages[1]["content"]


    def test_concurrent_submit_shutdown_error_returns_tool_errors(self, agent):
        """Submit-time interpreter shutdown should not escape the outer loop."""

        class ShutdownExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, *args, **kwargs):
                raise RuntimeError("cannot schedule new futures after interpreter shutdown")

            def shutdown(self, *args, **kwargs):
                pass

        tc1 = _mock_tool_call(name="web_search", arguments='{"q": "alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q": "beta"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("tools.daemon_pool.DaemonThreadPoolExecutor", ShutdownExecutor):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        assert messages[0]["tool_call_id"] == "c1"
        assert messages[1]["tool_call_id"] == "c2"
        assert all("Python interpreter is shutting down" in m["content"] for m in messages)


    def test_sequential_tool_callbacks_fire_in_order(self, agent):
        tool_call = _mock_tool_call(name="web_search", arguments='{"query":"hello"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("model_tools.handle_function_call", return_value='{"success": true}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts == [("c1", "web_search", {"query": "hello"})]
        assert completes == [("c1", "web_search", {"query": "hello"}, '{"success": true}')]


    def test_sequential_browser_type_callbacks_redact_api_key(self, agent):
        secret = "sk-proj-ABCD1234567890EFGH"
        tool_call = _mock_tool_call(
            name="browser_type",
            arguments=json.dumps({"ref": "@apikey", "text": secret}),
            call_id="c-secret",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        progress = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))
        agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress.append((event, name, preview, args))

        with patch("model_tools.handle_function_call", return_value='{"success": true, "typed": "sk-pro...EFGH"}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts[0][2]["text"].startswith("sk-pro")
        assert completes[0][2]["text"].startswith("sk-pro")
        assert progress[0][2].startswith("sk-pro")
        assert secret not in repr(starts + completes + progress)


    def test_sequential_blocked_tool_skips_checkpoints_and_callbacks(self, agent, monkeypatch):
        """Sequential path: blocked tool should not trigger checkpoints or start callbacks."""
        tool_call = _mock_tool_call(name="write_file",
                                    arguments='{"path":"test.txt","content":"hello"}',
                                    call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []

        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: ("Blocked by policy", None),
        )
        agent._checkpoint_mgr.enabled = True
        agent._checkpoint_mgr.ensure_checkpoint = MagicMock(
            side_effect=AssertionError("checkpoint should not run")
        )

        starts = []
        agent.tool_start_callback = lambda *a: starts.append(a)

        with patch("model_tools.handle_function_call", side_effect=AssertionError("should not run")):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        agent._checkpoint_mgr.ensure_checkpoint.assert_not_called()
        assert starts == []
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert json.loads(messages[0]["content"]) == {"error": "Blocked by policy"}


    @pytest.mark.parametrize("concurrent", [False, True])
    def test_tool_execution_middleware_replacement_emits_one_terminal_hook(
        self,
        agent,
        monkeypatch,
        concurrent,
    ):
        """A middleware replacement owns the result but not lifecycle closure."""
        tool_call = _mock_tool_call(
            name="terminal",
            arguments='{"command":"must-not-run"}',
            call_id="terminal-1",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        hook_calls = []

        def execution_middleware(**kwargs):
            return '{"intercepted":true}'

        manager = SimpleNamespace(_middleware={
            "tool_request": [],
            "tool_execution": [execution_middleware],
        })
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: hook_calls.append((hook_name, kwargs)) or [],
        )
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)

        with patch(
            "model_tools.handle_function_call",
            side_effect=AssertionError("middleware replacement must not dispatch"),
        ):
            if concurrent:
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")
            else:
                agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        post_calls = [
            payload for name, payload in hook_calls if name == "post_tool_call"
        ]
        assert len(post_calls) == 1
        assert post_calls[0]["tool_name"] == "terminal"
        assert post_calls[0]["tool_call_id"] == "terminal-1"
        assert post_calls[0]["status"] == "ok"
        assert post_calls[0]["result"] == '{"intercepted":true}'

    def test_blocked_memory_tool_does_not_reset_counter(self, agent, monkeypatch):
        """Blocked memory tool should not reset the nudge counter."""
        agent._turns_since_memory = 5
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *args, **kwargs: ("Blocked", None),
        )
        with patch("tools.memory_tool.memory_tool", side_effect=AssertionError("should not run")):
            result = agent._invoke_tool(
                "memory", {"action": "add", "target": "memory", "content": "x"}, "task-1",
            )

        assert json.loads(result) == {"error": "Blocked"}
        assert agent._turns_since_memory == 5


    def test_managed_tool_pipeline_rejects_second_dispatch(self, agent, monkeypatch):
        from agent import relay_tools, tool_executor

        dispatched = []
        duplicate_errors = []
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: SimpleNamespace(
                payload=args,
                trace=[],
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            lambda _name, args, callback, **_kwargs: callback(args),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *_args, **_kwargs: (None, None),
        )
        monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)

        def invoke_twice(name, args, callback, **kwargs):
            del name, kwargs
            result = callback(args)
            try:
                callback(args)
            except RuntimeError as exc:
                duplicate_errors.append(str(exc))
            return result, args

        monkeypatch.setattr(relay_tools, "execute", invoke_twice)

        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task-1",
            tool_call_id="call-1",
            execute=lambda args: dispatched.append(args) or "ok",
        )

        assert outcome.result == "ok"
        assert dispatched == [{"command": "true"}]
        assert len(duplicate_errors) == 1
        assert outcome.blocked is False

    def test_managed_tool_pipeline_allows_one_concurrent_dispatch(
        self,
        agent,
        monkeypatch,
    ):
        from agent import relay_tools, tool_executor

        dispatched = []
        results = []
        errors = []
        barrier = threading.Barrier(2)
        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            lambda _name, args, **_kwargs: SimpleNamespace(
                payload=args,
                trace=[],
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            lambda _name, args, callback, **_kwargs: callback(args),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            lambda *_args, **_kwargs: (None, None),
        )
        monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)

        def invoke_concurrently(name, args, callback, **kwargs):
            del name, kwargs

            def invoke():
                barrier.wait(timeout=2)
                try:
                    results.append(callback(args))
                except RuntimeError as exc:
                    errors.append(str(exc))

            threads = [threading.Thread(target=invoke) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
            return results[0], args

        monkeypatch.setattr(relay_tools, "execute", invoke_concurrently)

        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="terminal",
            function_args={"command": "true"},
            effective_task_id="task-1",
            tool_call_id="call-1",
            execute=lambda args: dispatched.append(args) or "ok",
        )

        assert outcome.result == "ok"
        assert dispatched == [{"command": "true"}]
        assert len(errors) == 1
        assert outcome.blocked is False

