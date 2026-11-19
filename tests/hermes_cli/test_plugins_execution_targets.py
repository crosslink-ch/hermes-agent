"""Behavior contracts split by scenario."""
import types
import pytest
from hermes_cli.plugins import (
    has_middleware,
)
from hermes_cli.middleware import (
    apply_llm_request_middleware,
    apply_tool_request_middleware,
    run_tool_execution_middleware,
)

class TestPluginDiscovery:
    @pytest.mark.parametrize(
        ("original_args", "replacement_args"),
        [
            (
                {"command": "true", "execution_target": "alpha"},
                {"command": "true", "execution_target": "beta"},
            ),
            (
                {"command": "true", "execution_target": "alpha"},
                {"command": "true"},
            ),
            (
                {"command": "true"},
                {"command": "true", "execution_target": "alpha"},
            ),
        ],
    )
    def test_tool_execution_middleware_cannot_replace_remove_or_inject_target(
        self, monkeypatch, original_args, replacement_args,
    ):
        from tools.execution_targets import ExecutionTargetError

        def execution_middleware(**kwargs):
            return kwargs["next_call"](replacement_args)

        manager = types.SimpleNamespace(
            _middleware={"tool_execution": [execution_middleware]},
        )
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        dispatched = []

        with pytest.raises(ExecutionTargetError):
            run_tool_execution_middleware(
                "terminal",
                original_args,
                lambda payload: dispatched.append(payload),
            )

        assert dispatched == []


    def test_tool_execution_middleware_rejects_polymorphic_routing_payloads(
        self, monkeypatch,
    ):
        from tools.execution_targets import ExecutionTargetError

        class EqualString(str):
            __hash__ = str.__hash__

            def __eq__(self, _other):
                return True

        class SplitDict(dict):
            def __getitem__(self, key):
                if key == "execution_target":
                    return "alpha"
                return super().__getitem__(key)

        replacements = [
            {"command": "true", "execution_target": EqualString("beta")},
            SplitDict(command="true", execution_target="beta"),
        ]
        dispatched = []

        for replacement in replacements:
            def execution_middleware(**kwargs):
                return kwargs["next_call"](replacement)

            manager = types.SimpleNamespace(
                _middleware={"tool_execution": [execution_middleware]},
            )
            monkeypatch.setattr(
                "hermes_cli.plugins.get_plugin_manager", lambda: manager,
            )

            with pytest.raises(ExecutionTargetError):
                run_tool_execution_middleware(
                    "terminal",
                    {"command": "true", "execution_target": "alpha"},
                    lambda payload: dispatched.append(payload),
                )

        assert dispatched == []


    def test_tool_execution_middleware_validates_before_copying_target_args(
        self, monkeypatch,
    ):
        from tools.execution_targets import ExecutionTargetError

        class DeepcopyTrap(dict):
            def __deepcopy__(self, _memo):
                raise AssertionError("target-aware payload must be validated first")

        manager = types.SimpleNamespace(
            _middleware={"tool_execution": [lambda **kwargs: kwargs["next_call"]()]},
        )
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

        with pytest.raises(ExecutionTargetError, match="plain JSON object"):
            run_tool_execution_middleware(
                "terminal",
                DeepcopyTrap(command="true", execution_target="alpha"),
                lambda payload: payload,
            )


    def test_tool_execution_middleware_cannot_mutate_target_in_place(
        self, monkeypatch,
    ):
        from tools.execution_targets import ExecutionTargetError

        def execution_middleware(**kwargs):
            kwargs["args"]["execution_target"] = "beta"
            return kwargs["next_call"]()

        manager = types.SimpleNamespace(
            _middleware={"tool_execution": [execution_middleware]},
        )
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        dispatched = []
        args = {"command": "true", "execution_target": "alpha"}

        with pytest.raises(ExecutionTargetError):
            run_tool_execution_middleware(
                "terminal",
                args,
                lambda payload: dispatched.append(payload),
            )

        assert dispatched == []


    def test_tool_execution_middleware_can_rewrite_non_routing_args(
        self, monkeypatch,
    ):
        def execution_middleware(**kwargs):
            return kwargs["next_call"]({
                **kwargs["args"],
                "command": "printf ok",
            })

        manager = types.SimpleNamespace(
            _middleware={"tool_execution": [execution_middleware]},
        )
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

        result = run_tool_execution_middleware(
            "terminal",
            {"command": "true", "execution_target": "alpha"},
            lambda payload: payload,
        )

        assert result == {
            "command": "printf ok",
            "execution_target": "alpha",
        }



    def test_middleware_helpers_skip_no_listener_work(self, monkeypatch):
        manager = types.SimpleNamespace(_middleware={})
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

        request = {"messages": []}
        args = {"path": "README.md"}

        llm_result = apply_llm_request_middleware(request)
        tool_result = apply_tool_request_middleware("read_file", args)

        assert llm_result.payload is request
        assert llm_result.original_payload is request
        assert llm_result.changed is False
        assert llm_result.trace == []
        assert tool_result.payload is args
        assert tool_result.original_payload is args
        assert tool_result.changed is False
        assert tool_result.trace == []
        assert run_tool_execution_middleware("terminal", args, lambda payload: payload) is args
        assert has_middleware("tool_request") is False
