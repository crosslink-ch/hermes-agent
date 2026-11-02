"""Lazy plugin CLI registration without eagerly loading plugins for built-ins."""
from __future__ import annotations

def _attach_plugin_cli_command(subparsers, cmd_info) -> None:
    """Register one plugin-provided top-level command from its descriptor."""
    plugin_parser = subparsers.add_parser(
        cmd_info["name"],
        help=cmd_info["help"],
        description=cmd_info.get("description", ""),
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
    )
    cmd_info["setup_fn"](plugin_parser)
    if cmd_info.get("handler_fn") is not None:
        plugin_parser.set_defaults(func=cmd_info["handler_fn"])


def _register_plugin_cli_commands(subparsers) -> None:
    """Register plugin-provided top-level commands (each plugin builds its own argparse tree).

    Skipped when the invocation targets a known built-in — eagerly importing
    every bundled plugin module costs 500-650ms.
    """
    from hermes_cli.main import _attach_plugin_cli_command, _first_positional_argv, _plugin_cli_discovery_needed, _resolve_deferred_platform_cli_command, logging
    if not _plugin_cli_discovery_needed():
        return
    try:
        from plugins.memory import discover_plugin_cli_commands
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        from hermes_cli.left_core_migration import recover_for_command

        recover_for_command(_first_positional_argv())
        seen_plugin_commands = set()
        for cmd_info in discover_plugin_cli_commands():
            _attach_plugin_cli_command(subparsers, cmd_info)
            seen_plugin_commands.add(cmd_info["name"])

        discover_plugins()
        # The invoked platform may still be a deferred entry; import it so its
        # register_cli_command side effect runs before we read _cli_commands.
        # See #54678.
        _resolve_deferred_platform_cli_command(_first_positional_argv())
        for cmd_info in get_plugin_manager()._cli_commands.values():
            if cmd_info["name"] not in seen_plugin_commands:
                _attach_plugin_cli_command(subparsers, cmd_info)
    except Exception as _exc:
        logging.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc, exc_info=True)
