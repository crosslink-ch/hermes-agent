"""TheChat rich command menu contracts against the platform implementation."""
import re
from hermes_cli.commands import COMMAND_REGISTRY
from hermes_cli.commands_platforms import thechat_menu_commands


class TestTheChatMenuCommands:
    """thechat_menu_commands returns rich entries for /bots/me/commands."""

    def test_returns_rich_entries_with_aliases_and_args_hints(self):
        menu, _hidden = thechat_menu_commands()
        by_name = {entry["command"]: entry for entry in menu}
        definitions = {command.name: command for command in COMMAND_REGISTRY}

        new = by_name["new"]
        assert new["description"]
        assert new["argsHint"] == definitions["new"].args_hint[:128]
        assert new["category"] == "Session"
        assert new["aliases"] == ["reset"]

        queue = by_name["queue"]
        assert queue["argsHint"] == definitions["queue"].args_hint[:128]
        assert queue["aliases"] == ["q"]

        # Argument metadata follows the canonical command registry as it evolves.
        assert by_name["help"]["argsHint"] == definitions["help"].args_hint[:128]
        assert all(by_name[name]["argsHint"] for name in ("new", "queue", "help"))

    def test_priority_commands_lead_the_menu(self):
        menu, _hidden = thechat_menu_commands()
        names = [entry["command"] for entry in menu]
        assert names[:4] == ["help", "new", "stop", "status"]

    def test_excludes_cli_only_and_platform_specific_commands(self):
        menu, _hidden = thechat_menu_commands()
        names = {entry["command"] for entry in menu}

        # CLI-only commands never reach gateway menus.
        for cli_only in ("quit", "config", "clear", "handoff", "copy", "paste"):
            assert cli_only not in names
        # Telegram-specific platform commands are excluded for TheChat.
        assert "start" not in names
        assert "topic" not in names
        # Hyphenated names stay hyphenated (no Telegram-style sanitization).
        assert "reload-mcp" in names
        assert by_alias_free(menu)

    def test_descriptions_within_thechat_limit(self):
        menu, _hidden = thechat_menu_commands()
        for entry in menu:
            assert 1 <= len(entry["description"]) <= 256
            assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", entry["command"]), (
                f"Invalid TheChat command name: {entry['command']!r}"
            )

    def test_skill_commands_fill_remaining_slots(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        fake_skills_dir = str(tmp_path / "skills")
        fake_cmds = {
            "/my-skill": {
                "name": "my-skill",
                "description": "Do skill things",
                "skill_md_path": f"{fake_skills_dir}/my-skill/SKILL.md",
                "skill_dir": f"{fake_skills_dir}/my-skill",
            },
        }
        with (
            patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
            patch("tools.skills_tool.SKILLS_DIR", tmp_path / "skills"),
        ):
            (tmp_path / "skills").mkdir(exist_ok=True)
            menu, _hidden = thechat_menu_commands()

        by_name = {entry["command"]: entry for entry in menu}
        assert by_name["my-skill"]["description"] == "Do skill things"
        assert by_name["my-skill"]["category"] == "Skills"


def by_alias_free(menu: list[dict]) -> bool:
    """No alias may duplicate a canonical command name."""
    names = {entry["command"] for entry in menu}
    return all(
        alias not in names
        for entry in menu
        for alias in entry.get("aliases", ())
    )
