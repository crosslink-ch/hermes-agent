"""Memory-file reads preserve Unicode content without uploading a UTF-8 BOM."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_owner_memory_upload_strips_bom(tmp_path, encoding):
    from plugins.memory.honcho.session_migration import SessionMigrationMixin

    (tmp_path / "USER.md").write_text("Owner facts: café", encoding=encoding)
    uploads = []
    session = SimpleNamespace(honcho_session_id="remote", user_peer_id="owner", assistant_peer_id="bot")
    manager = SimpleNamespace(
        _cached_session=lambda _: session, _sessions_cache={"remote": session},
        _declared_owner_peer_id=lambda: "owner", _get_or_create_peer=lambda peer: peer,
        _sdk_session=lambda _: SimpleNamespace(upload_file=lambda **kwargs: uploads.append(kwargs)),
        _authed_call=lambda _, call: call(),
    )
    assert SessionMigrationMixin.migrate_memory_files(manager, "local", str(tmp_path))
    assert len(uploads) == 1
    content = uploads[0]["file"][1].decode("utf-8")
    assert "Owner facts: café" in content and "\ufeff" not in content
    assert uploads[0]["peer"] == "owner"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("entry", ["identity", "migration"])
def test_identity_seed_strips_bom(tmp_path, monkeypatch, encoding, entry):
    from plugins.memory.honcho import cli

    path = tmp_path / "SOUL.md"
    path.write_text("Agent identity: café", encoding=encoding)
    seeds = []
    manager = SimpleNamespace(seed_ai_identity=lambda key, text, **kw: seeds.append((key, text, kw)) or True)
    if entry == "identity":
        monkeypatch.setattr(cli, "_read_config", dict)
        monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "synthetic")
        monkeypatch.setattr(cli, "_connect", lambda _: (SimpleNamespace(ai_peer="bot"), None))
        monkeypatch.setattr(cli, "_session_manager", lambda *args: (manager, "local"))
        cli.cmd_identity(SimpleNamespace(file=str(path), show=False))
    else:
        cli._migrate_seed(manager, "local", [path])
    assert seeds == [("local", "Agent identity: café", {"source": "SOUL.md"})]
