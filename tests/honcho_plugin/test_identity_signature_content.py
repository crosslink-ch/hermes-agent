"""Preserve fork identity invalidation through the generic memory-provider hook."""
import json
import os


def test_same_size_identity_edit_with_preserved_mtime_invalidates_cache(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.run_agent_cache import GatewayAgentCacheMixin

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayAgentCacheMixin, "_MEMORY_IDENTITY_PROVIDER_MEMO", {})
    path = tmp_path / "honcho.json"
    config = {"apiKey": "synthetic-test-key", "peerName": "alice", "pinPeerName": True}
    path.write_text(json.dumps(config), encoding="utf-8")
    stat = path.stat()
    before = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    assert before["memory.user_identity"] == "alice"

    config["peerName"] = "bruno"
    path.write_text(json.dumps(config), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    after = GatewayRunner._extract_cache_busting_config({"memory": {"provider": "honcho"}})
    assert after["memory.user_identity"] == "bruno"
    assert before["memory.user_identity"] == "alice"
