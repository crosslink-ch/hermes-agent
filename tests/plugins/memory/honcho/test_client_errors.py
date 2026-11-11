"""A Honcho failure is logged by message; only an error outside the known families gets a traceback."""
import logging

import httpx

from plugins.memory.honcho.session import HonchoSessionManager


def _guarded_record(caplog, exc):
    def fail():
        raise exc

    manager = HonchoSessionManager(honcho=object())
    with caplog.at_level(logging.DEBUG, logger="plugins.memory.honcho.session"):
        assert manager._guarded(fail, "fallback", logging.WARNING, "fetch failed: %s") == "fallback"
    (record,) = caplog.records
    return record


def test_transport_error_is_logged_without_a_traceback(caplog):
    record = _guarded_record(caplog, httpx.ConnectError("connection refused"))
    assert not record.exc_info
    assert "connection refused" in record.getMessage()


def test_unexpected_error_is_logged_with_its_traceback(caplog):
    record = _guarded_record(caplog, KeyError("peer_card"))
    assert record.exc_info and record.exc_info[0] is KeyError
