"""Capture the spawning agent's process lifecycle consumer, not ambient routing."""

from contextvars import ContextVar
import logging

_callback = ContextVar("process_activity_consumer", default=None)
logger = logging.getLogger(__name__)


def bind_process_activity(callback):
    _callback.set(callback)


def _emit(callback, event, **metadata):
    if callback is not None:
        try:
            callback(event, None, None, None, **metadata)
        except Exception:
            logger.debug("Process lifecycle consumer failed", exc_info=True)


def observe_process_start(registry, session):
    callback = _callback.get()
    if callback is None:
        return
    # A short-lived process may exit before routing is stamped. Serialize attach
    # with the registry move and replay that real exit exactly once if needed.
    with registry._lock:
        _emit(callback, "process.started", process_id=session.id)
        session._activity_consumer = callback
        if session.exited:
            observe_process_finish(session)


def observe_process_finish(session):
    callback = getattr(session, "_activity_consumer", None)
    session._activity_consumer = None
    reason = session.completion_reason
    status = (
        "killed"
        if reason == "killed" or getattr(session, "_activity_stop_requested", False)
        else "unknown"
        if reason in {"lost", "unknown"}
        else "completed"
        if session.exit_code == 0
        else "failed"
    )
    _emit(callback, "process.completed", process_id=session.id, status=status)
