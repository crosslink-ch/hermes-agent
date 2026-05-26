"""Small OpenTelemetry helpers for gateway integrations.

Hermes does not require OpenTelemetry at install time, but integrations can
emit spans when opentelemetry-api is present in the runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - exercised implicitly when opentelemetry is installed
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - default lightweight install
    trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]


class _NoopSpan:
    def set_attribute(self, _key: str, _value: Any) -> None:
        return None

    def record_exception(self, _exception: BaseException) -> None:
        return None

    def set_status(self, _status: Any) -> None:
        return None


def _set_attributes(span: Any, attributes: Mapping[str, Any] | None) -> None:
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        span.set_attribute(key, value)


@contextmanager
def start_span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    if trace is None:
        yield _NoopSpan()
        return

    tracer = trace.get_tracer("hermes-gateway")
    with tracer.start_as_current_span(name) as span:
        _set_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            if Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
