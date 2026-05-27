"""Small OpenTelemetry helpers for gateway integrations.

Hermes does not require OpenTelemetry at install time, but integrations can
emit spans when opentelemetry-api is present in the runtime.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - exercised implicitly when opentelemetry is installed
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover - default lightweight install
    trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on optional SDK installation
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - default lightweight install
    OTLPSpanExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]

_configured = False


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


def configure_tracing(service_name: str = "hermes-gateway") -> None:
    global _configured
    if _configured or trace is None:
        return
    if not all([OTLPSpanExporter, Resource, TracerProvider, BatchSpanProcessor]):
        return

    endpoint = _trace_endpoint()
    if not endpoint:
        return

    resource_attributes = {
        **_resource_attributes(),
        "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
        "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "hermes"),
    }
    provider = TracerProvider(resource=Resource.create(resource_attributes))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _configured = True


@contextmanager
def start_span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Any]:
    if trace is None:
        yield _NoopSpan()
        return

    configure_tracing()
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


def _trace_endpoint() -> str | None:
    explicit = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if explicit:
        return explicit
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base:
        return None
    return base if base.rstrip("/").endswith("/v1/traces") else f"{base.rstrip('/')}/v1/traces"


def _resource_attributes() -> dict[str, str]:
    attributes: dict[str, str] = {}
    raw = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if separator and key.strip() and value.strip():
            attributes[key.strip()] = value.strip()
    return attributes
