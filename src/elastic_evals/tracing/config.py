"""Tracing configuration for elastic-evals."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.trace import NoOpTracerProvider
from pydantic import BaseModel, Field

from .spans import set_run_id


class ExporterConfig(BaseModel):
    """Configuration for a single trace exporter."""

    type: Literal["otlp", "console"] = "otlp"
    endpoint: str | None = None  # None = use type-specific default
    headers: dict[str, str] | None = None


class TracingConfig(BaseModel):
    """Configuration for tracing with support for multiple exporters."""

    enabled: bool = True
    exporters: list[ExporterConfig] = Field(default_factory=lambda: [ExporterConfig()])
    service_name: str = "elastic-evals"
    run_id: str | None = None


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Normalize OTLP HTTP endpoint to include /v1/traces path."""
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def _create_exporter(config: ExporterConfig) -> SpanExporter | None:
    """Create a span exporter from configuration."""
    if config.type == "otlp":
        endpoint = config.endpoint or "http://localhost:4318/v1/traces"
        headers = dict(config.headers) if config.headers else None
        return OTLPSpanExporter(
            endpoint=_normalize_otlp_http_endpoint(endpoint), headers=headers
        )

    if config.type == "console":
        return ConsoleSpanExporter()

    return None


def init_tracing(config: TracingConfig) -> None:
    """Initialize OpenTelemetry tracing with configured exporters.

    Supports multiple exporters to send traces to multiple destinations.
    """
    set_run_id(config.run_id)

    if not config.enabled or not config.exporters:
        trace.set_tracer_provider(NoOpTracerProvider())
        return

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    # Add a BatchSpanProcessor for each configured exporter
    for exporter_config in config.exporters:
        exporter = _create_exporter(exporter_config)
        if exporter:
            provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
