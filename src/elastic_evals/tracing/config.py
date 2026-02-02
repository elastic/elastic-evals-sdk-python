"""Tracing configuration for elastic-evals."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.trace import NoOpTracerProvider
from pydantic import BaseModel, Field

from .spans import set_run_id


class ExporterConfig(BaseModel):
    """Configuration for a single trace exporter."""

    type: Literal["otlp", "phoenix", "console"] = "otlp"
    endpoint: str | None = None  # None = use type-specific default
    headers: dict[str, str] | None = None
    use_grpc: bool = False  # For Phoenix gRPC support

    # Phoenix-specific options
    project_name: str | None = None
    api_key: str | None = None


class TracingConfig(BaseModel):
    """Configuration for tracing with support for multiple exporters."""

    enabled: bool = True
    exporters: list[ExporterConfig] = Field(default_factory=lambda: [ExporterConfig()])
    service_name: str = "elastic-evals"
    run_id: str | None = None

    # Legacy single-exporter fields for backwards compatibility
    exporter: Literal["otlp", "console", "none"] | None = None
    endpoint: str | None = None

    @classmethod
    def from_legacy(
        cls,
        exporter: Literal["otlp", "console", "none"],
        endpoint: str = "http://localhost:4318/v1/traces",
        service_name: str = "elastic-evals",
        run_id: str | None = None,
    ) -> "TracingConfig":
        """Create TracingConfig from legacy single-exporter parameters."""
        if exporter == "none":
            return cls(enabled=False, exporters=[], service_name=service_name, run_id=run_id)

        return cls(
            enabled=True,
            exporters=[ExporterConfig(type=exporter, endpoint=endpoint)],
            service_name=service_name,
            run_id=run_id,
        )


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Normalize OTLP HTTP endpoint to include /v1/traces path."""
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def _normalize_phoenix_endpoint(endpoint: str, use_grpc: bool) -> str:
    """Normalize Phoenix endpoint based on transport type."""
    parsed = urlparse(endpoint)

    if use_grpc:
        # gRPC uses port 4317 by default
        # Convert default HTTP port (6006) to default gRPC port (4317)
        host = parsed.hostname or "localhost"
        if parsed.port is None or parsed.port == 6006:
            return f"{parsed.scheme}://{host}:4317"
        return f"{parsed.scheme}://{host}:{parsed.port}"

    # HTTP uses /v1/traces path
    base = f"{parsed.scheme}://{parsed.netloc}"
    if parsed.path in ("", "/"):
        return f"{base}/v1/traces"
    return endpoint


def _create_exporter(config: ExporterConfig) -> SpanExporter | None:
    """Create a span exporter from configuration."""
    if config.type == "phoenix":
        endpoint = config.endpoint or "http://localhost:6006"
        headers = dict(config.headers) if config.headers else {}

        # Add API key to headers if provided
        if config.api_key:
            headers["authorization"] = f"Bearer {config.api_key}"

        if config.use_grpc:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as GrpcOTLPSpanExporter,
                )

                grpc_endpoint = _normalize_phoenix_endpoint(endpoint, use_grpc=True)
                return GrpcOTLPSpanExporter(endpoint=grpc_endpoint, headers=headers or None)
            except ImportError as e:
                raise ImportError(
                    "gRPC exporter requires opentelemetry-exporter-otlp-proto-grpc. "
                    "Install with: pip install elastic-evals[phoenix]"
                ) from e

        http_endpoint = _normalize_phoenix_endpoint(endpoint, use_grpc=False)
        return OTLPSpanExporter(endpoint=http_endpoint, headers=headers or None)

    elif config.type == "otlp":
        endpoint = config.endpoint or "http://localhost:4318/v1/traces"
        headers = dict(config.headers) if config.headers else None
        return OTLPSpanExporter(
            endpoint=_normalize_otlp_http_endpoint(endpoint), headers=headers
        )

    elif config.type == "console":
        return ConsoleSpanExporter()

    return None


def init_tracing(config: TracingConfig) -> None:
    """Initialize OpenTelemetry tracing with configured exporters.

    Supports multiple exporters to send traces to multiple destinations
    simultaneously (e.g., both Elasticsearch APM and Arize Phoenix).
    """
    set_run_id(config.run_id)

    # Handle legacy single-exporter config
    if config.exporter is not None:
        if config.exporter == "none":
            trace.set_tracer_provider(NoOpTracerProvider())
            return
        # Convert legacy config to new format
        legacy_endpoint = config.endpoint or "http://localhost:4318/v1/traces"
        config = TracingConfig(
            enabled=config.enabled,
            exporters=[ExporterConfig(type=config.exporter, endpoint=legacy_endpoint)],
            service_name=config.service_name,
            run_id=config.run_id,
        )

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
