"""Tracing configuration for elastic-evals."""

from __future__ import annotations

from typing import Literal

from opentelemetry import trace
from urllib.parse import urlparse

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import NoOpTracerProvider
from pydantic import BaseModel

from .spans import set_run_id


class TracingConfig(BaseModel):
    enabled: bool = True
    exporter: Literal["otlp", "console", "none"] = "otlp"
    endpoint: str = "http://localhost:4318/v1/traces"
    service_name: str = "elastic-evals"
    run_id: str | None = None


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def init_tracing(config: TracingConfig) -> None:
    set_run_id(config.run_id)

    if not config.enabled or config.exporter == "none":
        trace.set_tracer_provider(NoOpTracerProvider())
        return

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)

    if config.exporter == "otlp":
        exporter = OTLPSpanExporter(endpoint=_normalize_otlp_http_endpoint(config.endpoint))
    elif config.exporter == "console":
        exporter = ConsoleSpanExporter()
    else:
        trace.set_tracer_provider(NoOpTracerProvider())
        return

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
