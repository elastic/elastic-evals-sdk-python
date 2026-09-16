# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Tracing configuration for elastic-evals."""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import ProxyTracerProvider
from pydantic import BaseModel

from .spans import set_run_id


class TracingConfig(BaseModel):
    """Configuration for tracing with a single OTLP exporter."""

    enabled: bool = True
    endpoint: str = "http://localhost:4318"
    api_key: str | None = None
    service_name: str = "elastic-evals"
    run_id: str | None = None


def _normalize_otlp_http_endpoint(endpoint: str) -> str:
    """Normalize OTLP HTTP endpoint to include /v1/traces path."""
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def _check_endpoint_reachable(endpoint: str) -> None:
    """Raise if nothing accepts connections at the OTLP endpoint."""
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((host, port), timeout=1.0).close()
    except OSError as exc:
        raise ConnectionError(
            f"Tracing is enabled but no OTLP collector is reachable at {host}:{port} ({exc}). "
            "Start a collector or point ELASTIC_OTLP_ENDPOINT at one, or set "
            "ELASTIC_EVALS_TRACING_ENABLED=false to run without traces."
        ) from exc


def init_tracing(config: TracingConfig) -> None:
    """Install the OTLP exporter if tracing is enabled and no tracer provider has been installed yet.

    OpenTelemetry allows one global provider per process, so calling this again is a no-op.
    Raises `ConnectionError` if the collector cannot be reached, so a run fails before it starts.
    """
    set_run_id(config.run_id)

    if not config.enabled or not isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
        return

    endpoint = _normalize_otlp_http_endpoint(config.endpoint)
    _check_endpoint_reachable(endpoint)

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)
    headers = {"Authorization": f"ApiKey {config.api_key}"} if config.api_key else {}
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
