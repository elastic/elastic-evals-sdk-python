# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def reset_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fresh process-global tracer provider for one test, with no network: the collector probe is
    stubbed to succeed and exported spans are kept in memory.

    OpenTelemetry allows the global provider to be set once per process and offers no public
    reset, so this touches the same private state the library's own tests use.
    """
    monkeypatch.setattr("elastic_evals.tracing.config.socket.create_connection", lambda *_, **__: socket.socket())
    monkeypatch.setattr("elastic_evals.tracing.config.OTLPSpanExporter", lambda **_: InMemorySpanExporter())
    saved_provider = trace._TRACER_PROVIDER
    saved_done = trace._TRACER_PROVIDER_SET_ONCE._done
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    try:
        yield
    finally:
        trace._TRACER_PROVIDER = saved_provider
        trace._TRACER_PROVIDER_SET_ONCE._done = saved_done
