# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NoOpTracerProvider, set_tracer_provider

from elastic_evals.tracing import propagated_headers


@pytest.fixture(autouse=True)
def _reset_tracer_provider() -> None:
    set_tracer_provider(NoOpTracerProvider())


def test_propagated_headers_empty_without_active_span() -> None:
    assert propagated_headers() == {}


def test_propagated_headers_returns_traceparent_inside_span() -> None:
    provider = TracerProvider()
    set_tracer_provider(provider)

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("test-span"):
        headers = propagated_headers()

    assert "traceparent" in headers
    parts = headers["traceparent"].split("-")
    assert len(parts) == 4
    assert parts[0] == "00"
