# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import ProxyTracerProvider

from elastic_evals.tracing import TracingConfig, init_tracing, propagated_headers

pytestmark = pytest.mark.usefixtures("reset_tracer_provider")


def test_propagated_headers_empty_without_active_span() -> None:
    assert propagated_headers() == {}


def test_propagated_headers_returns_traceparent_inside_span() -> None:
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    with provider.get_tracer("test").start_as_current_span("test-span"):
        headers = propagated_headers()

    assert "traceparent" in headers
    assert headers["traceparent"].startswith("00-")


def test_init_tracing_disabled_installs_nothing() -> None:
    init_tracing(TracingConfig(enabled=False))

    assert isinstance(trace.get_tracer_provider(), ProxyTracerProvider)


def test_init_tracing_installs_once_and_ignores_repeat_calls(caplog: pytest.LogCaptureFixture) -> None:
    config = TracingConfig(enabled=True, endpoint="http://localhost:9")

    init_tracing(config)
    first = trace.get_tracer_provider()
    with caplog.at_level("WARNING"):
        init_tracing(config)

    assert isinstance(first, TracerProvider)
    assert trace.get_tracer_provider() is first
    assert not [r for r in caplog.records if "Overriding" in r.getMessage()]
