"""Tracing span helpers for elastic-evals."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from opentelemetry import baggage, context, propagate, trace
from opentelemetry.context import Context

TRACER_NAME = "elastic-evals"
_run_id: str | None = None


def set_run_id(run_id: str | None) -> None:
    global _run_id
    _run_id = run_id


def _root_context() -> Context:
    root = context.Context()
    if _run_id:
        return baggage.set_baggage("elastic.evals.run_id", _run_id, context=root)
    return root


def get_current_trace_id() -> str | None:
    try:
        active_span = trace.get_current_span()
        span_context = active_span.get_span_context()
    except Exception:
        return None

    if not span_context.is_valid or span_context.trace_id == 0:
        return None
    return f"{span_context.trace_id:032x}"


def propagated_headers() -> dict[str, str]:
    """Return W3C traceparent/tracestate headers for the current span context."""
    headers: dict[str, str] = {}
    propagate.inject(headers)
    return headers


async def _with_span(
    name: str,
    attributes: dict[str, Any],
    fn: Callable[[], Awaitable[Any]],
) -> tuple[Any, str | None]:
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(name, context=_root_context()) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        result = await fn()
        trace_id = get_current_trace_id()
        return result, trace_id


async def with_task_span(
    name: str,
    attributes: dict[str, Any],
    fn: Callable[[], Awaitable[Any]],
) -> tuple[Any, str | None]:
    span_attributes = {
        "instrumentationScope.name": TRACER_NAME,
        "task.name": name,
        **attributes,
    }
    return await _with_span(name, span_attributes, fn)


async def with_evaluator_span(
    name: str,
    attributes: dict[str, Any],
    fn: Callable[[], Awaitable[Any]],
) -> tuple[Any, str | None]:
    span_attributes = {
        "instrumentationScope.name": TRACER_NAME,
        "evaluator.name": name,
        **attributes,
    }
    return await _with_span(name, span_attributes, fn)
