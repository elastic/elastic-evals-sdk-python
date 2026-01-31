"""Latency trace evaluators."""

from __future__ import annotations

import logging

from elasticsearch import AsyncElasticsearch

from elastic_evals.evaluators.trace_based.factory import (
    TraceBasedEvaluatorConfig,
    create_trace_based_evaluator,
)
from elastic_evals.evaluators.trace_based.queries import (
    build_latency_query,
    build_span_latency_query,
)


def create_latency_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Latency",
            build_query=build_latency_query,
            extract_result=lambda response: float(response["values"][0][0]),
        ),
    )


def create_span_latency_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    span_name: str,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Latency",
            build_query=lambda trace_id: build_span_latency_query(trace_id, span_name),
            extract_result=lambda response: float(response["values"][0][0]),
        ),
    )
