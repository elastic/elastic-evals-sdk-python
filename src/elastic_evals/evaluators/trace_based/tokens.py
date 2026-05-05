"""Token-based trace evaluators."""

from __future__ import annotations

import logging

from elasticsearch import AsyncElasticsearch

from elastic_evals.evaluators.trace_based.factory import (
    TraceBasedEvaluatorConfig,
    create_trace_based_evaluator,
)
from elastic_evals.evaluators.trace_based.queries import (
    build_cached_tokens_query,
    build_input_tokens_query,
    build_output_tokens_query,
)


def _extract_column_value(response, column_name: str) -> float:
    columns = response.get("columns", [])
    values = response.get("values", [])
    if not values:
        return 0.0
    row = values[0]
    column_index = next(
        (
            index
            for index, column in enumerate(columns)
            if column.get("name") == column_name
        ),
        None,
    )
    if column_index is None:
        return 0.0
    value = row[column_index]
    return float(value) if value is not None else 0.0


def create_input_tokens_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Input Tokens",
            build_query=build_input_tokens_query,
            extract_result=lambda response: _extract_column_value(
                response, "input_tokens"
            ),
        ),
    )


def create_output_tokens_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Output Tokens",
            build_query=build_output_tokens_query,
            extract_result=lambda response: _extract_column_value(
                response, "output_tokens"
            ),
        ),
    )


def create_cached_tokens_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Cached Tokens",
            build_query=build_cached_tokens_query,
            extract_result=lambda response: _extract_column_value(
                response, "cached_tokens"
            ),
        ),
    )
