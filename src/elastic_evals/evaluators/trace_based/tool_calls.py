"""Tool call trace evaluators."""

from __future__ import annotations

import logging

from elasticsearch import AsyncElasticsearch

from elastic_evals.evaluators.trace_based.factory import (
    TraceBasedEvaluatorConfig,
    create_trace_based_evaluator,
)
from elastic_evals.evaluators.trace_based.queries import build_tool_calls_query


def create_tool_calls_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None = None,
):
    return create_trace_based_evaluator(
        trace_es_client=trace_es_client,
        log=log,
        config=TraceBasedEvaluatorConfig(
            name="Tool Calls",
            build_query=build_tool_calls_query,
            extract_result=lambda response: float(response["values"][0][0]),
        ),
    )
