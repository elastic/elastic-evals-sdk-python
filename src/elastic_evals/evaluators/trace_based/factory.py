"""Trace-based evaluator factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from elasticsearch import AsyncElasticsearch
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.trace_based.utils import is_valid_trace_id
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams


class EsqlResponse(TypedDict):
    columns: list[dict[str, Any]]
    values: list[list[Any]]


@dataclass(frozen=True)
class TraceBasedEvaluatorConfig:
    name: str
    build_query: Callable[[str], str]
    extract_result: Callable[[EsqlResponse], float]


def _get_trace_id(output: Any) -> str | None:
    if not isinstance(output, dict):
        return None
    trace_id = output.get("traceId")
    if trace_id:
        return trace_id
    return output.get("trace_id")


async def _execute_esql_query(
    es_client: AsyncElasticsearch,
    query: str,
) -> EsqlResponse:
    response = await es_client.esql.query(query=query, format="json")
    payload = response.body if hasattr(response, "body") else response
    return payload  # type: ignore[return-value]


def create_trace_based_evaluator(
    *,
    trace_es_client: AsyncElasticsearch,
    log: logging.Logger | None,
    config: TraceBasedEvaluatorConfig,
) -> Evaluator:
    logger = log or logging.getLogger("elastic_evals")

    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        trace_id = _get_trace_id(params.output)

        if not trace_id:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation=f"No traceId available for {config.name} evaluation",
            )

        if not is_valid_trace_id(trace_id):
            logger.error("Invalid traceId for %s (traceId: %s)", config.name, trace_id)
            return EvaluationResult(
                score=None,
                label="error",
                explanation="Invalid traceId",
            )

        async def fetch_stats() -> float:
            query = config.build_query(trace_id)
            response = await _execute_esql_query(trace_es_client, query)

            values = response.get("values")
            if not values:
                raise ValueError("No data found for trace")

            return config.extract_result(response)

        def log_retry(retry_state) -> None:
            if retry_state.outcome is None:
                return
            error = retry_state.outcome.exception()
            logger.warning(
                "%s query failed on attempt %s; retrying... (traceId: %s) (%s)",
                config.name,
                retry_state.attempt_number,
                trace_id,
                error,
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(min=1, max=10),
                reraise=True,
                before_sleep=log_retry,
            ):
                with attempt:
                    return EvaluationResult(score=await fetch_stats())
        except Exception as error:
            logger.error("Failed to evaluate %s for trace %s: %s", config.name, trace_id, error)
            return EvaluationResult(
                label="error",
                explanation=f"Failed to retrieve {config.name}: {error}",
            )

        return EvaluationResult(
            label="error",
            explanation=f"Failed to retrieve {config.name}: unknown error",
        )

    return SimpleEvaluator(name=config.name, kind="CODE", evaluate=evaluate)
