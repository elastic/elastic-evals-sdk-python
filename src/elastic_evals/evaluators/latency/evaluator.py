# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Latency evaluator derived from indexed trace duration."""

from __future__ import annotations

import logging
import math
import re

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.tracing import ElasticsearchTraceClient, EsqlResponse
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

EVALUATOR_NAME = "latency"
TRACE_RETRY_ATTEMPTS = 6
TRACE_RETRY_WAIT = wait_exponential(multiplier=2, min=2, max=60)


def _build_query(trace_id: str) -> str:
    return f"""FROM traces-*
| WHERE trace.id == "{trace_id}"
| STATS total_duration_ns = MAX(duration)
| EVAL latency_seconds = TO_DOUBLE(total_duration_ns) / 1000000000
| KEEP latency_seconds"""


def _extract_result(response: EsqlResponse) -> int | float | None:
    return response.values[0][0]


def _is_valid_trace_id(trace_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", trace_id)) and trace_id != "0" * 32


def create_latency_evaluator(*, trace_client: ElasticsearchTraceClient, log: logging.Logger) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        trace_id = params.trace_id
        if not trace_id:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation=f"No trace ID available for {EVALUATOR_NAME} evaluation",
            )
        if not _is_valid_trace_id(trace_id):
            log.error("Invalid trace ID for %s: %s", EVALUATOR_NAME, trace_id)
            return EvaluationResult(score=None, label="error", explanation="Invalid trace ID")

        last_result: int | float | None = None
        has_result = False

        async def fetch_latency() -> float:
            nonlocal has_result, last_result
            response = await trace_client.query(_build_query(trace_id))
            if not response.values:
                raise ValueError("No data found for trace")

            result = _extract_result(response)
            has_result = True
            last_result = result
            if result is None or isinstance(result, float) and not math.isfinite(result):
                raise ValueError(f"{EVALUATOR_NAME} result looks incomplete: {result}")
            return float(result)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(TRACE_RETRY_ATTEMPTS),
                wait=TRACE_RETRY_WAIT,
                reraise=True,
            ):
                with attempt:
                    score = await fetch_latency()
                    return EvaluationResult(score=score)
        except Exception as exc:
            if has_result:
                log.warning(
                    "%s may be incomplete for trace %s: %r",
                    EVALUATOR_NAME,
                    trace_id,
                    last_result,
                )
                return EvaluationResult(
                    score=last_result,
                    label="potentially_incomplete",
                    explanation=f"{EVALUATOR_NAME} may be based on incomplete trace data",
                    metadata={"incomplete": True},
                )

            log.error("Failed to retrieve %s for trace %s", EVALUATOR_NAME, trace_id, exc_info=exc)
            return EvaluationResult(
                score=None,
                label="error",
                explanation=f"Failed to retrieve {EVALUATOR_NAME}: {exc}",
            )

        raise RuntimeError(f"{EVALUATOR_NAME} retries exhausted")

    return SimpleEvaluator(name=EVALUATOR_NAME, kind="CODE", evaluate=evaluate)
