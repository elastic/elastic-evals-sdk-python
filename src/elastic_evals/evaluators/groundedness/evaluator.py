# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Groundedness evaluators."""

from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.groundedness.prompt import PROMPT, tool_choice
from elastic_evals.evaluators.groundedness.scoring import calculate_groundedness_score
from elastic_evals.evaluators.groundedness.types import GroundednessAnalysis
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

QUALITATIVE_EVALUATOR_NAME = "Groundedness Analysis"
QUANTITATIVE_EVALUATOR_NAME = "Groundedness"


def _parse_tool_arguments(tool_call: Any) -> dict[str, Any]:
    if not tool_call or not tool_call.function:
        raise ValueError("No tool call found in LLM response")
    arguments = tool_call.function.get("arguments") if isinstance(tool_call.function, dict) else None
    if arguments is None:
        raise ValueError("No tool arguments found in LLM response")
    if isinstance(arguments, str):
        return json.loads(arguments)
    if isinstance(arguments, dict):
        return arguments
    raise ValueError("Invalid tool arguments in LLM response")


async def _run_with_retries(log: logging.Logger, fn, kind: str) -> GroundednessAnalysis:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    ):
        with attempt:
            return await fn()

    raise RuntimeError(f"{kind} retries exhausted")


def create_groundedness_analysis_evaluator(
    *, inference_client: KibanaInferenceClient, log: logging.Logger
) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        async def run_analysis() -> GroundednessAnalysis:
            user_query = (params.input or {}).get("question")
            messages = (params.output or {}).get("messages") or []
            latest_message = messages[-1].get("message") if messages else None
            steps = (params.output or {}).get("steps") or []

            response = await inference_client.prompt(
                prompt=PROMPT,
                input_data={
                    "user_query": str(user_query or ""),
                    "agent_response": str(latest_message or ""),
                    "tool_call_history": json.dumps(steps),
                },
                tool_choice=tool_choice(),
            )

            tool_calls = response.tool_calls or []
            if not tool_calls:
                raise ValueError("No tool call found in LLM response")
            analysis_payload = _parse_tool_arguments(tool_calls[0])
            return GroundednessAnalysis.model_validate(analysis_payload)

        try:
            groundedness_analysis = await _run_with_retries(log, run_analysis, "groundedness analysis")
        except Exception as exc:
            log.error(
                "Failed to retrieve groundedness analysis after retries (no valid tool call or malformed response)",
                exc_info=exc,
            )
            raise

        explanation = groundedness_analysis.summary_verdict
        return EvaluationResult(
            score=None,
            label="groundedness-analysis",
            explanation=explanation,
            metadata=groundedness_analysis.model_dump(),
        )

    return SimpleEvaluator(name=QUALITATIVE_EVALUATOR_NAME, kind="LLM", evaluate=evaluate)


def create_quantitative_groundedness_evaluator() -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        groundedness_analysis_data = (params.output or {}).get("groundednessAnalysis")
        if not groundedness_analysis_data:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation="No groundedness analysis available",
                metadata=params.metadata or None,
            )

        groundedness_analysis = GroundednessAnalysis.model_validate(groundedness_analysis_data)
        score = calculate_groundedness_score(groundedness_analysis)
        summary_text = groundedness_analysis.summary_verdict
        return EvaluationResult(
            score=score,
            label=summary_text,
            explanation=summary_text,
            metadata=params.metadata or None,
        )

    return SimpleEvaluator(name=QUANTITATIVE_EVALUATOR_NAME, kind="LLM", evaluate=evaluate)
