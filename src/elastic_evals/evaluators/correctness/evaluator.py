"""Correctness evaluators."""

from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.correctness.prompt import PROMPT, render_user_prompt, tool_choice
from elastic_evals.evaluators.correctness.scoring import (
    calculate_factual_score,
    calculate_procedural_fidelity_score,
    calculate_relevance_score,
)
from elastic_evals.evaluators.correctness.types import CorrectnessAnalysis
from elastic_evals.evaluators.filter import parse_selected_evaluators
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

QUALITATIVE_EVALUATOR_NAME = "Correctness Analysis"
FACTUALITY_EVALUATOR_NAME = "Factuality"
RELEVANCE_EVALUATOR_NAME = "Relevance"
SEQUENCE_ACCURACY_EVALUATOR_NAME = "Sequence Accuracy"


def _should_run_correctness_analysis() -> bool:
    selected = parse_selected_evaluators()
    if not selected:
        return True
    return any(
        evaluator in selected
        for evaluator in [
            QUALITATIVE_EVALUATOR_NAME,
            FACTUALITY_EVALUATOR_NAME,
            RELEVANCE_EVALUATOR_NAME,
            SEQUENCE_ACCURACY_EVALUATOR_NAME,
        ]
    )


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


async def _run_with_retries(log: logging.Logger, fn, kind: str) -> CorrectnessAnalysis:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    ):
        with attempt:
            return await fn()

    raise RuntimeError(f"{kind} retries exhausted")


def create_correctness_analysis_evaluator(
    *, inference_client: KibanaInferenceClient, log: logging.Logger
) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        if not _should_run_correctness_analysis():
            return EvaluationResult()

        async def run_analysis() -> CorrectnessAnalysis:
            user_query = (params.input or {}).get("question")
            messages = (params.output or {}).get("messages") or []
            latest_message = messages[-1].get("message") if messages else None
            ground_truth_response = (params.expected or {}).get("expected") if params.expected else None

            response = await inference_client.prompt(
                prompt=PROMPT,
                input_data={
                    "user_query": str(user_query or ""),
                    "agent_response": str(latest_message or ""),
                    "ground_truth_response": str(ground_truth_response or ""),
                },
                tool_choice=tool_choice(),
            )

            tool_calls = response.tool_calls or []
            if not tool_calls:
                raise ValueError("No tool call found in LLM response")
            analysis_payload = _parse_tool_arguments(tool_calls[0])
            return CorrectnessAnalysis.model_validate(analysis_payload)

        try:
            correctness_analysis = await _run_with_retries(log, run_analysis, "correctness analysis")
        except Exception as exc:
            log.error(
                "Failed to retrieve correctness analysis after retries (no valid tool call or malformed response)",
                exc_info=exc,
            )
            raise

        summary = correctness_analysis.summary
        explanation = (
            f"Factuality: {summary.factual_accuracy_summary}, "
            f"Relevance: {summary.relevance_summary}, "
            f"Sequence: {summary.sequence_accuracy_summary}"
        )

        return EvaluationResult(
            score=None,
            label="correctness-analysis",
            explanation=explanation,
            metadata=correctness_analysis.model_dump(),
        )

    return SimpleEvaluator(name="correctness", kind="LLM", evaluate=evaluate)


def create_quantitative_correctness_evaluators() -> list[Evaluator]:
    def extract_correctness_analysis(output: Any) -> CorrectnessAnalysis | None:
        analysis_data = (output or {}).get("correctnessAnalysis")
        if not analysis_data:
            return None
        return CorrectnessAnalysis.model_validate(analysis_data)

    def quantitative_evaluator(name: str, score_calculator, summary_key: str) -> Evaluator:
        async def evaluate(params: EvaluatorParams) -> EvaluationResult:
            correctness_analysis = extract_correctness_analysis(params.output)
            if not correctness_analysis:
                return EvaluationResult(
                    score=None,
                    label="unavailable",
                    explanation="No correctness analysis available",
                    metadata=params.metadata or None,
                )

            score = score_calculator(correctness_analysis)
            summary_text = getattr(correctness_analysis.summary, summary_key)
            return EvaluationResult(
                score=score,
                label=summary_text,
                explanation=summary_text,
                metadata=params.metadata or None,
            )

        return SimpleEvaluator(name=name, kind="LLM", evaluate=evaluate)

    return [
        quantitative_evaluator("Factuality", calculate_factual_score, "factual_accuracy_summary"),
        quantitative_evaluator("Relevance", calculate_relevance_score, "relevance_summary"),
        quantitative_evaluator(
            "Sequence Accuracy", calculate_procedural_fidelity_score, "sequence_accuracy_summary"
        ),
    ]
