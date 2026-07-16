# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Criteria evaluator implementation."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Iterable

from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.criteria.prompt import (
    build_prompt,
    render_system_prompt,
    render_user_prompt,
    tool_choice,
)
from elastic_evals.evaluators.criteria.short_id_table import ShortIdTable
from elastic_evals.evaluators.criteria.types import (
    EvaluationCriterion,
    EvaluationCriterionStructured,
)
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams


def _normalize_scores(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(value)


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


def _ensure_scores(
    evaluated: Iterable[dict[str, Any]],
    criteria_by_id: dict[str, EvaluationCriterionStructured],
) -> list[tuple[EvaluationCriterionStructured, dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    for evaluation in evaluated:
        unique.setdefault(evaluation["id"], evaluation)

    evaluated_ids = set(unique.keys())
    criteria_ids = set(criteria_by_id.keys())
    missing = criteria_ids - evaluated_ids
    if missing:
        raise ValueError(f"Missing scores for {', '.join(sorted(missing))}")

    results: list[tuple[EvaluationCriterionStructured, dict[str, Any]]] = []
    for criterion_id, evaluation in unique.items():
        criterion = criteria_by_id.get(criterion_id)
        if not criterion:
            raise ValueError(f'Could not find criterion for id "{criterion_id}"')
        results.append((criterion, evaluation))
    return results


def create_criteria_evaluator(
    *,
    inference_client: KibanaInferenceClient,
    criteria: list[EvaluationCriterion] | None = None,
    log: logging.Logger,
) -> Evaluator:
    table = ShortIdTable()

    structured_criteria: list[EvaluationCriterionStructured] = []
    for criterion in criteria or []:
        if isinstance(criterion, str):
            structured_criteria.append(EvaluationCriterionStructured(id=table.take(criterion), text=criterion, score=1))
        else:
            structured_criteria.append(
                EvaluationCriterionStructured(id=criterion.id, text=criterion.text, score=criterion.score or 1)
            )

    criteria_by_id = {criterion.id: criterion for criterion in structured_criteria}

    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        async def score_task() -> list[tuple[EvaluationCriterionStructured, dict[str, Any]]]:
            system_prompt = render_system_prompt(
                [f"{criterion.id}: {criterion.text}" for criterion in structured_criteria]
            )
            user_prompt = render_user_prompt(
                input_text=json.dumps(params.input),
                output_text=json.dumps(params.output),
                metadata_text=json.dumps(params.metadata),
            )
            response = await inference_client.prompt(
                prompt=build_prompt(system_prompt, user_prompt),
                input_data={},
                tool_choice=tool_choice(),
            )

            tool_calls = response.tool_calls or []
            if not tool_calls:
                raise ValueError("No tool call found in LLM response")

            tool_scores: list[tuple[EvaluationCriterionStructured, dict[str, Any]]] = []
            for tool_call in tool_calls:
                payload = _parse_tool_arguments(tool_call)
                criteria_scores = payload.get("criteria", [])
                tool_scores.extend(_ensure_scores(criteria_scores, criteria_by_id))

            return tool_scores

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(min=1, max=10),
                reraise=True,
            ):
                with attempt:
                    scores = await score_task()
                    break
        except Exception as exc:
            log.error("Failed to score criteria after retries", exc_info=exc)
            raise

        successful = [item for item in scores if item[1]["result"] == "PASS"]
        failed = [item for item in scores if item[1]["result"] == "FAIL"]
        not_applicable = [item for item in scores if item[1]["result"] == "N/A"]

        max_score = sum(criterion.score or 0 for criterion in structured_criteria)
        total_score = sum((criterion.score or 0) for criterion, _ in successful + not_applicable)

        explanation = "\n".join(
            f'"{criterion.id}": {evaluation.get("reason") or "No explanation given"}'
            for criterion, evaluation in scores
        )

        return EvaluationResult(
            score=_normalize_scores(total_score / max_score if max_score else 0),
            label=None,
            explanation=explanation,
            metadata={
                "successful": sum((criterion.score or 0) for criterion, _ in successful),
                "failed": sum((criterion.score or 0) for criterion, _ in failed),
                "not_applicable": sum((criterion.score or 0) for criterion, _ in not_applicable),
            },
        )

    return SimpleEvaluator(name="criteria", kind="LLM", evaluate=evaluate)
