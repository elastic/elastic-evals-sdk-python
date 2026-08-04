# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Correctness evaluators."""

from __future__ import annotations

import logging
from typing import Any

from elastic_evals.api import InstrumentationProfile, KibanaEvaluatorsClient
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.kibana import (
    KibanaEvaluatorConfig,
    KibanaSubScore,
    kibana_evaluators,
)
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

QUALITATIVE_EVALUATOR_NAME = "Correctness Analysis"
FACTUALITY_EVALUATOR_NAME = "Factuality"
RELEVANCE_EVALUATOR_NAME = "Relevance"
SEQUENCE_ACCURACY_EVALUATOR_NAME = "Sequence Accuracy"

_SUB_SCORES = (
    KibanaSubScore(key="factuality", evaluator_name=FACTUALITY_EVALUATOR_NAME),
    KibanaSubScore(key="relevance", evaluator_name=RELEVANCE_EVALUATOR_NAME),
    KibanaSubScore(key="sequence_accuracy", evaluator_name=SEQUENCE_ACCURACY_EVALUATOR_NAME),
)


def _config(connector_id: str) -> KibanaEvaluatorConfig:
    return KibanaEvaluatorConfig(
        name="correctness",
        kind="LLM",
        connector_id=connector_id,
        sub_scores=_SUB_SCORES,
    )


def create_quantitative_correctness_evaluators(
    *,
    client: KibanaEvaluatorsClient,
    connector_id: str,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
    log: logging.Logger | None = None,
) -> list[Evaluator]:
    return kibana_evaluators(
        [_config(connector_id)],
        client=client,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )


def create_correctness_analysis_evaluator(
    *,
    inference_client: KibanaInferenceClient,
    log: logging.Logger,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
) -> Evaluator:
    client = KibanaEvaluatorsClient(
        kibana_url=inference_client.kibana_url,
        api_key=inference_client.api_key,
        timeout=inference_client.timeout,
    )
    factuality = kibana_evaluators(
        [
            KibanaEvaluatorConfig(
                name="correctness",
                kind="LLM",
                connector_id=inference_client.connector_id,
                sub_scores=(KibanaSubScore(key="factuality", evaluator_name="correctness"),),
            )
        ],
        client=client,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )[0]

    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        result = await factuality.evaluate(params)
        if result.label in {"error", "unavailable"}:
            return result

        summary = (result.metadata or {}).get("summary", {})
        explanation = _analysis_explanation(summary, result.explanation)
        return EvaluationResult(
            score=None,
            label="correctness-analysis",
            explanation=explanation,
            metadata=result.metadata,
        )

    return SimpleEvaluator(name=QUALITATIVE_EVALUATOR_NAME, kind="LLM", evaluate=evaluate)


def _analysis_explanation(summary: Any, fallback: str | None) -> str | None:
    if not isinstance(summary, dict):
        return fallback
    return (
        f"Factuality: {summary.get('factual_accuracy_summary')}, "
        f"Relevance: {summary.get('relevance_summary')}, "
        f"Sequence: {summary.get('sequence_accuracy_summary')}"
    )
