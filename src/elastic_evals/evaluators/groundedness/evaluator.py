# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Groundedness evaluators."""

from __future__ import annotations

import logging

from elastic_evals.api import InstrumentationProfile, KibanaEvaluatorsClient
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.kibana import (
    KibanaEvaluatorConfig,
    KibanaSubScore,
    kibana_evaluators,
)
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

QUALITATIVE_EVALUATOR_NAME = "Groundedness Analysis"
QUANTITATIVE_EVALUATOR_NAME = "Groundedness"


def create_quantitative_groundedness_evaluator(
    *,
    client: KibanaEvaluatorsClient,
    connector_id: str,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
    log: logging.Logger | None = None,
) -> Evaluator:
    return create_groundedness_evaluators(
        client=client,
        connector_id=connector_id,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )[1]


def create_groundedness_evaluators(
    *,
    client: KibanaEvaluatorsClient,
    connector_id: str,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
    log: logging.Logger | None = None,
) -> list[Evaluator]:
    quantitative = kibana_evaluators(
        [
            KibanaEvaluatorConfig(
                name="groundedness",
                kind="LLM",
                connector_id=connector_id,
                sub_scores=(
                    KibanaSubScore(
                        key="groundedness",
                        evaluator_name=QUANTITATIVE_EVALUATOR_NAME,
                    ),
                ),
            )
        ],
        client=client,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )[0]
    return [_analysis_evaluator(quantitative), quantitative]


def create_groundedness_analysis_evaluator(
    *,
    inference_client: KibanaInferenceClient | None = None,
    client: KibanaEvaluatorsClient | None = None,
    connector_id: str | None = None,
    log: logging.Logger,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
) -> Evaluator:
    if client is None:
        if inference_client is None:
            raise ValueError("client and connector_id are required")
        client = KibanaEvaluatorsClient(
            kibana_url=inference_client.kibana_url,
            api_key=inference_client.api_key,
            timeout=inference_client.timeout,
        )
        connector_id = inference_client.connector_id
    elif inference_client is not None:
        raise ValueError("Pass either client or inference_client, not both")
    elif not connector_id:
        raise ValueError("connector_id is required")

    return create_groundedness_evaluators(
        client=client,
        connector_id=connector_id,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )[0]


def _analysis_evaluator(groundedness: Evaluator) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        result = await groundedness.evaluate(params)
        if result.label in {"error", "unavailable"}:
            return result
        return EvaluationResult(
            score=None,
            label="groundedness-analysis",
            explanation=result.explanation or result.label,
            metadata=result.metadata,
        )

    return SimpleEvaluator(name=QUALITATIVE_EVALUATOR_NAME, kind="LLM", evaluate=evaluate)
