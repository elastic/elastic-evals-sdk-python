# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Adapters for evaluators registered in Kibana."""

from __future__ import annotations

import asyncio
import logging
import weakref
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from elastic_evals.api.constants import MAX_EVALUATORS_PER_REQUEST
from elastic_evals.api.evaluators_client import KibanaEvaluatorsClient
from elastic_evals.api.evaluators_models import (
    EvaluateEvaluatorConfig,
    EvaluateRequest,
    EvaluateResponse,
    EvaluateResult,
    EvaluationInstrumentation,
    EvaluationSubject,
    EvaluationTrace,
    InstrumentationProfile,
)
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams


@dataclass(frozen=True)
class KibanaSubScore:
    key: str
    evaluator_name: str


@dataclass(frozen=True)
class KibanaEvaluatorConfig:
    name: str
    kind: Literal["LLM", "CODE"]
    version: str | None = None
    connector_id: str | None = None
    sub_scores: Sequence[KibanaSubScore] = ()


@dataclass(frozen=True)
class _ScoreSelector:
    config: KibanaEvaluatorConfig
    key: str | None
    evaluator_name: str


@dataclass(frozen=True)
class _CachedEvaluation:
    request_key: str
    task: asyncio.Task[EvaluateResponse]


class _EvaluationBatch:
    def __init__(
        self,
        *,
        client: KibanaEvaluatorsClient,
        configs: Sequence[KibanaEvaluatorConfig],
        instrumentation_profile: InstrumentationProfile,
    ) -> None:
        self._client = client
        self._configs = tuple(configs)
        self._instrumentation_profile = instrumentation_profile
        self._evaluations: weakref.WeakKeyDictionary[object, _CachedEvaluation] = weakref.WeakKeyDictionary()
        self._lock = asyncio.Lock()

    async def evaluate(self, params: EvaluatorParams) -> EvaluateResponse:
        trace_id = _trace_id(params)
        if not trace_id:
            raise ValueError("A trace ID is required for Kibana evaluators")

        request = self._build_request(params, trace_id)
        request_key = request.model_dump_json(exclude_none=True)
        scope = params._evaluation_scope

        async with self._lock:
            cached = self._evaluations.get(scope)
            if cached is None or cached.request_key != request_key:
                cached = _CachedEvaluation(
                    request_key=request_key,
                    task=asyncio.create_task(self._client.evaluate(request)),
                )
                self._evaluations[scope] = cached

        try:
            return await asyncio.shield(cached.task)
        except Exception:
            async with self._lock:
                current = self._evaluations.get(scope)
                if current is cached:
                    del self._evaluations[scope]
            raise

    def _build_request(self, params: EvaluatorParams, trace_id: str) -> EvaluateRequest:
        reference_data: dict[str, Any] | None
        if params.expected is None:
            reference_data = None
        elif isinstance(params.expected, dict):
            reference_data = params.expected
        else:
            reference_data = {"expected": params.expected}

        return EvaluateRequest(
            subject=EvaluationSubject(
                traces=[
                    EvaluationTrace(
                        trace_id=trace_id,
                        reference_data=reference_data,
                    )
                ],
                instrumentation=EvaluationInstrumentation(
                    profile=self._instrumentation_profile,
                ),
            ),
            evaluators=[
                EvaluateEvaluatorConfig(
                    name=config.name,
                    version=config.version,
                    connector_id=config.connector_id,
                )
                for config in self._configs
            ],
        )


def _trace_id(params: EvaluatorParams) -> str | None:
    if params.trace_id:
        return params.trace_id
    if not isinstance(params.output, dict):
        return None
    trace_id = (
        params.output.get("_interaction_trace_id") or params.output.get("traceId") or params.output.get("trace_id")
    )
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _error_result(message: str, *, result: EvaluateResult | None = None) -> EvaluationResult:
    metadata: dict[str, Any] = {}
    if result is not None:
        metadata["evaluator"] = result.evaluator.model_dump()
        if result.error and result.error.code:
            metadata["error_code"] = result.error.code
    return EvaluationResult(
        score=None,
        label="error",
        explanation=message,
        metadata=metadata or None,
    )


def _find_result(response: EvaluateResponse, config: KibanaEvaluatorConfig) -> EvaluateResult | None:
    for result in response.results:
        if result.evaluator.name != config.name:
            continue
        if config.version is not None and result.evaluator.version != config.version:
            continue
        return result
    return None


def _build_selectors(configs: Sequence[KibanaEvaluatorConfig]) -> list[_ScoreSelector]:
    selectors: list[_ScoreSelector] = []
    evaluator_names: set[str] = set()
    for config in configs:
        if config.sub_scores:
            outputs: Sequence[tuple[str | None, str]] = tuple(
                (sub_score.key, sub_score.evaluator_name) for sub_score in config.sub_scores
            )
        else:
            outputs = ((None, config.name),)
        for key, evaluator_name in outputs:
            if evaluator_name in evaluator_names:
                raise ValueError(f'Duplicate evaluator name "{evaluator_name}"')
            evaluator_names.add(evaluator_name)
            selectors.append(
                _ScoreSelector(
                    config=config,
                    key=key,
                    evaluator_name=evaluator_name,
                )
            )
    return selectors


def _validate_configs(configs: Sequence[KibanaEvaluatorConfig]) -> None:
    if not configs:
        raise ValueError("At least one Kibana evaluator is required")
    if len(configs) > MAX_EVALUATORS_PER_REQUEST:
        raise ValueError(f"Kibana supports at most {MAX_EVALUATORS_PER_REQUEST} evaluators per request")

    names: set[str] = set()
    for config in configs:
        if config.name in names:
            raise ValueError(f'Duplicate Kibana evaluator "{config.name}"')
        if config.kind == "LLM" and not config.connector_id:
            raise ValueError(f'connector_id is required for LLM evaluator "{config.name}"')
        names.add(config.name)


def kibana_evaluators(
    configs: Sequence[KibanaEvaluatorConfig],
    *,
    client: KibanaEvaluatorsClient,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
    log: logging.Logger | None = None,
) -> list[Evaluator]:
    """Create Python evaluators backed by one Kibana request per evaluation."""
    _validate_configs(configs)
    immutable_configs = tuple(configs)
    selectors = _build_selectors(immutable_configs)
    batch = _EvaluationBatch(
        client=client,
        configs=immutable_configs,
        instrumentation_profile=instrumentation_profile,
    )
    logger = log or logging.getLogger(__name__)

    def create_evaluator(selector: _ScoreSelector) -> Evaluator:
        async def evaluate(params: EvaluatorParams) -> EvaluationResult:
            if not _trace_id(params):
                return EvaluationResult(
                    score=None,
                    label="unavailable",
                    explanation=f"No trace ID available for {selector.evaluator_name} evaluation",
                )

            response = await batch.evaluate(params)
            result = _find_result(response, selector.config)
            if result is None:
                message = f'No evaluation result returned for "{selector.config.name}"'
                logger.error(message)
                return _error_result(message)

            if result.evaluator.kind.upper() != selector.config.kind:
                message = f'Unexpected evaluator kind returned for "{selector.config.name}"'
                logger.error(message)
                return _error_result(message, result=result)

            if result.status == "error":
                message = (
                    result.error.message
                    if result.error
                    else f'Evaluator "{selector.config.name}" failed without an error message'
                )
                logger.error(message)
                return _error_result(message, result=result)

            scores = result.scores or []
            score = (
                next((candidate for candidate in scores if candidate.name == selector.key), None)
                if selector.key is not None
                else next(iter(scores), None)
            )
            if score is None:
                description = f'score "{selector.key}"' if selector.key is not None else "score"
                message = f'No {description} returned for "{selector.config.name}"'
                logger.error(message)
                return _error_result(message, result=result)

            return EvaluationResult(
                score=score.score,
                label=score.label,
                explanation=score.explanation,
                metadata=score.metadata,
            )

        return SimpleEvaluator(
            name=selector.evaluator_name,
            kind=selector.config.kind,
            evaluate=evaluate,
        )

    return [create_evaluator(selector) for selector in selectors]
