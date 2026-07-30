# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest

from elastic_evals.api import (
    EvaluateResponse,
    EvaluateResult,
    EvaluatorError,
    EvaluatorIdentity,
    EvaluatorScore,
    KibanaEvaluatorsClient,
)
from elastic_evals.evaluators import (
    create_input_tokens_evaluator,
    create_latency_evaluator,
    create_output_tokens_evaluator,
    create_tool_calls_evaluator,
)
from elastic_evals.evaluators.correctness import (
    create_correctness_analysis_evaluator,
    create_quantitative_correctness_evaluators,
)
from elastic_evals.evaluators.groundedness import (
    create_groundedness_analysis_evaluator,
    create_quantitative_groundedness_evaluator,
)
from elastic_evals.evaluators.kibana import (
    KibanaEvaluatorConfig,
    KibanaSubScore,
    kibana_evaluators,
)
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.types import EvaluatorParams


def _configs() -> list[KibanaEvaluatorConfig]:
    return [
        KibanaEvaluatorConfig(
            name="correctness",
            kind="LLM",
            connector_id="evaluator-connector",
            sub_scores=[
                KibanaSubScore(key="factuality", evaluator_name="Factuality"),
                KibanaSubScore(key="relevance", evaluator_name="Relevance"),
                KibanaSubScore(key="sequence_accuracy", evaluator_name="Sequence Accuracy"),
            ],
        ),
        KibanaEvaluatorConfig(
            name="groundedness",
            kind="LLM",
            connector_id="evaluator-connector",
            sub_scores=[
                KibanaSubScore(key="groundedness", evaluator_name="Groundedness"),
            ],
        ),
        KibanaEvaluatorConfig(name="latency", kind="CODE"),
        KibanaEvaluatorConfig(name="input_tokens", kind="CODE"),
        KibanaEvaluatorConfig(name="output_tokens", kind="CODE"),
        KibanaEvaluatorConfig(name="tool_calls", kind="CODE"),
    ]


def _response() -> EvaluateResponse:
    analysis = {
        "summary": {
            "factual_accuracy_summary": "FULLY_SUPPORTED",
            "relevance_summary": "RELEVANT",
            "sequence_accuracy_summary": "MATCH",
        }
    }
    return EvaluateResponse(
        results=[
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="correctness", version="1.0.0", kind="llm"),
                scores=[
                    EvaluatorScore(name="factuality", score=0.9, label="FULLY_SUPPORTED", metadata=analysis),
                    EvaluatorScore(name="relevance", score=1.0, label="RELEVANT", metadata=analysis),
                    EvaluatorScore(name="sequence_accuracy", score=0.8, label="MATCH", metadata=analysis),
                ],
            ),
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="groundedness", version="1.0.0", kind="llm"),
                scores=[
                    EvaluatorScore(
                        name="groundedness",
                        score=0.95,
                        label="GROUNDED",
                        explanation="GROUNDED",
                        metadata={"summary_verdict": "GROUNDED"},
                    )
                ],
            ),
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="latency", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="latency", score=2.5)],
            ),
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="input_tokens", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="input_tokens", score=123)],
            ),
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="output_tokens", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="output_tokens", score=45)],
            ),
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="tool_calls", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="tool_calls", score=3)],
            ),
        ]
    )


def _params(trace_id: str = "1" * 32) -> EvaluatorParams:
    return EvaluatorParams(
        input={"question": "Question"},
        output={"answer": "Answer"},
        expected={"expected": "Expected answer"},
        metadata={"source": "test"},
        trace_id=trace_id,
    )


@pytest.mark.asyncio
async def test_correctness_and_groundedness_creators_use_kibana_evaluators() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = _response()
    correctness = create_quantitative_correctness_evaluators(
        client=client,
        connector_id="evaluator-connector",
    )

    results = await asyncio.gather(*(evaluator.evaluate(_params()) for evaluator in correctness))

    assert [evaluator.name for evaluator in correctness] == [
        "Factuality",
        "Relevance",
        "Sequence Accuracy",
    ]
    assert [result.score for result in results] == [0.9, 1.0, 0.8]
    request = client.evaluate.await_args.args[0]
    assert request.evaluators[0].name == "correctness"
    assert request.evaluators[0].connector_id == "evaluator-connector"
    assert client.evaluate.await_count == 1

    client.reset_mock()
    groundedness = create_quantitative_groundedness_evaluator(
        client=client,
        connector_id="evaluator-connector",
    )

    result = await groundedness.evaluate(_params())

    assert groundedness.name == "Groundedness"
    assert result.score == 0.95
    request = client.evaluate.await_args.args[0]
    assert request.evaluators[0].name == "groundedness"
    assert request.evaluators[0].connector_id == "evaluator-connector"


@pytest.mark.asyncio
async def test_analysis_creators_preserve_legacy_results_through_kibana(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate = AsyncMock(return_value=_response())
    monkeypatch.setattr(KibanaEvaluatorsClient, "evaluate", evaluate)
    inference_client = KibanaInferenceClient(
        kibana_url="http://kibana:5601",
        connector_id="evaluator-connector",
        api_key="secret-key",
    )
    correctness = create_correctness_analysis_evaluator(
        inference_client=inference_client,
        log=logging.getLogger(__name__),
    )
    groundedness = create_groundedness_analysis_evaluator(
        inference_client=inference_client,
        log=logging.getLogger(__name__),
    )
    params = _params(trace_id="")
    params = EvaluatorParams(
        input=params.input,
        output={"traceId": "3" * 32},
        expected=params.expected,
        metadata=params.metadata,
    )

    correctness_result = await correctness.evaluate(params)
    groundedness_result = await groundedness.evaluate(params)

    assert correctness_result.score is None
    assert correctness_result.label == "correctness-analysis"
    assert correctness_result.explanation == ("Factuality: FULLY_SUPPORTED, Relevance: RELEVANT, Sequence: MATCH")
    assert groundedness_result.score is None
    assert groundedness_result.label == "groundedness-analysis"
    assert groundedness_result.metadata == {"summary_verdict": "GROUNDED"}
    assert evaluate.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "creator"),
    [
        ("latency", create_latency_evaluator),
        ("input_tokens", create_input_tokens_evaluator),
        ("output_tokens", create_output_tokens_evaluator),
        ("tool_calls", create_tool_calls_evaluator),
    ],
)
async def test_code_evaluator_creators_use_kibana(
    name: str,
    creator: Any,
) -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = _response()
    evaluator = creator(client=client)

    result = await evaluator.evaluate(_params())

    assert result.score is not None
    request = client.evaluate.await_args.args[0]
    assert [config.name for config in request.evaluators] == [name]


@pytest.mark.asyncio
async def test_kibana_evaluators_batch_one_request_and_map_scores() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = _response()
    evaluators = kibana_evaluators(
        _configs(),
        client=client,
        instrumentation_profile="otel-genai-attributes",
    )

    results = await asyncio.gather(*(evaluator.evaluate(_params()) for evaluator in evaluators))

    assert [evaluator.name for evaluator in evaluators] == [
        "Factuality",
        "Relevance",
        "Sequence Accuracy",
        "Groundedness",
        "latency",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    ]
    assert [result.score for result in results] == [0.9, 1.0, 0.8, 0.95, 2.5, 123, 45, 3]
    assert results[0].metadata == {
        "summary": {
            "factual_accuracy_summary": "FULLY_SUPPORTED",
            "relevance_summary": "RELEVANT",
            "sequence_accuracy_summary": "MATCH",
        }
    }
    assert results[3].label == "GROUNDED"
    assert client.evaluate.await_count == 1

    request = client.evaluate.await_args.args[0]
    assert request.subject.traces[0].trace_id == "1" * 32
    assert request.subject.traces[0].reference_data == {"expected": "Expected answer"}
    assert request.subject.instrumentation is not None
    assert request.subject.instrumentation.profile == "otel-genai-attributes"
    assert [config.model_dump(exclude_none=True) for config in request.evaluators] == [
        {"name": "correctness", "connector_id": "evaluator-connector"},
        {"name": "groundedness", "connector_id": "evaluator-connector"},
        {"name": "latency"},
        {"name": "input_tokens"},
        {"name": "output_tokens"},
        {"name": "tool_calls"},
    ]


@pytest.mark.asyncio
async def test_kibana_evaluators_request_each_trace_once() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = _response()
    evaluators = kibana_evaluators(_configs(), client=client)

    await evaluators[0].evaluate(_params("1" * 32))
    await evaluators[1].evaluate(_params("1" * 32))
    await evaluators[0].evaluate(_params("2" * 32))

    assert client.evaluate.await_count == 2


@pytest.mark.asyncio
async def test_kibana_evaluators_fall_back_to_task_output_trace_id() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = _response()
    evaluator = kibana_evaluators(
        [KibanaEvaluatorConfig(name="latency", kind="CODE")],
        client=client,
    )[0]
    params = _params(trace_id="")
    params = EvaluatorParams(
        input=params.input,
        output={"traceId": "3" * 32},
        expected=params.expected,
        metadata=params.metadata,
    )

    await evaluator.evaluate(params)

    request = client.evaluate.await_args.args[0]
    assert request.subject.traces[0].trace_id == "3" * 32


@pytest.mark.asyncio
async def test_kibana_evaluators_use_first_score_without_sub_score_mapping() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = EvaluateResponse(
        results=[
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="custom", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="different_name", score=0.75)],
            )
        ]
    )
    evaluator = kibana_evaluators(
        [KibanaEvaluatorConfig(name="custom", kind="CODE")],
        client=client,
    )[0]

    result = await evaluator.evaluate(_params())

    assert result.score == 0.75


@pytest.mark.asyncio
async def test_kibana_evaluators_preserve_unavailable_and_partial_errors() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = EvaluateResponse(
        results=[
            EvaluateResult(
                status="ok",
                evaluator=EvaluatorIdentity(name="latency", version="1.0.0", kind="code"),
                scores=[EvaluatorScore(name="latency", score=None, label="unavailable")],
            ),
            EvaluateResult(
                status="error",
                evaluator=EvaluatorIdentity(name="groundedness", version="1.0.0", kind="llm"),
                error=EvaluatorError(code="evidence_unmet", message="Tool-call evidence is missing"),
            ),
        ]
    )
    evaluators = kibana_evaluators(
        [
            KibanaEvaluatorConfig(name="latency", kind="CODE"),
            KibanaEvaluatorConfig(
                name="groundedness",
                kind="LLM",
                connector_id="evaluator-connector",
                sub_scores=[KibanaSubScore(key="groundedness", evaluator_name="Groundedness")],
            ),
        ],
        client=client,
    )

    latency, groundedness = await asyncio.gather(*(evaluator.evaluate(_params()) for evaluator in evaluators))

    assert latency.score is None
    assert latency.label == "unavailable"
    assert groundedness.score is None
    assert groundedness.label == "error"
    assert groundedness.explanation == "Tool-call evidence is missing"
    assert groundedness.metadata == {
        "evaluator": {"name": "groundedness", "version": "1.0.0", "kind": "llm"},
        "error_code": "evidence_unmet",
    }
    assert client.evaluate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (EvaluateResponse(results=[]), 'No evaluation result returned for "latency"'),
        (
            EvaluateResponse(
                results=[
                    EvaluateResult(
                        status="ok",
                        evaluator=EvaluatorIdentity(
                            name="latency",
                            version="1.0.0",
                            kind="llm",
                        ),
                        scores=[EvaluatorScore(name="latency", score=2.5)],
                    )
                ]
            ),
            'Unexpected evaluator kind returned for "latency"',
        ),
        (
            EvaluateResponse(
                results=[
                    EvaluateResult(
                        status="ok",
                        evaluator=EvaluatorIdentity(
                            name="latency",
                            version="1.0.0",
                            kind="code",
                        ),
                        scores=[],
                    )
                ]
            ),
            'No score returned for "latency"',
        ),
    ],
)
async def test_kibana_evaluators_report_invalid_responses(
    response: EvaluateResponse,
    message: str,
) -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.return_value = response
    evaluator = kibana_evaluators(
        [KibanaEvaluatorConfig(name="latency", kind="CODE")],
        client=client,
    )[0]

    result = await evaluator.evaluate(_params())

    assert result.score is None
    assert result.label == "error"
    assert result.explanation == message


@pytest.mark.asyncio
async def test_kibana_evaluators_propagate_request_errors() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    client.evaluate.side_effect = RuntimeError("Kibana unavailable")
    evaluator = kibana_evaluators(
        [KibanaEvaluatorConfig(name="latency", kind="CODE")],
        client=client,
    )[0]

    with pytest.raises(RuntimeError, match="Kibana unavailable"):
        await evaluator.evaluate(_params())


@pytest.mark.asyncio
async def test_kibana_evaluators_do_not_request_without_trace_id() -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)
    evaluators = kibana_evaluators(
        [KibanaEvaluatorConfig(name="latency", kind="CODE")],
        client=client,
    )

    result = await evaluators[0].evaluate(_params(trace_id=""))

    assert result.score is None
    assert result.label == "unavailable"
    client.evaluate.assert_not_awaited()


@pytest.mark.parametrize(
    ("configs", "message"),
    [
        ([], "At least one Kibana evaluator is required"),
        (
            [
                KibanaEvaluatorConfig(name="latency", kind="CODE"),
                KibanaEvaluatorConfig(name="latency", kind="CODE"),
            ],
            'Duplicate Kibana evaluator "latency"',
        ),
        (
            [KibanaEvaluatorConfig(name="correctness", kind="LLM")],
            'connector_id is required for LLM evaluator "correctness"',
        ),
        (
            [
                KibanaEvaluatorConfig(
                    name="correctness",
                    kind="LLM",
                    connector_id="connector",
                    sub_scores=[
                        KibanaSubScore(key="factuality", evaluator_name="score"),
                        KibanaSubScore(key="relevance", evaluator_name="score"),
                    ],
                )
            ],
            'Duplicate evaluator name "score"',
        ),
    ],
)
def test_kibana_evaluators_validate_configs(
    configs: list[KibanaEvaluatorConfig],
    message: str,
) -> None:
    client = AsyncMock(spec=KibanaEvaluatorsClient)

    with pytest.raises(ValueError, match=message):
        kibana_evaluators(configs, client=client)
