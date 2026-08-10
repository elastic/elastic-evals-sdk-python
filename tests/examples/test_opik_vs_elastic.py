# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pandas")
pytest.importorskip("orca")

from elastic_evals.api import (  # noqa: E402
    Environment,
    EvaluatorDefinition,
    Model,
    RunMetadata,
)
from elastic_evals.config import ElasticEvalsConfig  # noqa: E402
from elastic_evals.integrations.agent_builder import (  # noqa: E402
    AgentBuilderClient,
    ConverseResponse,
    ConverseStep,
)
from elastic_evals.types import (  # noqa: E402
    EvaluationResult,
    EvaluationRun,
    EvaluatorParams,
    Example,
    RunData,
)
from examples.opik_vs_elastic import run as opik_vs_elastic_run  # noqa: E402
from examples.opik_vs_elastic import run2 as granular_run  # noqa: E402
from examples.opik_vs_elastic.helpers.helpers import _parse_relevant_doc_ids  # noqa: E402, PLC2701

TOOL_ID = "wix-knowledge-search"


def _tool_step(*document_ids: str, tool_id: str = TOOL_ID) -> dict:
    return {
        "type": "tool_call",
        "tool_id": tool_id,
        "results": [
            {"data": {"reference": {"index": "wix_knowledge_base", "id": document_ids[0]}}},
            {
                "data": {
                    "resources": [
                        {
                            "reference": {
                                "index": "wix_knowledge_base",
                                "id": document_id,
                            }
                        }
                        for document_id in document_ids[1:]
                    ]
                }
            },
        ],
    }


def test_parse_relevant_doc_ids() -> None:
    assert _parse_relevant_doc_ids("{'doc-1': 1, 'doc-2': 0, 'doc-3': True}") == [
        "doc-1",
        "doc-3",
    ]
    assert _parse_relevant_doc_ids(None) == []

    with pytest.raises(ValueError, match="must be a dictionary"):
        _parse_relevant_doc_ids("not a dictionary")


@pytest.mark.parametrize(
    "module",
    [opik_vs_elastic_run, granular_run],
    ids=["managed", "granular"],
)
@pytest.mark.asyncio
async def test_agent_builder_task_uses_client_and_preserves_trace(module: Any) -> None:
    client = AsyncMock(spec=AgentBuilderClient)
    client.call_converse.return_value = ConverseResponse(
        message="Use the site settings.",
        steps=[
            ConverseStep(
                type="tool_call",
                tool_id=TOOL_ID,
                results=[{"data": {"reference": {"id": "doc-1"}}}],
            )
        ],
        conversation_id="conversation-1",
        trace_id="1" * 32,
    )

    result = await module.agent_builder_task(
        Example(input={"question": "How do I configure my Wix site?"}),
        client,
        connector_id="task-connector",
        agent_id="wix-agent",
    )

    client.call_converse.assert_awaited_once_with(
        "How do I configure my Wix site?",
        agent_id="wix-agent",
        connector_id="task-connector",
    )
    assert result == {
        "messages": [{"message": "Use the site settings."}],
        "steps": [
            {
                "type": "tool_call",
                "tool_id": TOOL_ID,
                "results": [{"data": {"reference": {"id": "doc-1"}}}],
            }
        ],
        "traceId": "1" * 32,
        "conversation_id": "conversation-1",
        "_interaction_trace_id": "1" * 32,
    }


@pytest.mark.asyncio
async def test_document_recall_evaluator_scores_full_partial_and_zero_recall() -> None:
    evaluator = opik_vs_elastic_run.create_document_recall_evaluator(tool_id=TOOL_ID)
    metadata = {"relevant_doc_ids": ["doc-1", "doc-2"]}

    full = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "How do I configure my Wix site?"},
            output={"steps": [_tool_step("doc-1", "doc-2", "doc-2")]},
            expected={"expected": "Use the site settings."},
            metadata=metadata,
        )
    )
    assert full.score == 1.0
    assert full.label == "PASS"
    assert full.metadata == {
        "expected_document_ids": ["doc-1", "doc-2"],
        "retrieved_document_ids": ["doc-1", "doc-2"],
        "matched_document_ids": ["doc-1", "doc-2"],
        "missing_document_ids": [],
    }

    partial = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "How do I configure my Wix site?"},
            output={"steps": [_tool_step("doc-1")]},
            expected={"expected": "Use the site settings."},
            metadata=metadata,
        )
    )
    assert partial.score == 0.5
    assert partial.label == "PARTIAL"

    zero = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "How do I configure my Wix site?"},
            output={"steps": [_tool_step("doc-1", tool_id="another-tool")]},
            expected={"expected": "Use the site settings."},
            metadata=metadata,
        )
    )
    assert zero.score == 0.0
    assert zero.label == "FAIL"


@pytest.mark.asyncio
async def test_document_recall_evaluator_is_unavailable_without_ground_truth() -> None:
    evaluator = opik_vs_elastic_run.create_document_recall_evaluator(tool_id=TOOL_ID)

    result = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "How do I configure my Wix site?"},
            output={"steps": [_tool_step("doc-1")]},
            expected={"expected": "Use the site settings."},
            metadata={},
        )
    )

    assert result.score is None
    assert result.label == "unavailable"
    assert result.metadata == {
        "expected_document_ids": [],
        "retrieved_document_ids": ["doc-1"],
    }


def _executed_run() -> dict[str, Any]:
    return {
        "key": "run-key",
        "example_id": "example-1",
        "data": RunData(
            example_index=0,
            repetition=0,
            input={"question": "How do I configure my Wix site?"},
            expected={"expected": "Use the site settings."},
            metadata={"relevant_doc_ids": ["doc-1", "doc-2"]},
            output={"steps": [_tool_step("doc-1", "doc-9")]},
            trace_id="1" * 32,
        ),
    }


def test_granular_score_request_preserves_experiment_name_and_batches_scores() -> None:
    executed = _executed_run()
    config = ElasticEvalsConfig(connector_id="task-model")
    scored_runs = [
        (
            executed,
            EvaluationRun(name="score-a", result=EvaluationResult(score=0.5)),
        ),
        (
            executed,
            EvaluationRun(name="score-b", result=EvaluationResult(score=1.0)),
        ),
    ]

    context = {
        "config": config,
        "experiment_id": "experiment-1",
        "experiment_name": "Granular experiment",
        "dataset_id": "dataset-1",
        "dataset_name": "dataset",
        "task_model": Model(id="task-model"),
        "run_metadata": RunMetadata(total_repetitions=1),
        "environment": Environment(hostname="host"),
    }
    payload = granular_run._score_request(
        context,
        evaluator_model=Model(id="evaluator-model"),
        scored_runs=scored_runs,
    )

    assert payload is not None
    assert payload.experiment_name == "Granular experiment"
    assert payload.evaluator_model.id == "evaluator-model"
    assert [score.evaluator.name for score in payload.scores] == ["score-a", "score-b"]
    assert all(score.example.id == "example-1" for score in payload.scores)
    assert all(score.task.trace_id == "1" * 32 for score in payload.scores)


def test_granular_orca_scores_are_namespaced_and_use_stored_run() -> None:
    scored_runs = granular_run._evaluate_with_orca([_executed_run()])

    assert [evaluation.name for _, evaluation in scored_runs] == [
        "orca.precision-at-3",
        "orca.recall-at-3",
        "orca.f1-at-3",
    ]
    assert [evaluation.result.score for _, evaluation in scored_runs if evaluation.result] == [
        pytest.approx(1 / 3),
        pytest.approx(1 / 2),
        pytest.approx(0.4),
    ]
    assert all(
        evaluation.result and evaluation.result.metadata and evaluation.result.metadata["source"] == "orca"
        for _, evaluation in scored_runs
    )


def test_granular_evaluator_configs_use_reference_data_and_llm_connector() -> None:
    definitions = [
        EvaluatorDefinition(
            name="correctness",
            version="1.0.0",
            kind="llm",
            description="Correctness",
        ),
        EvaluatorDefinition(
            name="latency",
            version="1.0.0",
            kind="code",
            description="Latency",
        ),
    ]

    validate_configs, evaluate_configs = granular_run._evaluator_configs(
        definitions,
        "judge-connector",
    )

    assert [config.name for config in validate_configs] == ["correctness", "latency"]
    assert [config.connector_id for config in evaluate_configs] == [
        "judge-connector",
        None,
    ]
    assert granular_run._reference_data(_executed_run()["data"].expected) == {"expected": "Use the site settings."}


def test_granular_workflow_does_not_call_run_experiment() -> None:
    assert "run_experiment(" not in inspect.getsource(granular_run)
