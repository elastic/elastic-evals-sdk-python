# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from elastic_evals.api import IngestScoresRequest, IngestScoresResponse
from elastic_evals.api.scores_client import KibanaScoresClient
from elastic_evals.export import UNKNOWN_MODEL_ID, KibanaScoreSink
from elastic_evals.types import EvaluationResult, EvaluationRun, ExampleResult, ExampleWithId, RunContext, RunData


def _context(**overrides: Any) -> RunContext:
    fields: dict[str, Any] = {
        "run_id": "run-1",
        "experiment_id": "exp-1",
        "experiment_name": "named",
        "suite_id": "suite-1",
        "dataset_id": "dataset-1",
        "dataset_name": "dataset",
        "repetitions": 2,
        "hostname": "worker",
        "git_branch": "main",
        "git_commit_sha": "abc",
    }
    fields.update(overrides)
    return RunContext(**fields)


def _result(context: RunContext, runs: list[EvaluationRun], example_input: Any = None) -> ExampleResult:
    example_input = {"q": "hi"} if example_input is None else example_input
    return ExampleResult(
        context=context,
        example=ExampleWithId(id="ex-1", input=example_input if isinstance(example_input, dict) else {}),
        example_index=3,
        repetition=1,
        task_run=RunData(
            example_index=3,
            repetition=1,
            input={"q": "hi"},
            expected=None,
            metadata=None,
            output={"a": "hello"},
            trace_id="task-trace",
        ),
        evaluation_runs=runs,
    )


def _client() -> AsyncMock:
    client = AsyncMock(spec=KibanaScoresClient)
    client.ingest_scores.return_value = IngestScoresResponse(ingested=1, conflicted=0, failed=[])
    return client


def _sent(client: AsyncMock) -> IngestScoresRequest:
    client.ingest_scores.assert_awaited_once()
    return client.ingest_scores.await_args.args[0]


@pytest.mark.asyncio
async def test_write_sends_all_evaluator_results_in_one_request() -> None:
    client = _client()
    runs = [
        EvaluationRun(name="latency", result=EvaluationResult(score=0.5), trace_id="t-1"),
        EvaluationRun(name="correctness", result=EvaluationResult(score=1.0), trace_id="t-2"),
        EvaluationRun(name="groundedness", result=None, trace_id=None),
    ]

    await KibanaScoreSink(client).write(_result(_context(connector_id="conn"), runs))

    request = _sent(client)
    assert [item.evaluator.name for item in request.scores] == ["latency", "correctness", "groundedness"]
    assert request.experiment_id == "exp-1"
    assert request.experiment_name == "named"
    assert request.metadata.execution_id == "run-1"
    assert request.metadata.total_repetitions == 2
    assert request.metadata.git is not None and request.metadata.git.branch == "main"
    assert request.scores[0].example.dataset.id == "dataset-1"
    assert request.scores[0].example.index == 3
    assert request.scores[0].task.repetition_index == 1


@pytest.mark.asyncio
async def test_write_skips_the_request_when_there_are_no_evaluator_results() -> None:
    client = _client()

    await KibanaScoreSink(client).write(_result(_context(connector_id="conn"), []))

    client.ingest_scores.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_model_prefers_configured_model_then_connector_then_unknown() -> None:
    runs = [EvaluationRun(name="latency")]

    client = _client()
    await KibanaScoreSink(client).write(
        _result(_context(model={"id": "gpt-x", "family": "gpt", "provider": "openai"}, connector_id="conn"), runs)
    )
    assert _sent(client).task_model.model_dump() == {"id": "gpt-x", "family": "gpt", "provider": "openai"}

    client = _client()
    await KibanaScoreSink(client).write(_result(_context(connector_id="conn"), runs))
    assert _sent(client).task_model.id == "conn"

    client = _client()
    await KibanaScoreSink(client).write(_result(_context(), runs))
    assert _sent(client).task_model.id == UNKNOWN_MODEL_ID


@pytest.mark.asyncio
async def test_task_model_coerces_non_string_values_from_config() -> None:
    client = _client()

    await KibanaScoreSink(client).write(
        _result(_context(model={"id": 123, "family": 4, "provider": 7}), [EvaluationRun(name="latency")])
    )

    assert _sent(client).task_model.model_dump() == {"id": "123", "family": "4", "provider": "7"}


@pytest.mark.asyncio
async def test_evaluator_model_does_not_inherit_task_model_family_or_provider() -> None:
    client = _client()

    await KibanaScoreSink(client).write(
        _result(
            _context(model={"id": "gpt-x", "family": "gpt", "provider": "openai"}, evaluator_connector_id="judge"),
            [EvaluationRun(name="latency")],
        )
    )

    assert _sent(client).evaluator_model.model_dump() == {"id": "judge", "family": None, "provider": None}


@pytest.mark.asyncio
async def test_evaluator_model_prefers_evaluator_connector_then_connector_then_unknown() -> None:
    runs = [EvaluationRun(name="latency")]

    client = _client()
    await KibanaScoreSink(client).write(_result(_context(connector_id="conn", evaluator_connector_id="judge"), runs))
    assert _sent(client).evaluator_model.id == "judge"

    client = _client()
    await KibanaScoreSink(client).write(_result(_context(connector_id="conn"), runs))
    assert _sent(client).evaluator_model.id == "conn"

    client = _client()
    await KibanaScoreSink(client).write(_result(_context(), runs))
    assert _sent(client).evaluator_model.id == UNKNOWN_MODEL_ID
