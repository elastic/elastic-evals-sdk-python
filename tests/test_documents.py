# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

from typing import Any

from elastic_evals.api import Ci, Environment, Model, RunMetadata
from elastic_evals.export.documents import build_ingest_scores_request
from elastic_evals.types import EvaluationResult, EvaluationRun, RunData


def _common_kwargs() -> dict[str, Any]:
    return {
        "run_id": "run-3",
        "experiment_id": "exp-3",
        "suite_id": "suite-3",
        "task_model": Model(id="task-model"),
        "evaluator_model": Model(id="eval-model"),
        "run_metadata": RunMetadata(total_repetitions=2, git_branch="main", git_commit_sha="abc123"),
        "environment": Environment(hostname="worker-3"),
        "ci": None,
        "dataset_id": "dataset-3",
        "dataset_name": "dataset-name-3",
        "example_id": "example-3",
        "example_index": 1,
        "example_input": {"question": "hi"},
        "task_run": RunData(
            example_index=1,
            repetition=0,
            input={"question": "hi"},
            expected=None,
            metadata=None,
            output={"answer": "hello"},
            trace_id="task-trace",
        ),
        "experiment_name": "batched",
    }


def test_build_ingest_scores_request_puts_all_evaluation_runs_under_one_header() -> None:
    runs = [
        EvaluationRun(name="latency", result=EvaluationResult(score=0.5), trace_id="t-1"),
        EvaluationRun(name="correctness", result=EvaluationResult(score=1.0), trace_id="t-2"),
    ]

    payload = build_ingest_scores_request(**_common_kwargs(), evaluation_runs=runs)
    serialized = payload.model_dump(exclude_none=True)

    assert serialized["experiment_id"] == "exp-3"
    assert serialized["metadata"]["execution_id"] == "run-3"
    assert [item["evaluator"]["name"] for item in serialized["scores"]] == ["latency", "correctness"]
    assert [item["evaluator"]["score"] for item in serialized["scores"]] == [0.5, 1.0]
    assert {item["example"]["id"] for item in serialized["scores"]} == {"example-3"}


def test_build_ingest_scores_request_matches_scores_contract() -> None:
    payload = build_ingest_scores_request(
        run_id="run-1",
        experiment_id="exp-1",
        suite_id="suite-1",
        task_model=Model(id="task-model", family="gpt", provider="openai"),
        evaluator_model=Model(id="eval-model", family="gpt", provider="openai"),
        run_metadata=RunMetadata(
            total_repetitions=3,
            git_branch="main",
            git_commit_sha="abc123",
        ),
        environment=Environment(hostname="worker-1"),
        ci=Ci(),
        dataset_id="dataset-1",
        dataset_name="dataset-name",
        example_id="example-1",
        example_index=2,
        example_input={"messages": [{"role": "user", "content": "hello"}]},
        task_run=RunData(
            example_index=2,
            repetition=1,
            input={"messages": [{"role": "user", "content": "hello"}]},
            expected={"answer": "world"},
            metadata={"topic": "greeting"},
            output={"answer": {"text": "world", "citations": ["doc-1"]}},
            trace_id="task-trace",
        ),
        evaluation_runs=[
            EvaluationRun(
                name="correctness",
                result=EvaluationResult(
                    score=0.9,
                    label="pass",
                    explanation="matches expected output",
                    metadata={"reasoning": "close match"},
                ),
                trace_id="eval-trace",
            )
        ],
        experiment_name="named experiment",
    )

    assert payload.model_dump(exclude_none=True) == {
        "experiment_id": "exp-1",
        "experiment_name": "named experiment",
        "task_model": {"id": "task-model", "family": "gpt", "provider": "openai"},
        "evaluator_model": {"id": "eval-model", "family": "gpt", "provider": "openai"},
        "metadata": {
            "execution_id": "run-1",
            "suite_id": "suite-1",
            "total_repetitions": 3,
            "hostname": "worker-1",
            "git": {"branch": "main", "commit_sha": "abc123"},
        },
        "scores": [
            {
                "example": {
                    "id": "example-1",
                    "index": 2,
                    "dataset": {"id": "dataset-1", "name": "dataset-name"},
                    "input": {"messages": [{"role": "user", "content": "hello"}]},
                },
                "task": {
                    "repetition_index": 1,
                    "trace_id": "task-trace",
                    "output": {"answer": {"text": "world", "citations": ["doc-1"]}},
                },
                "evaluator": {
                    "name": "correctness",
                    "score": 0.9,
                    "label": "pass",
                    "explanation": "matches expected output",
                    "metadata": {"reasoning": "close match"},
                    "trace_id": "eval-trace",
                },
            }
        ],
    }


def test_build_ingest_scores_request_omits_evaluator_none_fields() -> None:
    payload = build_ingest_scores_request(
        run_id="run-2",
        experiment_id="exp-2",
        suite_id=None,
        task_model=Model(id="task-model"),
        evaluator_model=Model(id="eval-model"),
        run_metadata=RunMetadata(total_repetitions=1),
        environment=Environment(hostname="worker-2"),
        ci=None,
        dataset_id="dataset-2",
        dataset_name="dataset-name-2",
        example_id="example-2",
        example_index=0,
        example_input={"nested": {"input": ["a", "b"]}},
        task_run=RunData(
            example_index=0,
            repetition=0,
            input={"nested": {"input": ["a", "b"]}},
            expected=None,
            metadata=None,
            output={"nested": {"result": [1, 2, 3]}},
            trace_id=None,
        ),
        evaluation_runs=[EvaluationRun(name="groundedness", result=None, trace_id=None)],
    )

    serialized = payload.model_dump(exclude_none=True)
    evaluator = serialized["scores"][0]["evaluator"]
    task = serialized["scores"][0]["task"]
    example = serialized["scores"][0]["example"]

    assert evaluator == {"name": "groundedness"}
    assert "score" not in evaluator
    assert "label" not in evaluator
    assert "explanation" not in evaluator

    assert task["output"] == {"nested": {"result": [1, 2, 3]}}
    assert example["input"] == {"nested": {"input": ["a", "b"]}}
