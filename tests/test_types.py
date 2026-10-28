# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import dataclasses
import warnings

import pytest

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.types import (
    DatasetStore,
    EvaluationDataset,
    EvaluationResult,
    EvaluatorParams,
    ExampleResult,
    ExampleWithId,
    RunContext,
    RunData,
    ScoreStore,
)


def _context() -> RunContext:
    return RunContext(
        run_id="run-1",
        experiment_id="exp-1",
        experiment_name=None,
        suite_id=None,
        dataset_id="dataset-1",
        dataset_name="dataset",
        repetitions=1,
        hostname="worker",
    )


def test_example_result_is_immutable_and_defaults_to_no_evaluation_runs() -> None:
    result = ExampleResult(
        context=_context(),
        example=ExampleWithId(id="ex-1", input={"q": "hi"}),
        example_index=0,
        repetition=0,
        task_run=RunData(example_index=0, repetition=0, input={"q": "hi"}, expected=None, metadata=None, output="a"),
    )

    assert result.evaluation_runs == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.example_index = 1  # type: ignore[misc]


def test_run_context_optional_fields_default_to_none() -> None:
    context = _context()

    assert context.model is None
    assert context.connector_id is None
    assert context.evaluator_connector_id is None
    assert context.git_branch is None
    assert context.git_commit_sha is None


def test_store_and_score_store_accept_any_object_with_matching_methods() -> None:
    class FakeDatasetStore:
        async def resolve(self, dataset: EvaluationDataset) -> list[ExampleWithId]:
            return []

    class FakeScoreStore:
        async def write(self, result: ExampleResult) -> None:
            return None

    store: DatasetStore = FakeDatasetStore()
    score_store: ScoreStore = FakeScoreStore()

    assert callable(store.resolve)
    assert callable(score_store.write)


async def _no_score(params: EvaluatorParams) -> EvaluationResult:
    return EvaluationResult()


def test_uppercase_kind_is_lowercased_with_a_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match='kind="CODE" is deprecated, use "code"'):
        evaluator = SimpleEvaluator(name="exact", kind="CODE", evaluate=_no_score)

    assert evaluator.kind == "code"


def test_lowercase_kind_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluator = SimpleEvaluator(name="exact", kind="llm", evaluate=_no_score)

    assert evaluator.kind == "llm"
