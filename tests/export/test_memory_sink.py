# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

from elastic_evals.export import InMemoryScoreSink
from elastic_evals.types import EvaluationRun, ExampleResult, ExampleWithId, RunContext, RunData


def _result(example_index: int) -> ExampleResult:
    context = RunContext(
        run_id="run-1",
        experiment_id="exp-1",
        experiment_name=None,
        suite_id=None,
        dataset_id="dataset-1",
        dataset_name="dataset",
        repetitions=1,
        hostname="worker",
    )
    return ExampleResult(
        context=context,
        example=ExampleWithId(id=f"ex-{example_index}", input={"q": "hi"}),
        example_index=example_index,
        repetition=0,
        task_run=RunData(
            example_index=example_index, repetition=0, input={"q": "hi"}, expected=None, metadata=None, output="a"
        ),
        evaluation_runs=[EvaluationRun(name="latency")],
    )


@pytest.mark.asyncio
async def test_sink_starts_empty_and_keeps_results_in_write_order() -> None:
    sink = InMemoryScoreSink()
    assert sink.results == []

    await sink.write(_result(1))
    await sink.write(_result(0))

    assert [result.example_index for result in sink.results] == [1, 0]
    assert sink.results[0].evaluation_runs[0].name == "latency"
