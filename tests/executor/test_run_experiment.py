# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Core loop tests. Everything runs in memory: no HTTP, no mocks of transport."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest

from elastic_evals.api import compute_dataset_id
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.datasets import InMemoryDatasetStore
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.export import InMemoryScoreSink
from elastic_evals.tracing import TracingConfig
from elastic_evals.types import (
    EvaluationDataset,
    EvaluationResult,
    EvaluatorParams,
    Example,
    ExampleResult,
    ExampleWithId,
)


class RecordingEvaluator:
    kind: Literal["LLM", "CODE"] = "CODE"

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = score
        self.params: list[EvaluatorParams] = []

    async def evaluate(self, params: EvaluatorParams) -> EvaluationResult:
        self.params.append(params)
        return EvaluationResult(score=self.score, label=self.name)


def _config(**overrides: Any) -> ElasticEvalsConfig:
    fields: dict[str, Any] = {"tracing": TracingConfig(enabled=False), "concurrency": 1}
    fields.update(overrides)
    return ElasticEvalsConfig(**fields)


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="greetings",
        description="two greetings",
        examples=[
            Example(input={"q": "hi"}, output="hello", metadata={"lang": "en"}),
            Example(input={"q": "hola"}, output="hello", metadata=None),
        ],
    )


async def _echo_task(example: Example) -> dict[str, Any]:
    return {"answer": example.input["q"]}


def _sorted(sink: InMemoryScoreSink) -> list[ExampleResult]:
    return sorted(sink.results, key=lambda result: (result.repetition, result.example_index))


@pytest.mark.asyncio
async def test_task_receives_store_examples_and_evaluators_receive_task_output() -> None:
    seen: list[ExampleWithId] = []

    async def task(example: Example) -> str:
        assert isinstance(example, ExampleWithId)
        seen.append(example)
        return f"answer:{example.input['q']}"

    evaluator = RecordingEvaluator("exact", 1.0)
    client = ElasticEvalsClient.local(_config())
    expected_ids = [example.id for example in await InMemoryDatasetStore().resolve(_dataset())]

    await client.run_experiment(dataset=_dataset(), task=task, evaluators=[evaluator])

    assert [example.id for example in seen] == expected_ids
    assert [params.output for params in evaluator.params] == ["answer:hi", "answer:hola"]
    assert evaluator.params[0].expected == "hello"
    assert evaluator.params[0].metadata == {"lang": "en"}


@pytest.mark.asyncio
async def test_sink_receives_one_batch_per_example_with_all_evaluators_in_order() -> None:
    sink = InMemoryScoreSink()
    evaluators = [RecordingEvaluator("a", 0.1), RecordingEvaluator("b", 0.2), RecordingEvaluator("c", 0.3)]
    client = ElasticEvalsClient(_config(), dataset_store=InMemoryDatasetStore(), score_sink=sink)

    result = await client.run_experiment(dataset=_dataset(), task=_echo_task, evaluators=evaluators)

    batches = _sorted(sink)
    assert [batch.example_index for batch in batches] == [0, 1]
    assert all([run.name for run in batch.evaluation_runs] == ["a", "b", "c"] for batch in batches)
    assert batches[0].task_run.output == {"answer": "hi"}
    assert batches[0].evaluation_runs[1].result is not None
    assert batches[0].evaluation_runs[1].result.score == 0.2
    assert len(result.evaluation_runs) == 6
    assert len(result.runs) == 2


@pytest.mark.asyncio
async def test_repetitions_produce_a_batch_per_example_per_repetition() -> None:
    client = ElasticEvalsClient.local(_config(repetitions=2))

    await client.run_experiment(dataset=_dataset(), task=_echo_task, evaluators=[RecordingEvaluator("a", 1.0)])

    sink = client.score_sink
    assert isinstance(sink, InMemoryScoreSink)
    assert {(batch.example_index, batch.repetition) for batch in sink.results} == {(0, 0), (1, 0), (0, 1), (1, 1)}


@pytest.mark.asyncio
async def test_run_context_carries_run_level_facts() -> None:
    client = ElasticEvalsClient.local(
        _config(run_id="run-9", suite_id="suite-1", model={"id": "m"}, connector_id="conn", repetitions=1)
    )

    ran = await client.run_experiment(
        dataset=_dataset(), task=_echo_task, evaluators=[RecordingEvaluator("a", 1.0)], experiment_name="named"
    )

    sink = client.score_sink
    assert isinstance(sink, InMemoryScoreSink)
    context = sink.results[0].context
    assert context.run_id == "run-9"
    assert context.experiment_id == ran.id
    assert context.experiment_name == "named"
    assert context.suite_id == "suite-1"
    assert context.dataset_id == compute_dataset_id("greetings") == ran.dataset_id
    assert context.model == {"id": "m"}
    assert context.connector_id == "conn"
    assert context.repetitions == 1


@pytest.mark.asyncio
async def test_concurrency_limits_examples_in_flight() -> None:
    in_flight = 0
    max_in_flight = 0

    async def slow_task(example: Example) -> str:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "x"

    client = ElasticEvalsClient.local(_config(concurrency=2, repetitions=2))

    await client.run_experiment(dataset=_dataset(), task=slow_task, evaluators=[])

    assert max_in_flight == 2


@pytest.mark.asyncio
async def test_zero_evaluators_still_writes_each_example() -> None:
    client = ElasticEvalsClient.local(_config())

    await client.run_experiment(dataset=_dataset(), task=_echo_task, evaluators=[])

    sink = client.score_sink
    assert isinstance(sink, InMemoryScoreSink)
    assert len(sink.results) == 2
    assert all(batch.evaluation_runs == [] for batch in sink.results)


@pytest.mark.asyncio
async def test_failing_sink_aborts_the_run_and_records_no_experiment() -> None:
    class ExplodingSink:
        async def write(self, result: ExampleResult) -> None:
            raise RuntimeError("sink down")

    client = ElasticEvalsClient(_config(), dataset_store=InMemoryDatasetStore(), score_sink=ExplodingSink())

    with pytest.raises(RuntimeError, match="sink down"):
        await client.run_experiment(dataset=_dataset(), task=_echo_task, evaluators=[])

    assert await client.get_ran_experiments() == []


@pytest.mark.asyncio
async def test_local_run_does_not_log_a_kibana_results_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "elastic_evals.executor.client.log_results_url", lambda url, run_id: calls.append((url, run_id))
    )
    client = ElasticEvalsClient.local(_config())

    await client.run_experiment(dataset=_dataset(), task=_echo_task, evaluators=[])

    assert calls == []


@pytest.mark.asyncio
async def test_interaction_trace_id_is_popped_from_dict_output_before_storage() -> None:
    async def task(example: Example) -> dict[str, Any]:
        return {"answer": "x", "_interaction_trace_id": "trace-42"}

    evaluator = RecordingEvaluator("a", 1.0)
    client = ElasticEvalsClient.local(_config())

    await client.run_experiment(dataset=_dataset(), task=task, evaluators=[evaluator])

    sink = client.score_sink
    assert isinstance(sink, InMemoryScoreSink)
    assert sink.results[0].task_run.trace_id == "trace-42"
    assert sink.results[0].task_run.output == {"answer": "x"}
    assert evaluator.params[0].trace_id == "trace-42"
