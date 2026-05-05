from __future__ import annotations

import pytest

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.export.documents import ModelInfo
from elastic_evals.export.documents import build_flattened_score_documents
from elastic_evals.tracing import TracingConfig
from elastic_evals.types import EvaluationDataset, EvaluationResult, Example


@pytest.mark.asyncio
async def test_runner_end_to_end() -> None:
    dataset: EvaluationDataset[Example[dict[str, str], None, None]] = EvaluationDataset(
        name="tiny",
        description="tiny dataset",
        examples=[
            Example(input={"q": "one"}),
            Example(input={"q": "two"}),
        ],
    )

    config = ElasticEvalsConfig(
        connector_id="test-connector",
        evaluations_es_url="http://localhost:9220",
        repetitions=1,
        concurrency=1,
        tracing=TracingConfig(enabled=False),
    )
    client = ElasticEvalsClient(config=config)

    async def task(example: Example) -> dict[str, str]:
        return {"answer": example.input["q"]}

    async def evaluate(_: object) -> EvaluationResult:
        return EvaluationResult(score=1.0)

    evaluator = SimpleEvaluator(name="echo", kind="CODE", evaluate=evaluate)
    result = await client.run_experiment(
        dataset=dataset, task=task, evaluators=[evaluator]
    )

    assert len(result.evaluation_runs) == 2
    assert all(run.result is not None for run in result.evaluation_runs)
    assert all(run.result and run.result.score == 1.0 for run in result.evaluation_runs)
    assert all(run.experiment_run_id for run in result.evaluation_runs)

    documents = build_flattened_score_documents(
        experiments=[result],
        task_model=ModelInfo(family="x", provider="y"),
        evaluator_model=ModelInfo(family="x", provider="y"),
        run_id="r",
        total_repetitions=1,
    )

    assert len(documents) == 2
    assert all(doc.evaluator.score == 1.0 for doc in documents)
