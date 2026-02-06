"""Run Agent Builder evaluation example."""

from __future__ import annotations

import asyncio
from typing import Iterable

from elasticsearch import AsyncElasticsearch

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.correctness import (
    create_correctness_analysis_evaluator,
    create_quantitative_correctness_evaluators,
)
from elastic_evals.evaluators.groundedness import (
    create_groundedness_analysis_evaluator,
    create_quantitative_groundedness_evaluator,
)
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.export import EvaluationScoreRepository
from elastic_evals.export.documents import ModelInfo
from elastic_evals.export.repository import build_flattened_score_documents
from elastic_evals.reporting import DefaultReporter
from elastic_evals.reporting.stats import calculate_evaluator_stats
from elastic_evals.reporting.types import DatasetScoreWithStats
from elastic_evals.tracing import init_tracing
from elastic_evals.types import EvaluatorParams

from examples.agent_builder.datasets.ambiguous_queries import ambiguous_queries_dataset
from examples.agent_builder.tasks.agent_builder import agent_builder_task


def _model_info_from_config(config: ElasticEvalsConfig) -> ModelInfo:
    model = config.model or {}
    return ModelInfo(
        id=model.get("id"),
        family=model.get("family", "unknown"),
        provider=model.get("provider", "unknown"),
    )


def _build_dataset_scores_with_stats(
    *, repetitions: int, evaluator_names: Iterable[str], experiment
) -> DatasetScoreWithStats:
    total_examples = len(ambiguous_queries_dataset.examples) * repetitions
    evaluator_scores: dict[str, list[float]] = {name: [] for name in evaluator_names}

    for evaluation_run in experiment.evaluation_runs or []:
        score = evaluation_run.result.score if evaluation_run.result else None
        if score is None:
            continue
        evaluator_scores.setdefault(evaluation_run.name, []).append(score)

    return DatasetScoreWithStats(
        id=experiment.dataset_id,
        name=experiment.dataset_name or ambiguous_queries_dataset.name,
        num_examples=total_examples,
        evaluator_scores=evaluator_scores,
        evaluator_stats={
            name: calculate_evaluator_stats(scores, total_examples)
            for name, scores in evaluator_scores.items()
        },
        experiment_id=experiment.id,
    )


async def main() -> None:
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)

    client = ElasticEvalsClient(config)

    print(f"Running evaluation with run_id: {config.run_id}")
    print(f"Dataset: {ambiguous_queries_dataset.name}")
    print(f"Examples: {len(ambiguous_queries_dataset.examples)}")
    print(f"Repetitions: {config.repetitions}")
    print()

    inference_client = client.get_inference_client()
    log = config.logger
    correctness_analysis = create_correctness_analysis_evaluator(
        inference_client=inference_client, log=log
    )
    groundedness_analysis = create_groundedness_analysis_evaluator(
        inference_client=inference_client, log=log
    )
    quantitative_evaluators = [
        *create_quantitative_correctness_evaluators(),
        create_quantitative_groundedness_evaluator(),
    ]

    async def task(example):
        response = await agent_builder_task(example, config)
        params = EvaluatorParams(
            input=example.input,
            output=response,
            expected=example.output,
            metadata=example.metadata,
        )
        correctness_result = await correctness_analysis.evaluate(params)
        groundedness_result = await groundedness_analysis.evaluate(params)
        response["correctnessAnalysis"] = correctness_result.metadata
        response["groundednessAnalysis"] = groundedness_result.metadata
        return response

    result = await client.run_experiment(
        dataset=ambiguous_queries_dataset,
        task=task,
        evaluators=quantitative_evaluators,
    )

    if config.evaluations_es_url:
        es = AsyncElasticsearch(config.evaluations_es_url)
        repository = EvaluationScoreRepository(es, log)
        task_model = _model_info_from_config(config)
        evaluator_model = _model_info_from_config(config)
        documents = build_flattened_score_documents(
            experiments=[result],
            task_model=task_model,
            evaluator_model=evaluator_model,
            run_id=config.run_id,
            total_repetitions=config.repetitions,
        )
        await repository.export_scores(documents)
        await es.close()

    reporter = DefaultReporter()
    evaluator_names = [evaluator.name for evaluator in quantitative_evaluators]
    dataset_scores = [
        _build_dataset_scores_with_stats(
            repetitions=config.repetitions,
            evaluator_names=evaluator_names,
            experiment=result,
        )
    ]
    reporter.report(
        dataset_scores=dataset_scores,
        task_model=_model_info_from_config(config),
        evaluator_model=_model_info_from_config(config),
    )


if __name__ == "__main__":
    asyncio.run(main())
