"""Run chatbot RAG app evaluation example."""

from __future__ import annotations

import asyncio
from typing import Any

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.criteria import (
    EvaluationCriterion,
    create_criteria_evaluator,
)
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.tracing import init_tracing
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

from examples.chatbot_rag_app.datasets.workplace_questions import (
    workplace_questions_dataset,
)
from examples.chatbot_rag_app.evaluators.source_citation import (
    create_source_citation_evaluator,
)
from examples.chatbot_rag_app.tasks.chatbot_rag import chatbot_rag_task


def _read_criteria(metadata: dict[str, Any] | None) -> list[EvaluationCriterion]:
    if not metadata:
        return []
    raw_criteria = metadata.get("criteria")
    if not isinstance(raw_criteria, list):
        return []
    return [criterion for criterion in raw_criteria if isinstance(criterion, str)]


def create_metadata_criteria_evaluator(*, inference_client: Any, log: Any) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        criteria = _read_criteria(params.metadata)
        evaluator = create_criteria_evaluator(
            inference_client=inference_client,
            criteria=criteria,
            log=log,
        )
        return await evaluator.evaluate(params)

    return SimpleEvaluator(name="criteria", kind="LLM", evaluate=evaluate)


async def main() -> None:
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)

    client = ElasticEvalsClient(config)

    print(f"Running evaluation with run_id: {config.run_id}")
    print(f"Dataset: {workplace_questions_dataset.name}")
    print(f"Examples: {len(workplace_questions_dataset.examples)}")
    print(f"Repetitions: {config.repetitions}")
    print()

    inference_client = client.get_inference_client()
    log = config.logger
    evaluators = [
        create_metadata_criteria_evaluator(
            inference_client=inference_client,
            log=log,
        ),
        create_source_citation_evaluator(),
    ]

    async def task(example):
        return await chatbot_rag_task(example, config)

    await client.run_experiment(
        dataset=workplace_questions_dataset,
        task=task,
        evaluators=evaluators,
    )


if __name__ == "__main__":
    asyncio.run(main())
