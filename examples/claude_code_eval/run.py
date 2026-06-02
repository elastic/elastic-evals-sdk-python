"""Run Claude Code eval example."""

from __future__ import annotations

import asyncio
import os
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

from examples.claude_code_eval.datasets.coding_tasks import coding_tasks_dataset
from examples.claude_code_eval.evaluators.latency import create_latency_evaluator
from examples.claude_code_eval.evaluators.tool_use import create_tool_use_evaluator
from examples.claude_code_eval.tasks.claude_code import claude_code_task


def _read_criteria(metadata: dict[str, Any] | None) -> list[EvaluationCriterion]:
    if not metadata:
        return []
    raw = metadata.get("criteria")
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, str)]


def create_criteria_evaluator_from_metadata(
    *, inference_client: Any, log: Any
) -> Evaluator:
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

    edot_endpoint = os.environ.get("EDOT_ENDPOINT", "http://localhost:4318")

    print(f"Running evaluation with run_id: {config.run_id}")
    print(f"Dataset: {coding_tasks_dataset.name}")
    print(f"Examples: {len(coding_tasks_dataset.examples)}")
    print(f"Repetitions: {config.repetitions}")
    print(f"EDOT collector: {edot_endpoint}")
    print()

    evaluators: list[Evaluator] = [
        create_latency_evaluator(),
        create_tool_use_evaluator(),
    ]

    # LLM criteria evaluator is optional — only added when an evaluator connector is configured.
    if config.evaluator_connector_id or config.connector_id:
        inference_client = client.get_inference_client()
        evaluators.append(
            create_criteria_evaluator_from_metadata(
                inference_client=inference_client,
                log=config.logger,
            )
        )

    async def task(example):
        return await claude_code_task(example, config)

    await client.run_experiment(
        dataset=coding_tasks_dataset,
        task=task,
        evaluators=evaluators,
    )


if __name__ == "__main__":
    asyncio.run(main())
