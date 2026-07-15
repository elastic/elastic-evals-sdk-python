# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Run Agent Builder evaluation example."""

from __future__ import annotations

import asyncio

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
from elastic_evals.tracing import init_tracing
from elastic_evals.types import EvaluatorParams

from examples.agent_builder.datasets.ambiguous_queries import ambiguous_queries_dataset
from examples.agent_builder.tasks.agent_builder import agent_builder_task


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

    await client.run_experiment(
        dataset=ambiguous_queries_dataset,
        task=task,
        evaluators=quantitative_evaluators,
    )


if __name__ == "__main__":
    asyncio.run(main())
