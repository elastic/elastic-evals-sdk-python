# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Run Agent Builder evaluation example."""

from __future__ import annotations

import asyncio

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators import (
    KibanaEvaluatorConfig,
    KibanaSubScore,
    kibana_evaluators,
)
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.tracing import init_tracing
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

    connector_id = config.evaluator_connector_id or config.connector_id
    evaluators = kibana_evaluators(
        [
            KibanaEvaluatorConfig(
                name="correctness",
                kind="LLM",
                connector_id=connector_id,
                sub_scores=[
                    KibanaSubScore(key="factuality", evaluator_name="Factuality"),
                    KibanaSubScore(key="relevance", evaluator_name="Relevance"),
                    KibanaSubScore(key="sequence_accuracy", evaluator_name="Sequence Accuracy"),
                ],
            ),
            KibanaEvaluatorConfig(
                name="groundedness",
                kind="LLM",
                connector_id=connector_id,
            ),
        ],
        client=client.get_evaluators_client(),
        log=config.logger,
    )

    async def task(example):
        return await agent_builder_task(example, config)

    await client.run_experiment(
        dataset=ambiguous_queries_dataset,
        task=task,
        evaluators=evaluators,
    )


if __name__ == "__main__":
    asyncio.run(main())
