# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Score sink that posts each finished example to the Kibana evals scores API."""

from __future__ import annotations

from typing import Any

from elastic_evals.api.scores_client import KibanaScoresClient
from elastic_evals.api.scores_models import Environment, Model, RunMetadata
from elastic_evals.export.documents import build_ingest_scores_request
from elastic_evals.types import ExampleResult, RunContext

UNKNOWN_MODEL_ID = "unknown"
"""Model id sent to Kibana when neither a model nor a connector is configured."""


class KibanaScoreSink:
    """Translates one `ExampleResult` into a single Kibana ingest request and sends it.

    One request per (example, repetition) keeps ingestion incremental: a crash mid-run
    loses only the in-flight example. Per-example batches are bounded by the number of
    evaluators, so the API's 1000-item batch limit is never approached here.
    """

    def __init__(self, client: KibanaScoresClient) -> None:
        self._client = client

    async def write(self, result: ExampleResult) -> None:
        if not result.evaluation_runs:
            return
        context = result.context
        request = build_ingest_scores_request(
            run_id=context.run_id,
            experiment_id=context.experiment_id,
            experiment_name=context.experiment_name,
            suite_id=context.suite_id,
            task_model=_task_model(context),
            evaluator_model=_evaluator_model(context),
            run_metadata=RunMetadata(
                total_repetitions=context.repetitions,
                git_branch=context.git_branch,
                git_commit_sha=context.git_commit_sha,
            ),
            environment=Environment(hostname=context.hostname),
            ci=None,
            dataset_id=context.dataset_id,
            dataset_name=context.dataset_name,
            example_id=result.example.id,
            example_index=result.example_index,
            example_input=_dict_or_none(result.example.input),
            task_run=result.task_run,
            evaluation_runs=result.evaluation_runs,
        )
        await self._client.ingest_scores(request)


def _task_model(context: RunContext) -> Model:
    """Describe the model under test for the Kibana score header.

    `context.model` is the user's optional `ElasticEvalsConfig.model` dict and may carry
    `id`, `family` and `provider`. `Model.id` is required by the API.
    """
    model = context.model or {}
    model_id = model.get("id")
    family = model.get("family")
    provider = model.get("provider")
    return Model(
        id=str(model_id) if model_id is not None else context.connector_id or UNKNOWN_MODEL_ID,
        family=str(family) if family is not None else None,
        provider=str(provider) if provider is not None else None,
    )


def _evaluator_model(context: RunContext) -> Model:
    """Describe the judge model used by LLM evaluators for the Kibana score header."""
    return Model(id=context.evaluator_connector_id or context.connector_id or UNKNOWN_MODEL_ID)


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None
