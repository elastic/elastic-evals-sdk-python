# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Experiment execution client for elastic-evals."""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from elastic_evals.api import (
    Ci,
    Environment,
    KibanaDatasetsClient,
    KibanaEvaluatorsClient,
    Model,
    RunMetadata,
    UpsertDatasetExamplePayload,
    compute_dataset_id,
)
from elastic_evals.api.scores_client import KibanaScoresClient
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.export.documents import build_ingest_score_item
from elastic_evals.export.git_metadata import get_git_metadata
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.tracing import (
    get_current_trace_id,
    with_evaluator_span,
    with_task_span,
)
from elastic_evals.types import (
    EvaluationDataset,
    EvaluationRun,
    Evaluator,
    EvaluatorParams,
    Example,
    ExampleWithId,
    RanExperiment,
    RunData,
    TaskOutput,
)
from elastic_evals.utils.logging import (
    log_evaluation_start,
    log_evaluator_complete,
    log_evaluator_start,
    log_experiment_complete,
    log_experiment_start,
    log_results_url,
    log_task_execution,
)

ExperimentTask = Callable[[Example], Awaitable[TaskOutput]]


class ElasticEvalsClient:
    def __init__(self, config: ElasticEvalsConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self._logger = logger or config.logger
        self._experiments: list[RanExperiment] = []
        self._inference_client: KibanaInferenceClient | None = None
        self._datasets_client = KibanaDatasetsClient(
            kibana_url=self.config.kibana_url,
            api_key=self.config.kibana_api_key,
        )
        self._scores_client = KibanaScoresClient(
            kibana_url=self.config.kibana_url,
            api_key=self.config.kibana_api_key,
        )
        self._evaluators_client = KibanaEvaluatorsClient(
            kibana_url=self.config.kibana_url,
            api_key=self.config.kibana_api_key,
        )

    def get_inference_client(self) -> KibanaInferenceClient:
        if self._inference_client is None:
            connector_id = self.config.evaluator_connector_id or self.config.connector_id
            self._inference_client = KibanaInferenceClient(
                kibana_url=self.config.kibana_url,
                connector_id=connector_id,
                api_key=self.config.kibana_api_key,
            )
        return self._inference_client

    def get_evaluators_client(self) -> KibanaEvaluatorsClient:
        return self._evaluators_client

    async def run_experiment(
        self,
        *,
        dataset: EvaluationDataset,
        task: ExperimentTask,
        evaluators: list[Evaluator],
        experiment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        concurrency: int | None = None,
    ) -> RanExperiment:
        run_concurrency = max(1, concurrency or self.config.concurrency)
        semaphore = asyncio.Semaphore(run_concurrency)
        dataset_id = compute_dataset_id(dataset.name, self._datasets_client.space_id)
        experiment_id = str(uuid.uuid4())
        repetitions = self.config.repetitions
        task_model = self._build_task_model()
        evaluator_model = self._build_evaluator_model()
        run_metadata = self._build_run_metadata()
        environment = Environment(hostname=socket.gethostname())
        ci: Ci | None = None

        await self._datasets_client.upsert(
            dataset.name,
            dataset.description,
            [
                UpsertDatasetExamplePayload(
                    input=self._dict_or_none(example.input),
                    output=self._dict_or_none(example.output),
                    metadata=self._dict_or_none(example.metadata),
                )
                for example in dataset.examples
            ],
        )
        upstream_dataset = await self._datasets_client.get(dataset_id)
        upstream_examples = [
            ExampleWithId(
                id=example.id,
                input=example.input or {},
                output=example.output,
                metadata=example.metadata,
            )
            for example in upstream_dataset.examples
        ]

        runs: dict[str, RunData] = {}
        evaluation_runs: list[EvaluationRun] = []

        log_experiment_start(self.config.run_id, dataset.name, len(evaluators), run_concurrency)

        async def run_example(example: ExampleWithId, example_index: int, repetition: int) -> None:
            async with semaphore:
                run_key = f"{example_index}-{repetition}-{uuid.uuid4()}"
                log_task_execution(dataset_id, example_index, repetition)

                async def task_runner() -> TaskOutput:
                    return await task(example)

                task_output, task_trace_id = await with_task_span("task", {}, task_runner)

                if isinstance(task_output, dict):
                    task_trace_id = task_output.pop("_interaction_trace_id", task_trace_id)

                runs[run_key] = RunData(
                    example_index=example_index,
                    repetition=repetition,
                    input=example.input,
                    expected=example.output,
                    metadata=example.metadata or {},
                    output=task_output,
                    trace_id=task_trace_id,
                )

                log_evaluation_start(example_index, repetition, len(evaluators))

                params = EvaluatorParams(
                    input=example.input,
                    output=task_output,
                    expected=example.output,
                    metadata=example.metadata,
                    trace_id=task_trace_id,
                )

                for evaluator in evaluators:
                    log_evaluator_start(evaluator.name, example_index, repetition)

                    async def evaluator_runner() -> Any:
                        return await evaluator.evaluate(params)

                    result, eval_trace_id = await with_evaluator_span(evaluator.name, {}, evaluator_runner)
                    evaluation_runs.append(
                        EvaluationRun(
                            name=evaluator.name,
                            result=result,
                            example_index=example_index,
                            repetition_index=repetition,
                            experiment_run_id=run_key,
                            trace_id=eval_trace_id or get_current_trace_id(),
                            example_id=example.id,
                        )
                    )
                    task_run = runs[run_key]
                    score_payload = build_ingest_score_item(
                        run_id=self.config.run_id,
                        experiment_id=experiment_id,
                        suite_id=self.config.suite_id,
                        task_model=task_model,
                        evaluator_model=evaluator_model,
                        run_metadata=run_metadata,
                        environment=environment,
                        ci=ci,
                        dataset_id=dataset_id,
                        dataset_name=dataset.name,
                        example_id=example.id,
                        example_index=example_index,
                        example_input=self._dict_or_none(example.input),
                        task_run=task_run,
                        evaluation_run=evaluation_runs[-1],
                        experiment_name=experiment_name,
                    )
                    await self._scores_client.ingest_scores(score_payload)
                    log_evaluator_complete(evaluator.name, example_index, repetition)

        jobs: list[Awaitable[None]] = []
        for rep in range(repetitions):
            for example_index, example in enumerate(upstream_examples):
                jobs.append(run_example(example, example_index, rep))

        await asyncio.gather(*jobs)
        log_experiment_complete(experiment_id)
        log_results_url(self.config.kibana_url, self.config.run_id)

        experiment_metadata: dict[str, Any] = {"run_id": self.config.run_id}
        if metadata:
            experiment_metadata.update(metadata)
        if self.config.model is not None:
            experiment_metadata["model"] = self.config.model

        ran_experiment = RanExperiment(
            id=experiment_id,
            dataset_id=dataset_id,
            dataset_name=dataset.name,
            dataset_description=dataset.description,
            runs=runs,
            evaluation_runs=evaluation_runs,
            experiment_metadata=experiment_metadata,
        )
        self._experiments.append(ran_experiment)
        return ran_experiment

    async def get_ran_experiments(self) -> list[RanExperiment]:
        return self._experiments

    @staticmethod
    def _dict_or_none(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        return None

    def _build_task_model(self) -> Model:
        configured_model = self.config.model or {}
        model_id = configured_model.get("id")
        model_family = configured_model.get("family")
        model_provider = configured_model.get("provider")
        return Model(
            id=str(model_id) if model_id is not None else self.config.connector_id,
            family=str(model_family) if model_family is not None else None,
            provider=str(model_provider) if model_provider is not None else None,
        )

    def _build_evaluator_model(self) -> Model:
        return Model(id=self.config.evaluator_connector_id or self.config.connector_id)

    def _build_run_metadata(self) -> RunMetadata:
        git_metadata = get_git_metadata()
        return RunMetadata(
            total_repetitions=self.config.repetitions,
            git_branch=git_metadata.branch,
            git_commit_sha=git_metadata.commit_sha,
        )
