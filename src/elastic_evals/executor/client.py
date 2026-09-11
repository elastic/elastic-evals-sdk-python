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

from elastic_evals.api import KibanaDatasetsClient, KibanaEvaluatorsClient, compute_dataset_id
from elastic_evals.api.scores_client import KibanaScoresClient
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.datasets import InMemoryDatasetStore, KibanaDatasetStore
from elastic_evals.export import InMemoryScoreSink, KibanaScoreSink
from elastic_evals.export.git_metadata import get_git_metadata
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.tracing import (
    get_current_trace_id,
    with_evaluator_span,
    with_task_span,
)
from elastic_evals.types import (
    DatasetStore,
    EvaluationDataset,
    EvaluationRun,
    Evaluator,
    EvaluatorParams,
    Example,
    ExampleResult,
    ExampleWithId,
    RanExperiment,
    RunContext,
    RunData,
    ScoreSink,
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
    """Runs experiments: calls the task, runs evaluators, and hands results to a score sink.

    Where examples come from and where scores go are pluggable. By default both are Kibana,
    built lazily from `config` on first use. Pass `dataset_store` / `score_sink` to replace
    either, or use `ElasticEvalsClient.local` for a fully in-memory run.
    """

    def __init__(
        self,
        config: ElasticEvalsConfig,
        logger: logging.Logger | None = None,
        *,
        dataset_store: DatasetStore | None = None,
        score_sink: ScoreSink | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or config.logger
        self._experiments: list[RanExperiment] = []
        self._inference_client: KibanaInferenceClient | None = None
        self._evaluators_client: KibanaEvaluatorsClient | None = None
        self._dataset_store = dataset_store
        self._score_sink = score_sink
        # Only the default Kibana sink has a results page worth linking to.
        self._log_kibana_results_url = score_sink is None

    @classmethod
    def local(cls, config: ElasticEvalsConfig, logger: logging.Logger | None = None) -> ElasticEvalsClient:
        """Client that runs entirely in memory: no Kibana calls and no connector required."""
        return cls(config, logger, dataset_store=InMemoryDatasetStore(), score_sink=InMemoryScoreSink())

    @property
    def dataset_store(self) -> DatasetStore:
        if self._dataset_store is None:
            self._dataset_store = KibanaDatasetStore(
                KibanaDatasetsClient(kibana_url=self.config.kibana_url, api_key=self.config.kibana_api_key)
            )
        return self._dataset_store

    @property
    def score_sink(self) -> ScoreSink:
        if self._score_sink is None:
            self._score_sink = KibanaScoreSink(
                KibanaScoresClient(kibana_url=self.config.kibana_url, api_key=self.config.kibana_api_key)
            )
        return self._score_sink

    def get_inference_client(self) -> KibanaInferenceClient:
        if self._inference_client is None:
            connector_id = self.config.evaluator_connector_id or self.config.connector_id
            if not connector_id:
                raise ValueError(
                    "An inference connector is required for LLM calls: set ElasticEvalsConfig.connector_id "
                    "or evaluator_connector_id (env CONNECTOR_ID / EVALUATION_CONNECTOR_ID)."
                )
            self._inference_client = KibanaInferenceClient(
                kibana_url=self.config.kibana_url,
                connector_id=connector_id,
                api_key=self.config.kibana_api_key,
            )
        return self._inference_client

    def get_evaluators_client(self) -> KibanaEvaluatorsClient:
        if self._evaluators_client is None:
            self._evaluators_client = KibanaEvaluatorsClient(
                kibana_url=self.config.kibana_url,
                api_key=self.config.kibana_api_key,
            )
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
        dataset_id = compute_dataset_id(dataset.name)
        experiment_id = str(uuid.uuid4())
        repetitions = self.config.repetitions
        git_metadata = get_git_metadata()
        context = RunContext(
            run_id=self.config.run_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            suite_id=self.config.suite_id,
            dataset_id=dataset_id,
            dataset_name=dataset.name,
            repetitions=repetitions,
            hostname=socket.gethostname(),
            model=self.config.model,
            connector_id=self.config.connector_id,
            evaluator_connector_id=self.config.evaluator_connector_id,
            git_branch=git_metadata.branch,
            git_commit_sha=git_metadata.commit_sha,
        )

        examples = await self.dataset_store.resolve(dataset)

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

                example_evaluation_runs: list[EvaluationRun] = []
                for evaluator in evaluators:
                    log_evaluator_start(evaluator.name, example_index, repetition)

                    async def evaluator_runner() -> Any:
                        return await evaluator.evaluate(params)

                    result, eval_trace_id = await with_evaluator_span(evaluator.name, {}, evaluator_runner)
                    evaluation_run = EvaluationRun(
                        name=evaluator.name,
                        result=result,
                        example_index=example_index,
                        repetition_index=repetition,
                        experiment_run_id=run_key,
                        trace_id=eval_trace_id or get_current_trace_id(),
                        example_id=example.id,
                    )
                    evaluation_runs.append(evaluation_run)
                    example_evaluation_runs.append(evaluation_run)
                    log_evaluator_complete(evaluator.name, example_index, repetition)

                # One write per finished example keeps ingestion incremental: a crash later in
                # the run cannot lose scores that were already handed to the sink.
                await self.score_sink.write(
                    ExampleResult(
                        context=context,
                        example=example,
                        example_index=example_index,
                        repetition=repetition,
                        task_run=runs[run_key],
                        evaluation_runs=example_evaluation_runs,
                    )
                )

        jobs: list[Awaitable[None]] = []
        for rep in range(repetitions):
            for example_index, example in enumerate(examples):
                jobs.append(run_example(example, example_index, rep))

        await asyncio.gather(*jobs)
        log_experiment_complete(experiment_id)
        if self._log_kibana_results_url:
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
