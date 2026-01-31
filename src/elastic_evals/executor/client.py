"""Experiment execution client for elastic-evals."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.tracing import get_current_trace_id, with_evaluator_span, with_task_span
from elastic_evals.types import (
    EvaluationDataset,
    EvaluationRun,
    Evaluator,
    EvaluatorParams,
    Example,
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
    log_task_execution,
)

ExperimentTask = Callable[[Example], Awaitable[TaskOutput]]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return True
    return False


def _normalize_example(example: Example) -> dict[str, Any]:
    metadata = example.metadata or {}
    normalized_metadata = {key: val for key, val in metadata.items() if not _is_empty(val)}
    return {
        "input": example.input,
        "output": example.output,
        "metadata": normalized_metadata,
    }


class ElasticEvalsClient:
    def __init__(self, config: ElasticEvalsConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self._logger = logger or config.logger
        self._experiments: list[RanExperiment] = []
        self._inference_client: KibanaInferenceClient | None = None

    def get_inference_client(self) -> KibanaInferenceClient:
        if self._inference_client is None:
            connector_id = self.config.evaluator_connector_id or self.config.connector_id
            self._inference_client = KibanaInferenceClient(
                kibana_url=self.config.kibana_url,
                connector_id=connector_id,
                auth=self.config.kibana_auth,
            )
        return self._inference_client

    async def run_experiment(
        self,
        *,
        dataset: EvaluationDataset,
        task: ExperimentTask,
        evaluators: list[Evaluator],
        metadata: dict[str, Any] | None = None,
        concurrency: int | None = None,
    ) -> RanExperiment:
        run_concurrency = max(1, concurrency or self.config.concurrency)
        semaphore = asyncio.Semaphore(run_concurrency)
        dataset_id = self._compute_dataset_id(dataset)
        experiment_id = str(uuid.uuid4())
        repetitions = self.config.repetitions

        runs: dict[str, RunData] = {}
        evaluation_runs: list[EvaluationRun] = []

        log_experiment_start(self.config.run_id, dataset.name, len(evaluators), run_concurrency)

        async def run_example(example: Example, example_index: int, repetition: int) -> None:
            async with semaphore:
                run_key = f"{example_index}-{repetition}-{uuid.uuid4()}"
                log_task_execution(dataset_id, example_index, repetition)

                async def task_runner() -> TaskOutput:
                    return await task(example)

                task_output, task_trace_id = await with_task_span("task", {}, task_runner)

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

                for evaluator in evaluators:
                    log_evaluator_start(evaluator.name, example_index, repetition)
                    params = EvaluatorParams(
                        input=example.input,
                        output=task_output,
                        expected=example.output,
                        metadata=example.metadata,
                    )

                    async def evaluator_runner() -> Any:
                        return await evaluator.evaluate(params)

                    result, eval_trace_id = await with_evaluator_span(
                        evaluator.name, {}, evaluator_runner
                    )
                    evaluation_runs.append(
                        EvaluationRun(
                            name=evaluator.name,
                            result=result,
                            example_index=example_index,
                            repetition_index=repetition,
                            experiment_run_id=run_key,
                            trace_id=eval_trace_id or get_current_trace_id(),
                        )
                    )
                    log_evaluator_complete(evaluator.name, example_index, repetition)

        jobs: list[Awaitable[None]] = []
        for rep in range(repetitions):
            for example_index, example in enumerate(dataset.examples):
                jobs.append(run_example(example, example_index, rep))

        await asyncio.gather(*jobs)
        log_experiment_complete(experiment_id)

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

    def _compute_dataset_id(self, dataset: EvaluationDataset) -> str:
        payload = {
            "name": dataset.name,
            "description": dataset.description,
            "examples": [_normalize_example(example) for example in dataset.examples],
        }
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
