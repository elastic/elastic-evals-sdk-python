"""Score export document models."""

from __future__ import annotations

from datetime import datetime
from socket import gethostname
from typing import Any
from typing import Iterable

from pydantic import BaseModel, Field

from elastic_evals.export.git_metadata import get_git_metadata
from elastic_evals.tracing import get_current_trace_id
from elastic_evals.types import EvaluationResult, RanExperiment


class DatasetInfo(BaseModel):
    id: str
    name: str


class ExampleInfo(BaseModel):
    id: str
    index: int
    dataset: DatasetInfo


class ModelInfo(BaseModel):
    id: str | None = None
    family: str
    provider: str


class TaskInfo(BaseModel):
    trace_id: str | None
    repetition_index: int
    model: ModelInfo


class EvaluatorInfo(BaseModel):
    name: str
    score: float | None
    label: str | None
    explanation: str | None
    metadata: dict[str, Any] | None
    trace_id: str | None
    model: ModelInfo


class RunMetadata(BaseModel):
    git_branch: str | None
    git_commit_sha: str | None
    total_repetitions: int


class EnvironmentInfo(BaseModel):
    hostname: str


class EvaluationScoreDocument(BaseModel):
    timestamp: datetime = Field(alias="@timestamp")
    run_id: str
    experiment_id: str
    example: ExampleInfo
    task: TaskInfo
    evaluator: EvaluatorInfo
    run_metadata: RunMetadata
    environment: EnvironmentInfo

    model_config = {"populate_by_name": True}


def build_flattened_score_documents(
    *,
    experiments: Iterable[RanExperiment],
    task_model: ModelInfo,
    evaluator_model: ModelInfo,
    run_id: str,
    total_repetitions: int,
) -> list[EvaluationScoreDocument]:
    documents: list[EvaluationScoreDocument] = []
    timestamp = datetime.utcnow()
    git_metadata = get_git_metadata()
    host_name = gethostname()

    for experiment in experiments:
        dataset_id = experiment.dataset_id
        dataset_name = experiment.dataset_name or dataset_id
        runs_by_id = experiment.runs or {}
        runs_list = list(runs_by_id.values())

        for eval_run in experiment.evaluation_runs or []:
            run_entry = None
            if eval_run.experiment_run_id and eval_run.experiment_run_id in runs_by_id:
                run_entry = runs_by_id[eval_run.experiment_run_id]
            elif (
                eval_run.example_index is not None
                and eval_run.repetition_index is not None
            ):
                run_entry = next(
                    (
                        run
                        for run in runs_list
                        if run.example_index == eval_run.example_index
                        and run.repetition == eval_run.repetition_index
                    ),
                    None,
                )

            example_index = (
                run_entry.example_index if run_entry else (eval_run.example_index or 0)
            )
            if eval_run.repetition_index is not None:
                repetition_index = eval_run.repetition_index
            elif run_entry:
                repetition_index = run_entry.repetition
            else:
                repetition_index = 0
            example_id = (
                eval_run.example_id
                or getattr(run_entry, "dataset_example_id", None)
                or str(example_index)
            )
            trace_id = run_entry.trace_id if run_entry else get_current_trace_id()

            evaluator_result = eval_run.result or EvaluationResult()
            documents.append(
                EvaluationScoreDocument.model_validate(
                    {
                        "@timestamp": timestamp,
                        "run_id": run_id,
                        "experiment_id": experiment.id or "",
                        "example": ExampleInfo(
                            id=example_id,
                            index=example_index,
                            dataset=DatasetInfo(id=dataset_id, name=dataset_name),
                        ),
                        "task": TaskInfo(
                            trace_id=trace_id,
                            repetition_index=repetition_index,
                            model=task_model,
                        ),
                        "evaluator": EvaluatorInfo(
                            name=eval_run.name,
                            score=evaluator_result.score,
                            label=evaluator_result.label,
                            explanation=evaluator_result.explanation,
                            metadata=evaluator_result.metadata,
                            trace_id=eval_run.trace_id or get_current_trace_id(),
                            model=evaluator_model,
                        ),
                        "run_metadata": RunMetadata(
                            git_branch=git_metadata.branch,
                            git_commit_sha=git_metadata.commit_sha,
                            total_repetitions=total_repetitions,
                        ),
                        "environment": EnvironmentInfo(hostname=host_name),
                    }
                )
            )

    return documents
