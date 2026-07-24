# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Score ingest request builder helpers."""

from __future__ import annotations

from typing import Any

from elastic_evals.api import (
    Ci,
    Dataset,
    Environment,
    IngestEvaluator,
    IngestExample,
    IngestGit,
    IngestMetadata,
    IngestScoreItem,
    IngestScoresRequest,
    IngestTask,
    Model,
    RunMetadata,
)
from elastic_evals.types import EvaluationRun, RunData


def build_ingest_score_item(
    *,
    run_id: str,
    experiment_id: str,
    suite_id: str | None,
    task_model: Model,
    evaluator_model: Model,
    run_metadata: RunMetadata,
    environment: Environment,
    ci: Ci | None,
    dataset_id: str,
    dataset_name: str,
    example_id: str,
    example_index: int,
    example_input: dict[str, Any] | None,
    task_run: RunData,
    evaluation_run: EvaluationRun,
    experiment_name: str | None = None,
) -> IngestScoresRequest:
    evaluator_result = evaluation_run.result

    git: IngestGit | None = None
    if run_metadata.git_branch or run_metadata.git_commit_sha:
        git = IngestGit(
            branch=run_metadata.git_branch,
            commit_sha=run_metadata.git_commit_sha,
        )

    metadata = IngestMetadata(
        execution_id=run_id,
        suite_id=suite_id,
        total_repetitions=run_metadata.total_repetitions,
        hostname=environment.hostname,
        git=git,
        ci=ci.buildkite if ci else None,
    )

    return IngestScoresRequest(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        task_model=task_model,
        evaluator_model=evaluator_model,
        metadata=metadata,
        scores=[
            IngestScoreItem(
                example=IngestExample(
                    id=example_id,
                    index=example_index,
                    dataset=Dataset(id=dataset_id, name=dataset_name),
                    input=example_input,
                ),
                task=IngestTask(
                    repetition_index=task_run.repetition,
                    trace_id=task_run.trace_id,
                    output=task_run.output if isinstance(task_run.output, dict) else None,
                ),
                evaluator=IngestEvaluator(
                    name=evaluation_run.name,
                    score=evaluator_result.score if evaluator_result else None,
                    label=evaluator_result.label if evaluator_result else None,
                    explanation=evaluator_result.explanation if evaluator_result else None,
                    metadata=evaluator_result.metadata if evaluator_result else None,
                    trace_id=evaluation_run.trace_id,
                ),
            )
        ],
    )
