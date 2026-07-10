# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""DTOs for /internal/evals/scores ingest API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Model(BaseModel):
    id: str
    family: str | None = None
    provider: str | None = None


# Kept for use in executor/client.py and documents.py — not sent directly to the API.
class RunMetadata(BaseModel):
    total_repetitions: int
    git_branch: str | None = None
    git_commit_sha: str | None = None


class Environment(BaseModel):
    hostname: str


class BuildkiteMetadata(BaseModel):
    build_id: str | None = None
    job_id: str | None = None
    build_url: str | None = None
    pipeline_slug: str | None = None
    pull_request: str | None = None
    branch: str | None = None
    commit: str | None = None


class Ci(BaseModel):
    buildkite: BuildkiteMetadata | None = None


class IngestGit(BaseModel):
    branch: str | None = None
    commit_sha: str | None = None


class IngestMetadata(BaseModel):
    """Maps to the `metadata` object in the Kibana /internal/evals/scores request body."""

    total_repetitions: int
    hostname: str
    execution_id: str | None = None
    suite_id: str | None = None
    git: IngestGit | None = None
    ci: BuildkiteMetadata | None = None


class Dataset(BaseModel):
    id: str
    name: str


class IngestExample(BaseModel):
    id: str
    index: int
    dataset: Dataset
    input: dict[str, Any] | None = None


class IngestTask(BaseModel):
    repetition_index: int
    trace_id: str | None = None
    output: dict[str, Any] | None = None


class IngestEvaluator(BaseModel):
    name: str
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None
    trace_id: str | None = None


class IngestScoreItem(BaseModel):
    example: IngestExample
    task: IngestTask
    evaluator: IngestEvaluator


class IngestScoresRequest(BaseModel):
    experiment_id: str
    experiment_name: str | None = None
    task_model: Model
    evaluator_model: Model
    metadata: IngestMetadata
    scores: list[IngestScoreItem]


class IngestScoresFailure(BaseModel):
    index: int
    status: int
    reason: str


class IngestScoresResponse(BaseModel):
    ingested: int
    conflicted: int
    failed: list[IngestScoresFailure]
