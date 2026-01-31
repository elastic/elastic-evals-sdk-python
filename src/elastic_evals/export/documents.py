"""Score export document models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    id: str
    name: str


class ExampleInfo(BaseModel):
    id: str
    index: int
    input_hash: str
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
