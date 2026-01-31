"""Reporting data models for elastic-evals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from elastic_evals.export.documents import ModelInfo

StatKey = Literal["percentage", "mean", "median", "std_dev", "min", "max"]


class EvaluatorStats(BaseModel):
    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    count: int
    percentage: float


class DatasetScore(BaseModel):
    id: str
    name: str
    num_examples: int
    evaluator_scores: dict[str, list[float]]
    experiment_id: str


class DatasetScoreWithStats(DatasetScore):
    evaluator_stats: dict[str, EvaluatorStats]


class EvaluatorDisplayOptions(BaseModel):
    decimal_places: int | None = None
    unit_suffix: str | None = None
    stats_to_include: list[StatKey] | None = None


class EvaluatorDisplayGroup(BaseModel):
    evaluator_names: list[str]
    combined_column_name: str


class ReportDisplayOptions(BaseModel):
    evaluator_display_options: dict[str, EvaluatorDisplayOptions] = Field(default_factory=dict)
    evaluator_display_groups: list[EvaluatorDisplayGroup] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    dataset_scores_with_stats: list[DatasetScoreWithStats]
    model: ModelInfo
    evaluator_model: ModelInfo
    repetitions: int
    run_id: str
