"""Reporting data models for elastic-evals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from elastic_evals.export.documents import ModelInfo

StatKey = Literal["mean", "median", "std_dev", "min", "max"]


class StatsDisplay(BaseModel):
    mean: float
    median: float
    std_dev: float
    min: float
    max: float
    count: int


class EvaluatorStats(BaseModel):
    dataset_id: str
    dataset_name: str
    evaluator_name: str
    stats: StatsDisplay


class EvaluatorDisplayOptions(BaseModel):
    decimal_places: int | None = None
    unit_suffix: str | None = None
    stats_to_include: list[StatKey] | None = None


class EvaluatorDisplayGroup(BaseModel):
    evaluator_names: list[str]
    combined_column_name: str


class ReportDisplayOptions(BaseModel):
    evaluator_display_options: dict[str, EvaluatorDisplayOptions] = Field(
        default_factory=dict
    )
    evaluator_display_groups: list[EvaluatorDisplayGroup] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    stats: list[EvaluatorStats]
    model: ModelInfo
    evaluator_model: ModelInfo
    repetitions: int
    run_id: str


class RunStats(BaseModel):
    stats: list[EvaluatorStats]
    task_model: ModelInfo
    evaluator_model: ModelInfo
    total_repetitions: int
