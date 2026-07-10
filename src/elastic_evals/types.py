# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared Pydantic types for elastic-evals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

from pydantic import BaseModel

TInput = TypeVar("TInput", bound=dict[str, Any])
TExpected = TypeVar("TExpected")
TMetadata = TypeVar("TMetadata", bound=dict[str, Any] | None)
TTaskOutput = TypeVar("TTaskOutput")
TExample = TypeVar("TExample", bound="Example")

TaskOutput: TypeAlias = Any


class Example(BaseModel, Generic[TInput, TExpected, TMetadata]):
    input: TInput
    output: TExpected | None = None
    metadata: TMetadata | None = None


class ExampleWithId(
    Example[TInput, TExpected, TMetadata], Generic[TInput, TExpected, TMetadata]
):
    id: str


class EvaluationDataset(BaseModel, Generic[TExample]):
    """User-provided dataset definition used by the runner.

    During `run_experiment`, examples are upserted and re-fetched from Kibana, then the
    task callable receives those upstream JSON-shaped examples (including server ids)
    rather than the original in-memory Pydantic instances.
    """

    name: str
    description: str
    examples: list[TExample]
    metadata: dict[str, Any] | None = None


class EvaluationDatasetWithId(EvaluationDataset[TExample], Generic[TExample]):
    id: str


class EvaluationResult(BaseModel):
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    reasoning: str | None = None
    details: Any | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvaluatorParams(Generic[TInput, TExpected, TMetadata, TTaskOutput]):
    input: TInput
    output: TTaskOutput
    expected: TExpected | None
    metadata: TMetadata


class Evaluator(Protocol, Generic[TInput, TExpected, TMetadata, TTaskOutput]):
    name: str
    kind: Literal["LLM", "CODE"]

    async def evaluate(
        self, params: EvaluatorParams[TInput, TExpected, TMetadata, TTaskOutput]
    ) -> EvaluationResult: ...


class RunData(BaseModel):
    example_index: int
    repetition: int
    input: dict[str, Any]
    expected: Any | None
    metadata: dict[str, Any] | None
    output: Any
    trace_id: str | None = None


class EvaluationRun(BaseModel):
    name: str
    result: EvaluationResult | None = None
    example_index: int | None = None
    repetition_index: int | None = None
    experiment_run_id: str | None = None
    trace_id: str | None = None
    example_id: str | None = None


class RanExperiment(BaseModel):
    id: str
    dataset_id: str
    dataset_name: str
    dataset_description: str | None = None
    runs: dict[str, RunData]
    evaluation_runs: list[EvaluationRun]
    experiment_metadata: dict[str, Any] | None = None


__all__ = [
    "EvaluationDataset",
    "EvaluationDatasetWithId",
    "EvaluationResult",
    "EvaluationRun",
    "Evaluator",
    "EvaluatorParams",
    "Example",
    "ExampleWithId",
    "RanExperiment",
    "RunData",
    "TaskOutput",
]
