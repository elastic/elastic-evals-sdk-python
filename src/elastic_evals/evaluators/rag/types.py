"""Types for RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeAlias, TypeVar

TOutput = TypeVar("TOutput")
TReferenceOutput = TypeVar("TReferenceOutput")


@dataclass(frozen=True)
class RetrievedDoc:
    index: str
    id: str


GroundTruth: TypeAlias = dict[str, dict[str, float]]
RetrievedDocsExtractor: TypeAlias = Callable[[TOutput], list[RetrievedDoc]]
GroundTruthExtractor: TypeAlias = Callable[[TReferenceOutput], GroundTruth]


@dataclass(frozen=True)
class RagEvaluatorConfig(Generic[TOutput, TReferenceOutput]):
    k: int
    extract_retrieved_docs: RetrievedDocsExtractor[TOutput]
    extract_ground_truth: GroundTruthExtractor[TReferenceOutput]
    relevance_threshold: float | None = None
    filter_by_ground_truth_indices: bool | None = None
