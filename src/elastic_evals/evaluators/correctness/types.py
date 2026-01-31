"""Types for correctness evaluation outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CorrectnessSummary(BaseModel):
    factual_accuracy_summary: str
    relevance_summary: str
    sequence_accuracy_summary: str


class CorrectnessClaimAnalysis(BaseModel):
    claim: str
    centrality: Literal["central", "peripheral"]
    centrality_reason: str
    verdict: Literal["FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "NOT_IN_GROUND_TRUTH"]
    sequence_match: Literal["MATCH", "MISMATCH", "NOT_APPLICABLE"]
    justification_snippet: str | None
    explanation: str


class CorrectnessAnalysis(BaseModel):
    summary: CorrectnessSummary
    analysis: list[CorrectnessClaimAnalysis]
