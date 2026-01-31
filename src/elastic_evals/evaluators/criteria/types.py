"""Types for criteria evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

EvaluationCriterionText = str


class EvaluationCriterionStructured(BaseModel):
    id: str
    text: str
    score: float | None = None


EvaluationCriterion = EvaluationCriterionStructured | EvaluationCriterionText


class CriterionEvaluation(BaseModel):
    id: str
    result: Literal["PASS", "FAIL", "N/A"]
    reason: str | None = None
