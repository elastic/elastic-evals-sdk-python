# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

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
