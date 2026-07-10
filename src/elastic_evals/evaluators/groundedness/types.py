# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

"""Types for groundedness evaluation outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GroundednessEvidence(BaseModel):
    tool_call_id: str | None
    tool_id: str | None
    evidence_snippet: str | None


class GroundednessClaimAnalysis(BaseModel):
    claim: str
    centrality: Literal["central", "peripheral"]
    centrality_reason: str
    verdict: Literal[
        "FULLY_SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "CONTRADICTED",
        "NOT_FOUND",
        "UNGROUNDED_BUT_DISCLOSED",
    ]
    evidence: GroundednessEvidence | None
    explanation: str


class GroundednessAnalysis(BaseModel):
    summary_verdict: Literal[
        "GROUNDED",
        "GROUNDED_WITH_DISCLOSURE",
        "MINOR_HALLUCINATIONS",
        "MAJOR_HALLUCINATIONS",
    ]
    analysis: list[GroundednessClaimAnalysis]
