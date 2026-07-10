# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

"""Scoring helpers for groundedness evaluation."""

from __future__ import annotations

from elastic_evals.evaluators.groundedness.types import GroundednessAnalysis

CLAIM_FACTUAL_SCORE_MAP = {
    "FULLY_SUPPORTED": 1.0,
    "PARTIALLY_SUPPORTED": {"central": 0.9, "peripheral": 0.95},
    "CONTRADICTED": {"central": 0.0, "peripheral": 0.1},
    "NOT_IN_GROUND_TRUTH": {"central": 0.1, "peripheral": 0.5},
    "UNGROUNDED_BUT_DISCLOSED": {"central": 0.75, "peripheral": 0.9},
}


def calculate_groundedness_score(groundedness_analysis: GroundednessAnalysis) -> float:
    analysis = groundedness_analysis.analysis
    if not analysis:
        return 0.0

    product_of_scores = 1.0
    for claim in analysis:
        verdict = claim.verdict or "NOT_IN_GROUND_TRUTH"
        centrality = claim.centrality or "peripheral"
        score_entry = CLAIM_FACTUAL_SCORE_MAP.get(verdict)
        claim_score = 0.0
        if isinstance(score_entry, dict):
            claim_score = score_entry.get(centrality, 0.0)
        elif isinstance(score_entry, (int, float)):
            claim_score = float(score_entry)
        product_of_scores *= claim_score

    num_claims = len(analysis)
    return product_of_scores ** (1 / num_claims) if product_of_scores > 0 else 0.0
