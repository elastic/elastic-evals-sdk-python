# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Scoring helpers for correctness evaluation."""

from __future__ import annotations

from elastic_evals.evaluators.correctness.types import CorrectnessAnalysis

CLAIM_FACTUAL_SCORE_MAP = {
    "FULLY_SUPPORTED": 1.0,
    "PARTIALLY_SUPPORTED": {"central": 0.9, "peripheral": 0.95},
    "CONTRADICTED": {"central": 0.0, "peripheral": 0.1},
    "NOT_IN_GROUND_TRUTH": {"central": 0.1, "peripheral": 0.5},
}


def calculate_factual_score(correctness_evaluation: CorrectnessAnalysis) -> float:
    analysis = correctness_evaluation.analysis
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


def calculate_relevance_score(correctness_evaluation: CorrectnessAnalysis) -> float:
    analysis = correctness_evaluation.analysis
    if not analysis:
        return 0.0

    num_claims = len(analysis)
    central_claims = sum(1 for claim in analysis if claim.centrality == "central")
    return central_claims / num_claims


def calculate_procedural_fidelity_score(
    correctness_evaluation: CorrectnessAnalysis,
) -> float:
    summary = correctness_evaluation.summary
    analysis = correctness_evaluation.analysis

    if summary.sequence_accuracy_summary == "NOT_APPLICABLE":
        return 1.0

    if not analysis:
        return 1.0

    central_claims = [claim for claim in analysis if claim.centrality == "central"]
    if not central_claims:
        return 1.0

    matched = sum(1 for claim in central_claims if claim.sequence_match == "MATCH")
    return matched / len(central_claims)
