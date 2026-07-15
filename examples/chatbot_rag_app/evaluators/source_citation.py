# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Source citation evaluator for the chatbot RAG example."""

from __future__ import annotations

from typing import Any

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def create_source_citation_evaluator() -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        expected = _to_string_list((params.expected or {}).get("expected_sources", []))
        got = _to_string_list((params.output or {}).get("sources", []))

        expected_set = set(expected)
        got_set = set(got)

        if not expected_set:
            score = 1.0 if not got_set else 0.0
            label = "PASS" if score == 1.0 else "FAIL"
            return EvaluationResult(
                score=score,
                label=label,
                metadata={
                    "expected_sources": expected,
                    "retrieved_sources": got,
                },
            )

        overlap = len(expected_set & got_set)
        score = overlap / len(expected_set)
        label = "PASS" if score == 1.0 else "FAIL"

        return EvaluationResult(
            score=score,
            label=label,
            metadata={
                "expected_sources": expected,
                "retrieved_sources": got,
            },
        )

    return SimpleEvaluator(name="SourceCitation", kind="CODE", evaluate=evaluate)
