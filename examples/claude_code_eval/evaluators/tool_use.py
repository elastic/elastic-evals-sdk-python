"""Tool use evaluator for Claude Code eval."""

from __future__ import annotations

from typing import Any

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def create_tool_use_evaluator() -> Evaluator:
    """Score whether Claude Code used the expected tools.

    - expected_tools empty  → PASS (1.0) when no tools were used; FAIL otherwise
    - expected_tools set    → partial credit: overlap / len(expected)
    """

    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        output = params.output or {}
        metadata = params.metadata or {}

        expected_tools = _to_string_list(metadata.get("expected_tools", []))
        actual_tools = _to_string_list(output.get("tool_calls", []))
        actual_set = set(actual_tools)

        if not expected_tools:
            score = 1.0 if not actual_set else 0.0
            label = "PASS" if score == 1.0 else "FAIL"
            return EvaluationResult(
                score=score,
                label=label,
                metadata={
                    "expected_tools": expected_tools,
                    "actual_tools": actual_tools,
                    "tool_call_count": len(actual_tools),
                },
            )

        expected_set = set(expected_tools)
        overlap = len(expected_set & actual_set)
        score = overlap / len(expected_set)
        if score >= 1.0:
            label = "PASS"
        elif score > 0:
            label = "PARTIAL"
        else:
            label = "FAIL"

        return EvaluationResult(
            score=score,
            label=label,
            metadata={
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_call_count": len(actual_tools),
                "matched_tools": sorted(expected_set & actual_set),
                "missing_tools": sorted(expected_set - actual_set),
            },
        )

    return SimpleEvaluator(name="ToolUse", kind="CODE", evaluate=evaluate)
