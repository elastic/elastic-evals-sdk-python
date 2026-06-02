"""Latency evaluator for Claude Code eval."""

from __future__ import annotations

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

# (upper bound ms, score, label)
_THRESHOLDS: list[tuple[int, float, str]] = [
    (5_000, 1.0, "FAST"),
    (15_000, 0.8, "GOOD"),
    (30_000, 0.6, "OK"),
    (60_000, 0.4, "SLOW"),
]


def create_latency_evaluator() -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        output = params.output or {}
        latency_ms: float = output.get("latency_ms", float("inf"))

        score = 0.2
        label = "VERY_SLOW"
        for threshold, s, lbl in _THRESHOLDS:
            if latency_ms <= threshold:
                score = s
                label = lbl
                break

        return EvaluationResult(
            score=score,
            label=label,
            metadata={
                "latency_ms": latency_ms,
                "claude_duration_ms": output.get("claude_duration_ms"),
                "num_turns": output.get("num_turns"),
            },
        )

    return SimpleEvaluator(name="Latency", kind="CODE", evaluate=evaluate)
