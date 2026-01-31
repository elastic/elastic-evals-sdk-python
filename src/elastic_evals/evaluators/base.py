"""Shared evaluator helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from elastic_evals.types import EvaluationResult, EvaluatorParams


@dataclass
class SimpleEvaluator:
    name: str
    kind: Literal["LLM", "CODE"]
    _evaluate: Callable[[EvaluatorParams], Awaitable[EvaluationResult]]

    def __init__(
        self,
        name: str,
        kind: Literal["LLM", "CODE"],
        evaluate: Callable[[EvaluatorParams], Awaitable[EvaluationResult]],
    ) -> None:
        self.name = name
        self.kind = kind
        self._evaluate = evaluate

    async def evaluate(self, params: EvaluatorParams) -> EvaluationResult:
        return await self._evaluate(params)
