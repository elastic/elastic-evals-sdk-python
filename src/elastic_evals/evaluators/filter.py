"""Evaluator selection helpers."""

from __future__ import annotations

import os
from typing import Iterable, TypeVar

from elastic_evals.types import Evaluator

T = TypeVar("T")


def parse_selected_evaluators() -> list[str]:
    selection = os.environ.get("SELECTED_EVALUATORS")
    if not selection:
        return []
    return [item.strip() for item in selection.split(",") if item.strip()]


def select_evaluators(evaluators: Iterable[Evaluator]) -> list[Evaluator]:
    selected = parse_selected_evaluators()
    evaluators_list = list(evaluators)
    if not selected:
        return evaluators_list
    return [evaluator for evaluator in evaluators_list if evaluator.name in selected]
