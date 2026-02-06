"""Evaluation statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from elastic_evals.reporting.types import EvaluatorStats, StatsDisplay


def get_unique_evaluator_names(stats: Sequence[EvaluatorStats]) -> list[str]:
    return sorted({entry.evaluator_name for entry in stats})


def get_unique_datasets(stats: Sequence[EvaluatorStats]) -> list[dict[str, str]]:
    datasets = {(entry.dataset_id, entry.dataset_name) for entry in stats}
    return [
        {"id": dataset_id, "name": dataset_name}
        for dataset_id, dataset_name in sorted(datasets)
    ]


def calculate_overall_stats(stats: Sequence[EvaluatorStats]) -> list[EvaluatorStats]:
    overall_stats: list[EvaluatorStats] = []
    evaluator_names = get_unique_evaluator_names(stats)

    for evaluator_name in evaluator_names:
        evaluator_stats = [
            entry for entry in stats if entry.evaluator_name == evaluator_name
        ]
        total_count = sum(entry.stats.count for entry in evaluator_stats)

        weighted_mean = (
            sum(entry.stats.mean * entry.stats.count for entry in evaluator_stats)
            / total_count
            if total_count > 0
            else 0.0
        )

        if total_count <= 1:
            pooled_variance = 0.0
        else:
            pooled_variance = sum(
                (entry.stats.count - 1) * entry.stats.std_dev**2
                + entry.stats.count * (entry.stats.mean - weighted_mean) ** 2
                for entry in evaluator_stats
            ) / (total_count - 1)

        min_value = min((entry.stats.min for entry in evaluator_stats), default=0.0)
        max_value = max((entry.stats.max for entry in evaluator_stats), default=0.0)

        overall_stats.append(
            EvaluatorStats(
                dataset_id="overall",
                dataset_name="Overall",
                evaluator_name=evaluator_name,
                stats=StatsDisplay(
                    mean=weighted_mean,
                    median=weighted_mean,
                    std_dev=math.sqrt(pooled_variance),
                    min=min_value,
                    max=max_value,
                    count=total_count,
                ),
            )
        )

    return overall_stats
