"""Evaluation statistics helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from elastic_evals.reporting.types import DatasetScore, DatasetScoreWithStats, EvaluatorStats


def calculate_evaluator_stats(scores: list[float], total_examples: int) -> EvaluatorStats:
    if not scores:
        return EvaluatorStats(
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            min=0.0,
            max=0.0,
            count=0,
            percentage=0.0,
        )

    total_score = sum(scores)
    std_dev = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return EvaluatorStats(
        mean=float(np.mean(scores)),
        median=float(np.median(scores)),
        std_dev=std_dev,
        min=float(np.min(scores)),
        max=float(np.max(scores)),
        count=len(scores),
        percentage=total_score / total_examples if total_examples > 0 else 0.0,
    )


def get_unique_evaluator_names(dataset_scores: Sequence[DatasetScore]) -> list[str]:
    evaluator_names: set[str] = set()
    for dataset in dataset_scores:
        if dataset.evaluator_scores:
            evaluator_names.update(dataset.evaluator_scores.keys())
        elif hasattr(dataset, "evaluator_stats"):
            evaluator_stats = getattr(dataset, "evaluator_stats", None)
            if evaluator_stats:
                evaluator_names.update(evaluator_stats.keys())
    return sorted(evaluator_names)


def _pooled_variance(stats: list[EvaluatorStats], weighted_mean: float) -> float:
    total_count = sum(item.count for item in stats)
    if total_count <= 1:
        return 0.0
    numerator = sum(
        (item.count - 1) * item.std_dev**2 + item.count * (item.mean - weighted_mean) ** 2
        for item in stats
    )
    return numerator / (total_count - 1)


def calculate_overall_stats(
    dataset_scores: Sequence[DatasetScore | DatasetScoreWithStats],
) -> dict[str, EvaluatorStats]:
    overall_stats: dict[str, EvaluatorStats] = {}
    total_examples = sum(dataset.num_examples for dataset in dataset_scores)

    evaluator_names = get_unique_evaluator_names(dataset_scores)
    for evaluator_name in evaluator_names:
        all_scores: list[float] = []
        total_score = 0.0
        for dataset in dataset_scores:
            scores = dataset.evaluator_scores.get(evaluator_name, [])
            if scores:
                all_scores.extend(scores)
                total_score += sum(scores)

        if all_scores:
            std_dev = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else 0.0
            overall_stats[evaluator_name] = EvaluatorStats(
                mean=float(np.mean(all_scores)),
                median=float(np.median(all_scores)),
                std_dev=std_dev,
                min=float(np.min(all_scores)),
                max=float(np.max(all_scores)),
                count=len(all_scores),
                percentage=total_score / total_examples if total_examples > 0 else 0.0,
            )
            continue

        stats_by_dataset: list[EvaluatorStats] = []
        for dataset in dataset_scores:
            evaluator_stats = getattr(dataset, "evaluator_stats", None) or {}
            stats = evaluator_stats.get(evaluator_name)
            if stats and stats.count > 0:
                stats_by_dataset.append(stats)

        if not stats_by_dataset:
            overall_stats[evaluator_name] = EvaluatorStats(
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min=0.0,
                max=0.0,
                count=0,
                percentage=0.0,
            )
            continue

        total_count = sum(item.count for item in stats_by_dataset)
        weighted_mean = (
            sum(item.mean * item.count for item in stats_by_dataset) / total_count
            if total_count > 0
            else 0.0
        )
        pooled_variance = _pooled_variance(stats_by_dataset, weighted_mean)
        overall_stats[evaluator_name] = EvaluatorStats(
            mean=weighted_mean,
            median=weighted_mean,
            std_dev=math.sqrt(pooled_variance),
            min=min(item.min for item in stats_by_dataset),
            max=max(item.max for item in stats_by_dataset),
            count=total_count,
            percentage=(weighted_mean * total_count) / total_examples if total_examples > 0 else 0.0,
        )

    return overall_stats
