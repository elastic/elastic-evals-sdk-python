"""Paired t-test comparisons for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.text import Text

from elastic_evals.export.documents import EvaluationScoreDocument

MAX_BETA_ITERATIONS = 100
BETA_EPSILON = 3e-7
BETA_TINY = 1e-30


@dataclass(frozen=True)
class PairedScore:
    dataset_id: str
    dataset_name: str
    evaluator_name: str
    score_a: float
    score_b: float


@dataclass(frozen=True)
class PairingResult:
    pairs: list[PairedScore]
    skipped_missing_pairs: int
    skipped_null_scores: int


@dataclass(frozen=True)
class PairedTTestResult:
    dataset_id: str
    dataset_name: str
    evaluator_name: str
    sample_size: int
    mean_a: float
    mean_b: float
    p_value: float | None


def _build_pair_key(score: EvaluationScoreDocument) -> str:
    return "|".join(
        [
            score.example.dataset.id,
            score.example.id,
            score.evaluator.name,
            str(score.task.repetition_index),
        ]
    )


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def pair_scores(
    scores_a: list[EvaluationScoreDocument],
    scores_b: list[EvaluationScoreDocument],
) -> PairingResult:
    reference_map: dict[str, EvaluationScoreDocument] = {}
    skipped_null_scores = 0

    for score in scores_b:
        if not _is_finite(score.evaluator.score):
            skipped_null_scores += 1
            continue
        reference_map[_build_pair_key(score)] = score

    pairs: list[PairedScore] = []
    skipped_missing_pairs = 0

    for score_a in scores_a:
        if not _is_finite(score_a.evaluator.score):
            skipped_null_scores += 1
            continue
        match = reference_map.get(_build_pair_key(score_a))
        if not match:
            skipped_missing_pairs += 1
            continue
        if not _is_finite(match.evaluator.score):
            skipped_null_scores += 1
            continue

        pairs.append(
            PairedScore(
                dataset_id=score_a.example.dataset.id,
                dataset_name=score_a.example.dataset.name,
                evaluator_name=score_a.evaluator.name,
                score_a=score_a.evaluator.score or 0.0,
                score_b=match.evaluator.score or 0.0,
            )
        )

    return PairingResult(
        pairs=pairs,
        skipped_missing_pairs=skipped_missing_pairs,
        skipped_null_scores=skipped_null_scores,
    )


def _t_statistic(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    if variance <= 0:
        return 0.0
    return mean_value / (math.sqrt(variance) / math.sqrt(len(values)))


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - (qab * x) / qap
    if abs(d) < BETA_TINY:
        d = BETA_TINY
    d = 1.0 / d
    h = d

    for m in range(1, MAX_BETA_ITERATIONS + 1):
        m2 = 2 * m
        aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < BETA_TINY:
            d = BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < BETA_TINY:
            c = BETA_TINY
        d = 1.0 / d
        h *= d * c

        aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < BETA_TINY:
            d = BETA_TINY
        c = 1.0 + aa / c
        if abs(c) < BETA_TINY:
            c = BETA_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < BETA_EPSILON:
            break

    return h


def _incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1 - x)
    )
    bt = math.exp(log_beta)

    if x < (a + 1) / (a + b + 2):
        return (bt * _beta_continued_fraction(a, b, x)) / a
    return 1.0 - (bt * _beta_continued_fraction(b, a, 1 - x)) / b


def _t_statistic_to_p_value(t_statistic: float, degrees_of_freedom: int) -> float:
    if not math.isfinite(t_statistic) or degrees_of_freedom <= 0:
        return 1.0
    t_value = abs(t_statistic)
    x = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    return _clamp_probability(_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5))


def compute_paired_t_test_results(
    scores_a: list[EvaluationScoreDocument],
    scores_b: list[EvaluationScoreDocument],
) -> list[PairedTTestResult]:
    pairing = pair_scores(scores_a, scores_b)

    groups: dict[tuple[str, str], list[PairedScore]] = defaultdict(list)
    for pair in pairing.pairs:
        groups[(pair.dataset_id, pair.evaluator_name)].append(pair)

    results: list[PairedTTestResult] = []
    for group in groups.values():
        scores_a_values = [pair.score_a for pair in group]
        scores_b_values = [pair.score_b for pair in group]
        differences = [a - b for a, b in zip(scores_a_values, scores_b_values)]

        p_value: float | None = None
        if len(differences) >= 2:
            t_stat = _t_statistic(differences)
            p_value = _t_statistic_to_p_value(t_stat, len(differences) - 1)

        results.append(
            PairedTTestResult(
                dataset_id=group[0].dataset_id,
                dataset_name=group[0].dataset_name,
                evaluator_name=group[0].evaluator_name,
                sample_size=len(group),
                mean_a=sum(scores_a_values) / len(scores_a_values),
                mean_b=sum(scores_b_values) / len(scores_b_values),
                p_value=p_value,
            )
        )

    return results


def compare_runs(
    scores_a: list[EvaluationScoreDocument],
    scores_b: list[EvaluationScoreDocument],
) -> list[PairedTTestResult]:
    return compute_paired_t_test_results(scores_a, scores_b)


def _format_number(value: float) -> str:
    return f"{value:.2f}" if math.isfinite(value) else "-"


def _format_p_value(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.2f}"


def _format_difference(value: float) -> Text:
    if not math.isfinite(value):
        return Text("-", style="dim")
    if value > 0:
        return Text(f"+{value:.2f}", style="bold green")
    if value < 0:
        return Text(f"{value:.2f}", style="bold red")
    return Text(f"{value:.2f}")


def print_comparison_report(
    *,
    results: list[PairedTTestResult],
    run_id_a: str,
    run_id_b: str,
    significance_threshold: float = 0.05,
    console: Console | None = None,
) -> None:
    console = console or Console()

    sorted_results = sorted(
        results, key=lambda result: (result.dataset_name, result.evaluator_name)
    )
    significant_count = sum(
        1
        for result in sorted_results
        if result.p_value is not None and result.p_value < significance_threshold
    )

    console.print(f"Run A: {run_id_a}")
    console.print(f"Run B: {run_id_b}")
    console.print(f"Significance threshold: p < {significance_threshold}")
    console.print(f"Significant differences: {significant_count}/{len(sorted_results)}")
    console.print()

    grouped: dict[str, list[PairedTTestResult]] = defaultdict(list)
    for result in sorted_results:
        grouped[result.dataset_name].append(result)

    for dataset_name, dataset_results in grouped.items():
        console.print(Text(dataset_name, style="bold"))
        table = Table(show_header=True, header_style="bold")
        table.add_column("Evaluator")
        table.add_column("N", justify="right")
        table.add_column("Mean A", justify="right")
        table.add_column("Mean B", justify="right")
        table.add_column("Diff", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Significant", justify="right")

        for result in dataset_results:
            delta = result.mean_a - result.mean_b
            if result.p_value is None:
                significant_label = Text("n/a", style="dim")
            elif result.p_value < significance_threshold:
                significant_label = Text("yes", style="bold green")
            else:
                significant_label = Text("no", style="dim")

            table.add_row(
                result.evaluator_name,
                str(result.sample_size),
                _format_number(result.mean_a),
                _format_number(result.mean_b),
                _format_difference(delta),
                _format_p_value(result.p_value),
                significant_label,
            )

        console.print(table)
        console.print()
