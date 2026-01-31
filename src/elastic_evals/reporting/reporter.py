"""Terminal reporting for evaluation runs."""

from __future__ import annotations

from rich.console import Console

from elastic_evals.export.documents import ModelInfo
from elastic_evals.reporting.stats import calculate_overall_stats, get_unique_evaluator_names
from elastic_evals.reporting.table import create_report_table
from elastic_evals.reporting.types import DatasetScoreWithStats, ReportDisplayOptions


def build_report_header(task_model: ModelInfo, evaluator_model: ModelInfo) -> list[str]:
    lines = [
        f"Model: {task_model.id} ({task_model.family}/{task_model.provider})",
        f"Evaluator Model: {evaluator_model.id} ({evaluator_model.family}/{evaluator_model.provider})",
    ]
    return lines


class DefaultReporter:
    def __init__(self, *, console: Console | None = None) -> None:
        self._console = console or Console()

    def report(
        self,
        dataset_scores: list[DatasetScoreWithStats],
        task_model: ModelInfo,
        evaluator_model: ModelInfo,
        display_options: ReportDisplayOptions | None = None,
    ) -> None:
        evaluator_names = get_unique_evaluator_names(dataset_scores)
        overall_stats = calculate_overall_stats(dataset_scores)
        table = create_report_table(dataset_scores, overall_stats, evaluator_names, display_options)

        header = build_report_header(task_model, evaluator_model)
        self._console.print()
        for line in header:
            self._console.print(line)
        self._console.print()
        self._console.print("═══ EVALUATION RESULTS ═══", style="bold blue")
        self._console.print(table)
