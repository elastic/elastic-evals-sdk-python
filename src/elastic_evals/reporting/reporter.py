"""Terminal reporting for evaluation runs."""

from __future__ import annotations

from rich.console import Console

from elastic_evals.export.documents import ModelInfo
from elastic_evals.reporting.table import create_report_table
from elastic_evals.reporting.types import EvaluatorStats, ReportDisplayOptions


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
        stats: list[EvaluatorStats],
        repetitions: int,
        task_model: ModelInfo,
        evaluator_model: ModelInfo,
        display_options: ReportDisplayOptions | None = None,
    ) -> None:
        table = create_report_table(stats, repetitions, display_options)

        header = build_report_header(task_model, evaluator_model)
        self._console.print()
        for line in header:
            self._console.print(line)
        self._console.print()
        self._console.print("═══ EVALUATION RESULTS ═══", style="bold blue")
        self._console.print(table)
