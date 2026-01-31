"""Terminal report table rendering."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table
from rich.text import Text

from elastic_evals.reporting.types import (
    DatasetScoreWithStats,
    EvaluatorDisplayGroup,
    EvaluatorDisplayOptions,
    EvaluatorStats,
    ReportDisplayOptions,
)


def _group_evaluator_names(
    evaluator_names: list[str],
    evaluator_display_groups: Iterable[EvaluatorDisplayGroup],
) -> tuple[list[str], dict[str, EvaluatorDisplayGroup]]:
    group_mapping: dict[str, EvaluatorDisplayGroup] = {}
    grouped: set[str] = set()

    for group in evaluator_display_groups:
        if all(name in evaluator_names for name in group.evaluator_names):
            group_mapping[group.combined_column_name] = group
            grouped.update(group.evaluator_names)

    column_names = [name for name in evaluator_names if name not in grouped] + list(group_mapping.keys())
    return column_names, group_mapping


def _format_stat_value(value: float, decimal_places: int) -> str:
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.{decimal_places}f}"


def _format_stats_cell(
    stats: EvaluatorStats,
    evaluator_name: str,
    is_overall: bool,
    evaluator_options: dict[str, EvaluatorDisplayOptions],
) -> Text:
    color = "bold green" if is_overall else "cyan"
    percentage_color = "bold yellow"
    options = evaluator_options.get(evaluator_name)
    decimal_places = options.decimal_places if options and options.decimal_places is not None else 2
    unit_suffix = options.unit_suffix if options and options.unit_suffix else ""
    stats_to_include = options.stats_to_include if options else None

    stat_labels: list[tuple[str, str]] = [
        ("percentage", "percentage"),
        ("mean", "mean"),
        ("median", "median"),
        ("std_dev", "std"),
        ("min", "min"),
        ("max", "max"),
    ]

    lines: list[Text] = []
    for key, label in stat_labels:
        if stats_to_include and key not in stats_to_include:
            continue
        value = getattr(stats, key)
        if key == "percentage":
            lines.append(Text(f"{value * 100:.1f}%", style=percentage_color))
            continue
        lines.append(
            Text(
                f"{label}: {_format_stat_value(float(value), decimal_places)}{unit_suffix}",
                style=color,
            )
        )

    return Text("\n").join(lines)


def _format_group_cell(
    evaluator_stats: dict[str, EvaluatorStats],
    group: EvaluatorDisplayGroup,
    is_overall: bool,
    evaluator_options: dict[str, EvaluatorDisplayOptions],
) -> Text:
    sections: list[Text] = []
    separator = Text("────────────────", style="dim")

    for evaluator_name in group.evaluator_names:
        stats = evaluator_stats.get(evaluator_name)
        if not stats or stats.count == 0:
            continue
        section = Text(evaluator_name, style="white")
        section.append("\n")
        section.append(_format_stats_cell(stats, evaluator_name, is_overall, evaluator_options))
        sections.append(section)

    if not sections:
        return Text("-", style="dim")

    combined = Text("")
    for index, section in enumerate(sections):
        if index:
            combined.append("\n")
            combined.append(separator)
            combined.append("\n")
        combined.append(section)
    return combined


def _format_row_cells(
    evaluator_stats: dict[str, EvaluatorStats],
    column_names: list[str],
    group_mapping: dict[str, EvaluatorDisplayGroup],
    is_overall: bool,
    evaluator_options: dict[str, EvaluatorDisplayOptions],
) -> list[Text]:
    cells: list[Text] = []
    for column_name in column_names:
        group = group_mapping.get(column_name)
        if group:
            cells.append(_format_group_cell(evaluator_stats, group, is_overall, evaluator_options))
            continue

        stats = evaluator_stats.get(column_name)
        if stats and stats.count > 0:
            cells.append(_format_stats_cell(stats, column_name, is_overall, evaluator_options))
        else:
            cells.append(Text("-", style="bold green" if is_overall else "dim"))
    return cells


def create_report_table(
    dataset_scores: list[DatasetScoreWithStats],
    overall_stats: dict[str, EvaluatorStats],
    evaluator_names: list[str],
    display_options: ReportDisplayOptions | None = None,
) -> Table:
    table = Table(show_header=True, header_style="bold")
    evaluator_options = (
        display_options.evaluator_display_options if display_options else {}
    )
    evaluator_display_groups = (
        display_options.evaluator_display_groups if display_options else []
    )
    column_names, group_mapping = _group_evaluator_names(evaluator_names, evaluator_display_groups)

    table.add_column("Dataset", style="white")
    table.add_column("#", justify="right")
    for name in column_names:
        table.add_column(name, justify="right")

    for dataset in dataset_scores:
        row = [
            Text(dataset.name, style="white"),
            Text(str(dataset.num_examples), style="white"),
            *_format_row_cells(
                dataset.evaluator_stats,
                column_names,
                group_mapping,
                False,
                evaluator_options,
            ),
        ]
        table.add_row(*row)

    total_examples = sum(dataset.num_examples for dataset in dataset_scores)
    overall_row = [
        Text("Overall", style="bold green"),
        Text(str(total_examples), style="bold green"),
        *_format_row_cells(
            overall_stats,
            column_names,
            group_mapping,
            True,
            evaluator_options,
        ),
    ]
    table.add_row(*overall_row)

    return table
