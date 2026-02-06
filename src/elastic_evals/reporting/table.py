"""Terminal report table rendering."""

from __future__ import annotations

from collections.abc import Iterable

from rich.table import Table
from rich.text import Text

from elastic_evals.reporting.stats import (
    calculate_overall_stats,
    get_unique_datasets,
    get_unique_evaluator_names,
)
from elastic_evals.reporting.types import (
    EvaluatorDisplayGroup,
    EvaluatorDisplayOptions,
    EvaluatorStats,
    ReportDisplayOptions,
    StatsDisplay,
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

    column_names = [name for name in evaluator_names if name not in grouped] + list(
        group_mapping.keys()
    )
    return column_names, group_mapping


def _format_stat_value(value: float, decimal_places: int) -> str:
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.{decimal_places}f}"


def _format_stats_cell(
    stats: StatsDisplay,
    evaluator_name: str,
    is_overall: bool,
    evaluator_options: dict[str, EvaluatorDisplayOptions],
) -> Text:
    color = "bold green" if is_overall else "cyan"
    options = evaluator_options.get(evaluator_name)
    decimal_places = (
        options.decimal_places if options and options.decimal_places is not None else 2
    )
    unit_suffix = options.unit_suffix if options and options.unit_suffix else ""
    stats_to_include = options.stats_to_include if options else None

    stat_labels: list[tuple[str, str]] = [
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
        lines.append(
            Text(
                f"{label}: {_format_stat_value(float(value), decimal_places)}{unit_suffix}",
                style=color,
            )
        )

    return Text("\n").join(lines)


def _format_group_cell(
    evaluator_stats: dict[str, StatsDisplay],
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
        section.append(
            _format_stats_cell(stats, evaluator_name, is_overall, evaluator_options)
        )
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
    stats: list[EvaluatorStats],
    column_names: list[str],
    group_mapping: dict[str, EvaluatorDisplayGroup],
    is_overall: bool,
    evaluator_options: dict[str, EvaluatorDisplayOptions],
) -> list[Text]:
    stats_map = {entry.evaluator_name: entry.stats for entry in stats}
    cells: list[Text] = []
    for column_name in column_names:
        group = group_mapping.get(column_name)
        if group:
            cells.append(
                _format_group_cell(stats_map, group, is_overall, evaluator_options)
            )
            continue

        stat_entry = stats_map.get(column_name)
        if stat_entry and stat_entry.count > 0:
            cells.append(
                _format_stats_cell(
                    stat_entry, column_name, is_overall, evaluator_options
                )
            )
        else:
            cells.append(Text("-", style="bold green" if is_overall else "dim"))
    return cells


def create_report_table(
    stats: list[EvaluatorStats],
    repetitions: int,
    display_options: ReportDisplayOptions | None = None,
) -> Table:
    table = Table(show_header=True, header_style="bold")
    evaluator_options = (
        display_options.evaluator_display_options if display_options else {}
    )
    evaluator_display_groups = (
        display_options.evaluator_display_groups if display_options else []
    )
    evaluator_names = get_unique_evaluator_names(stats)
    column_names, group_mapping = _group_evaluator_names(
        evaluator_names, evaluator_display_groups
    )

    table.add_column("Dataset", style="white")
    table.add_column("#", justify="right")
    for name in column_names:
        table.add_column(name, justify="right")

    for dataset in get_unique_datasets(stats):
        dataset_stats = [entry for entry in stats if entry.dataset_id == dataset["id"]]
        dataset_count = max((entry.stats.count for entry in dataset_stats), default=0)
        if repetitions > 1:
            count_display = f"{repetitions} x {dataset_count // repetitions}"
        else:
            count_display = str(dataset_count)
        row = [
            Text(dataset["name"], style="white"),
            Text(count_display, style="white"),
            *_format_row_cells(
                dataset_stats,
                column_names,
                group_mapping,
                False,
                evaluator_options,
            ),
        ]
        table.add_row(*row)

    overall_stats = calculate_overall_stats(stats)
    overall_count = max((entry.stats.count for entry in overall_stats), default=0)
    if repetitions > 1:
        overall_count_display = f"{repetitions} x {overall_count // repetitions}"
    else:
        overall_count_display = str(overall_count)
    overall_row = [
        Text("Overall", style="bold green"),
        Text(overall_count_display, style="bold green"),
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
