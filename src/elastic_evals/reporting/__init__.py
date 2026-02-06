"""Reporting helpers for elastic-evals."""

from .compare import (
    PairedTTestResult,
    PairingResult,
    compare_runs,
    compute_paired_t_test_results,
    pair_scores,
    print_comparison_report,
)
from .reporter import DefaultReporter, build_report_header
from .stats import (
    calculate_overall_stats,
    get_unique_datasets,
    get_unique_evaluator_names,
)
from .table import create_report_table
from .types import (
    EvaluationReport,
    EvaluatorDisplayGroup,
    EvaluatorDisplayOptions,
    EvaluatorStats,
    ReportDisplayOptions,
    RunStats,
    StatsDisplay,
)

__all__ = [
    "DefaultReporter",
    "EvaluationReport",
    "EvaluatorDisplayGroup",
    "EvaluatorDisplayOptions",
    "EvaluatorStats",
    "ReportDisplayOptions",
    "RunStats",
    "StatsDisplay",
    "PairedTTestResult",
    "PairingResult",
    "build_report_header",
    "calculate_overall_stats",
    "get_unique_datasets",
    "compare_runs",
    "compute_paired_t_test_results",
    "create_report_table",
    "get_unique_evaluator_names",
    "pair_scores",
    "print_comparison_report",
]
