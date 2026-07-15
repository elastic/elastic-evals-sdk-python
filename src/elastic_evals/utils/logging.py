# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Logging helpers for elastic-evals."""

from __future__ import annotations

import logging
from urllib.parse import urlparse, urlunparse

from rich.console import Console
from rich.logging import RichHandler

_CONSOLE = Console()
_LOGGER_NAME = "elastic_evals"


def setup_logging(level: str = "INFO") -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=_CONSOLE, rich_tracebacks=True)],
        )
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    return logger


log = setup_logging()


def log_experiment_start(run_id: str, dataset_name: str, evaluator_count: int, concurrency: int) -> None:
    log.info(
        '🧪 Starting experiment "Run ID: %s - Dataset: %s" with %s evaluators and %s concurrent runs',
        run_id,
        dataset_name,
        evaluator_count,
        concurrency,
    )


def log_task_execution(dataset_id: str, example_index: int, repetition: int) -> None:
    log.info(
        '🔧 Running task "task" on dataset "%s" (exampleIndex=%s, repetition=%s)',
        dataset_id,
        example_index,
        repetition,
    )


def log_evaluation_start(example_index: int, repetition: int, evaluator_count: int) -> None:
    log.info(
        "🧠 Evaluating run (exampleIndex=%s, repetition=%s) with %s evaluators",
        example_index,
        repetition,
        evaluator_count,
    )


def log_evaluator_start(evaluator_name: str, example_index: int, repetition: int) -> None:
    log.info(
        '🧠 Evaluating run (exampleIndex=%s, repetition=%s) with evaluator "%s"',
        example_index,
        repetition,
        evaluator_name,
    )


def log_evaluator_complete(evaluator_name: str, example_index: int, repetition: int) -> None:
    log.info(
        '✅ Evaluator "%s" on run (exampleIndex=%s, repetition=%s) completed',
        evaluator_name,
        example_index,
        repetition,
    )


def log_experiment_complete(experiment_id: str) -> None:
    log.info("✅ Experiment %s completed", experiment_id)


def _strip_auth_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        netloc = f"{host}:{parsed.port}" if parsed.port else host
        return urlunparse(parsed._replace(netloc=netloc))
    return url


def log_results_url(kibana_url: str, run_id: str) -> None:
    base = _strip_auth_from_url(kibana_url.rstrip("/"))
    results_url = f"{base}/app/management/ai/evals/runs/{run_id}"
    log.info("📊 View results: %s", results_url)


def log_export_header() -> None:
    _CONSOLE.print("═══ EXPORTING TO ELASTICSEARCH ═══", style="bold blue")


def log_export_success() -> None:
    log.info("✅ Evaluation scores exported successfully!")


def log_export_query_hint(hostname: str, model_id: str | None, run_id: str) -> None:
    model_filter = f'task.model.id:"{model_id}"' if model_id else "task.model.id:*"
    log.info(
        'You can query the data using: environment.hostname:"%s" AND %s AND run_id:"%s"',
        hostname,
        model_filter,
        run_id,
    )


def log_no_scores_warning() -> None:
    log.warning("No evaluation scores to export")


def log_index_template_created() -> None:
    log.debug("Created Elasticsearch index template for evaluation scores")


def log_scores_indexed(count: int) -> None:
    log.debug("Successfully indexed %s evaluation scores", count)


def log_bulk_error(failed: int, total: int) -> None:
    log.error("Bulk indexing had %s failed operations out of %s", failed, total)
