"""Run evaluation scripts."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator

import click  # type: ignore[import-not-found]

from elastic_evals.runner.suites import get_suite


def _format_env_prefix(overrides: dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in overrides.items())


def _validate_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise click.ClickException(f"{name} must be at least 1.")
    return value


def _apply_overrides(
    overrides: dict[str, str], env: dict[str, str], keys: Iterable[str]
) -> None:
    for key in keys:
        if key in overrides:
            env[key] = overrides[key]


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {}
    for key, value in overrides.items():
        original[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, original_value in original.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@click.command("run")
@click.argument(
    "script",
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--suite", help="Suite ID registered via elastic_evals.suites entry points."
)
@click.option("--run-id", help="Override run ID used by elastic-evals.")
@click.option("--repetitions", "-r", type=int, help="Number of repetitions.")
@click.option("--concurrency", "-c", type=int, help="Concurrency level.")
@click.option("--trace-es-url", help="Elasticsearch URL for traces.")
@click.option("--kibana-url", help="Kibana base URL.")
@click.option("--connector-id", help="Kibana connector ID for tasks.")
@click.option("--evaluation-connector-id", help="Connector ID for evaluator LLMs.")
@click.option("--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR).")
@click.option(
    "--tracing-exporter",
    default="otlp",
    show_default=True,
    help="Tracing exporter (otlp, console, none).",
)
@click.option(
    "--tracing-endpoint",
    help="Tracing endpoint (for otlp exporter).",
)
@click.option("--dry-run", is_flag=True, help="Print command without executing.")
def run_cmd(
    script: str | None,
    suite: str | None,
    run_id: str | None,
    repetitions: int | None,
    concurrency: int | None,
    trace_es_url: str | None,
    kibana_url: str | None,
    connector_id: str | None,
    evaluation_connector_id: str | None,
    log_level: str | None,
    tracing_exporter: str | None,
    tracing_endpoint: str | None,
    dry_run: bool,
) -> None:
    """Run an evaluation script or registered suite."""
    if bool(script) == bool(suite):
        raise click.ClickException("Provide either a script path or --suite.")
    _validate_positive_int(repetitions, "Repetitions")
    _validate_positive_int(concurrency, "Concurrency")

    overrides: dict[str, str] = {}
    if run_id:
        overrides["ELASTIC_EVALS_RUN_ID"] = run_id
    if repetitions is not None:
        overrides["ELASTIC_EVALS_REPETITIONS"] = str(repetitions)
    if concurrency is not None:
        overrides["ELASTIC_EVALS_CONCURRENCY"] = str(concurrency)
    if trace_es_url:
        overrides["TRACE_ES_URL"] = trace_es_url
    if kibana_url:
        overrides["KIBANA_URL"] = kibana_url
    if connector_id:
        overrides["CONNECTOR_ID"] = connector_id
    if evaluation_connector_id:
        overrides["EVALUATION_CONNECTOR_ID"] = evaluation_connector_id
    if log_level:
        overrides["ELASTIC_EVALS_LOG_LEVEL"] = log_level.upper()
    if tracing_exporter:
        overrides["ELASTIC_EVALS_TRACING_EXPORTER"] = tracing_exporter
    if tracing_endpoint:
        overrides["ELASTIC_EVALS_TRACING_ENDPOINT"] = tracing_endpoint

    prefix = _format_env_prefix(overrides)
    if suite:
        preview = f"{prefix} elastic-evals run --suite {suite}".strip()
        click.echo(f"Running suite: {preview}")
    else:
        preview = f"{prefix} {sys.executable} {script}".strip()
        click.echo(f"Running: {preview}")

    if dry_run:
        return

    if suite:
        suite_def = get_suite(suite)
        if not suite_def:
            raise click.ClickException(f'Unknown suite "{suite}".')
        try:
            with _temporary_env(overrides):
                suite_result = suite_def.run()
                if asyncio.iscoroutine(suite_result):
                    asyncio.run(suite_result)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc
        return

    assert script is not None
    command = [sys.executable, script]
    env = os.environ.copy()
    _apply_overrides(overrides, env, overrides.keys())
    process_result = subprocess.run(command, env=env)
    raise SystemExit(process_result.returncode)
