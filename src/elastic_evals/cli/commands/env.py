"""List supported environment variables."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

ENV_VARS = [
    ("KIBANA_URL", "Kibana base URL", "http://localhost:5601"),
    ("CONNECTOR_ID", "Kibana connector ID for tasks", "abc-123"),
    ("EVALUATION_CONNECTOR_ID", "Connector ID for LLM evaluators", "def-456"),
    ("EVALUATIONS_ES_URL", "Elasticsearch URL for scores", "http://localhost:9200"),
    ("TRACE_ES_URL", "Elasticsearch URL for traces", "http://localhost:9200"),
    ("ELASTIC_EVALS_RUN_ID", "Override run ID", "my-custom-run-id"),
    ("ELASTIC_EVALS_REPETITIONS", "Number of repetitions", "3"),
    ("ELASTIC_EVALS_CONCURRENCY", "Concurrency level", "5"),
    ("ELASTIC_EVALS_LOG_LEVEL", "Log level (DEBUG, INFO, WARNING, ERROR)", "INFO"),
    (
        "ELASTIC_EVALS_MODEL",
        "JSON model metadata override",
        '{"id":"gpt-4","family":"openai","provider":"openai"}',
    ),
    ("ELASTIC_EVALS_TRACING_ENABLED", "Enable tracing (true/false)", "true"),
    (
        "ELASTIC_EVALS_TRACING_EXPORTER",
        "Tracing exporter (otlp, console, none)",
        "otlp",
    ),
    (
        "ELASTIC_EVALS_TRACING_ENDPOINT",
        "OTLP HTTP endpoint",
        "http://localhost:4318/v1/traces",
    ),
    ("ELASTIC_EVALS_TRACING_SERVICE_NAME", "Tracing service name", "elastic-evals"),
]


@click.command("env")
def env_cmd() -> None:
    """List supported environment variables."""
    console = Console()
    table = Table(show_header=True, header_style="bold")

    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Example")

    for name, description, example in ENV_VARS:
        table.add_row(name, description, example)

    console.print(table)
