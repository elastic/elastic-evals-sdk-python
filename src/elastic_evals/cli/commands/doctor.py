"""Doctor command for checking prerequisites."""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import click


def _check_socket(host: str, port: int, timeout: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _parse_host_port(url: str) -> tuple[str, int | None]:
    parsed = urlparse(url)
    if parsed.hostname is None:
        return url, None
    return parsed.hostname, parsed.port


@click.command("doctor")
def doctor_cmd() -> None:
    """Check prerequisites for running evaluations."""
    issues: list[str] = []
    warnings: list[str] = []

    if not os.environ.get("CONNECTOR_ID"):
        issues.append("CONNECTOR_ID environment variable not set.")

    if not os.environ.get("EVALUATION_CONNECTOR_ID"):
        warnings.append(
            "EVALUATION_CONNECTOR_ID not set (required for LLM evaluators)."
        )

    if not os.environ.get("EVALUATIONS_ES_URL"):
        warnings.append("EVALUATIONS_ES_URL not set - ES export will be disabled.")

    if not os.environ.get("TRACE_ES_URL"):
        warnings.append("TRACE_ES_URL not set - trace-based evaluators will fail.")

    kibana_url = os.environ.get("KIBANA_URL")
    if not kibana_url:
        issues.append("KIBANA_URL environment variable not set.")
        return

    host, port = _parse_host_port(kibana_url)
    if port is None:
        warnings.append(f"Could not parse Kibana port from {kibana_url}.")
    else:
        try:
            if not _check_socket(host, port):
                warnings.append(f"Cannot connect to Kibana at {kibana_url}.")
        except OSError as exc:
            warnings.append(f"Could not check Kibana connectivity: {exc}.")

    if issues:
        click.echo(click.style("Issues:", fg="red", bold=True))
        for issue in issues:
            click.echo(click.style(f"  - {issue}", fg="red"))

    if warnings:
        click.echo(click.style("\nWarnings:", fg="yellow", bold=True))
        for warning in warnings:
            click.echo(click.style(f"  - {warning}", fg="yellow"))

    if not issues and not warnings:
        click.echo(click.style("All checks passed.", fg="green", bold=True))
    elif not issues:
        click.echo(click.style("\nNo blocking issues found.", fg="green"))
