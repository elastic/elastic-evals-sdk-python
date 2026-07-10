# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

"""List available evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import click  # type: ignore[import-not-found]
from rich.console import Console
from rich.table import Table

from elastic_evals.runner.suites import discover_suites


def _find_examples_root() -> Path | None:
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "examples").is_dir() and (parent / "pyproject.toml").is_file():
            return parent / "examples"
    return None


def _collect_run_scripts(examples_root: Path) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for path in sorted(examples_root.rglob("run.py")):
        if not path.is_file():
            continue
        relative = path.relative_to(examples_root)
        suite_id = (
            relative.parent.as_posix()
            if relative.name == "run.py"
            else relative.as_posix()
        )
        runs.append((suite_id, str(path)))
    return runs


@click.command("list")
def list_cmd() -> None:
    """List available evaluation scripts."""
    console = Console()
    discovery = discover_suites()

    if discovery.errors:
        click.echo("Suite discovery errors:")
        for error in discovery.errors:
            click.echo(f"  - {error}")

    if discovery.suites:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Suite")
        table.add_column("Description")
        for suite in discovery.suites:
            table.add_row(suite.id, suite.description)
        console.print(table)

    examples_root = _find_examples_root()
    if not examples_root:
        if not discovery.suites:
            raise click.ClickException(
                "Could not locate examples folder or entry-point suites."
            )
        return

    runs = _collect_run_scripts(examples_root)
    if not runs:
        if not discovery.suites:
            click.echo("No evaluation scripts found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Evaluation (examples)")
    table.add_column("Path")

    for suite_id, path in runs:
        table.add_row(suite_id, path)

    console.print(table)
