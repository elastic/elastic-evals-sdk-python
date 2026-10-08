# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""CLI entrypoint for elastic-evals."""

from __future__ import annotations

import sys

try:
    import click  # type: ignore[import-not-found]
except ImportError as exc:
    print(
        "elastic-evals CLI requires the [runner] extra. Install with: pip install 'elastic-evals[runner]'",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from elastic_evals.runner.cli.commands import list_cmd, run


@click.group()
@click.version_option(package_name="elastic-evals")
def main() -> None:
    """elastic-evals - Python SDK for running offline LLM evaluations."""


main.add_command(run.run_cmd)
main.add_command(list_cmd.list_cmd)
