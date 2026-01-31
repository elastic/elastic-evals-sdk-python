"""CLI entrypoint for elastic-evals."""

from __future__ import annotations

import click

from elastic_evals.cli.commands import compare, doctor, env, list_cmd, run


@click.group()
@click.version_option(package_name="elastic-evals")
def main() -> None:
    """elastic-evals - Python SDK for running offline LLM evaluations."""


main.add_command(run.run_cmd)
main.add_command(list_cmd.list_cmd)
main.add_command(compare.compare_cmd)
main.add_command(doctor.doctor_cmd)
main.add_command(env.env_cmd)
