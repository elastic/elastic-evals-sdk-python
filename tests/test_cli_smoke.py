# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

pytest.importorskip("click")

from click.testing import CliRunner  # noqa: E402

from elastic_evals.runner.cli.main import main  # noqa: E402


def test_cli_help_shows_run_and_list() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "list" in result.output


def test_cli_list_smoke() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["list"])

    assert result.exit_code == 0
