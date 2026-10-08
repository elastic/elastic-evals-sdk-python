# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_cli_run_script_opts_the_child_process_into_logging(tmp_path: Path) -> None:
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    script = tmp_path / "probe.py"
    script.write_text(
        "import logging\n"
        "import helper  # a module next to the script, as under plain `python script.py`\n"
        "from rich.logging import RichHandler\n"
        "sdk = logging.getLogger('elastic_evals')\n"
        "ok = sdk.level == logging.DEBUG and any(isinstance(h, RichHandler) for h in sdk.handlers)\n"
        "raise SystemExit(0 if ok else 3)\n"
    )

    result = CliRunner().invoke(main, ["run", str(script), "--log-level", "DEBUG"])

    assert result.exit_code == 0, result.output


def test_cli_run_suite_applies_the_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    levels: list[str] = []
    monkeypatch.setattr("elastic_evals.runner.cli.commands.run.setup_logging", lambda level: levels.append(level))
    monkeypatch.setattr(
        "elastic_evals.runner.cli.commands.run.get_suite", lambda _name: SimpleNamespace(run=lambda: None)
    )

    result = CliRunner().invoke(main, ["run", "--suite", "fake", "--log-level", "debug"])

    assert result.exit_code == 0, result.output
    assert levels == ["DEBUG"]
