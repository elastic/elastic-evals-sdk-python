# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Agent Builder suite plugin."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from elastic_evals.runner.suites import EvaluationSuite


def _require_env(name: str) -> None:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Missing required env var "{name}".')


def _run_agent_builder() -> None:
    _require_env("CONNECTOR_ID")
    script_path = Path(__file__).resolve().parent / "run.py"
    if not script_path.is_file():
        raise RuntimeError("Agent Builder script not found.")
    subprocess.run([sys.executable, str(script_path)], check=True)


def get_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="agent-builder",
        description="Agent Builder evaluation suite (examples/agent_builder/run.py).",
        run=_run_agent_builder,
    )
