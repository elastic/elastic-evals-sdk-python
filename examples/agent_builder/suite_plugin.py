"""Agent Builder suite plugin."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from elastic_evals.suites import EvaluationSuite


def _require_env(name: str) -> None:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Missing required env var "{name}".')


def _run_agent_builder() -> None:
    _require_env("CONNECTOR_ID")
    _require_env("KIBANA_AUTH")
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
