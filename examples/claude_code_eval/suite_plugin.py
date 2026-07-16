# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Claude Code eval suite plugin."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from elastic_evals.runner.suites import EvaluationSuite


def _run_claude_code_eval() -> None:
    script_path = Path(__file__).resolve().parent / "run.py"
    if not script_path.is_file():
        raise RuntimeError("Claude Code eval script not found.")
    subprocess.run([sys.executable, str(script_path)], check=True)


def get_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="claude-code-eval",
        description="Claude Code eval: latency and tool use metrics (examples/claude_code_eval/run.py).",
        run=_run_claude_code_eval,
    )
