"""Chatbot RAG example suite plugin."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from elastic_evals.runner.suites import EvaluationSuite


def _require_env(name: str) -> None:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Missing required env var "{name}".')


def _run_chatbot_rag_example() -> None:
    _require_env("CONNECTOR_ID")
    _require_env("EVALUATIONS_ES_URL")

    script_path = Path(__file__).resolve().parent / "run.py"
    if not script_path.is_file():
        raise RuntimeError("Chatbot RAG example script not found.")

    subprocess.run([sys.executable, str(script_path)], check=True)


def get_suite() -> EvaluationSuite:
    return EvaluationSuite(
        id="chatbot-rag-eval-example",
        description=(
            "Chatbot RAG app evaluation suite (examples/chatbot_rag_app/run.py)."
        ),
        run=_run_chatbot_rag_example,
    )
