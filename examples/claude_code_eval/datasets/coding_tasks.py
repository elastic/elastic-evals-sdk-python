"""Coding tasks dataset for Claude Code eval."""

from __future__ import annotations

from typing import Any

from elastic_evals.types import EvaluationDataset, Example

CodingTaskExample = Example[dict[str, str], None, dict[str, Any]]

coding_tasks_dataset: EvaluationDataset[CodingTaskExample] = EvaluationDataset(
    name="claude-code-eval: coding-tasks",
    description="Tasks that exercise Claude Code's code generation and tool use capabilities.",
    examples=[
        Example(
            input={
                "prompt": (
                    "Write a Python function called `validate_email` that checks whether a "
                    "string is a valid email address using only the standard library. "
                    "Use type hints and return True if valid, False otherwise."
                )
            },
            output=None,
            metadata={
                "expected_tools": [],
                "task_type": "code_generation",
                "criteria": [
                    "The response includes a Python function named validate_email",
                    "The function uses type hints (str -> bool or similar)",
                    "The function uses only standard library modules",
                    "The function handles basic valid and invalid email formats",
                ],
            },
        ),
        Example(
            input={
                "prompt": (
                    "List the top-level Python files in the current working directory "
                    "and briefly describe the purpose of each one based on its content."
                )
            },
            output=None,
            metadata={
                "expected_tools": ["bash", "read_file"],
                "task_type": "file_exploration",
                "criteria": [
                    "The response lists Python files found in the current directory",
                    "The response describes the purpose of at least one file",
                    "The response reads or inspects file contents rather than guessing",
                ],
            },
        ),
        Example(
            input={
                "prompt": (
                    "Read the pyproject.toml in the current directory and output a "
                    "Markdown table with two columns — Dependency and Min Version — "
                    "for all entries under [project.dependencies]."
                )
            },
            output=None,
            metadata={
                "expected_tools": ["read_file"],
                "task_type": "file_reading",
                "criteria": [
                    "The response contains a Markdown table",
                    "The table has columns for Dependency and Min Version",
                    "The table includes pydantic, opentelemetry entries",
                    "The data is read from the actual file, not fabricated",
                ],
            },
        ),
    ],
)
