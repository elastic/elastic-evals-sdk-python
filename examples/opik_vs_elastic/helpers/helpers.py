# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared constants and data helpers for the Wix evaluation examples."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

ENV_PATH = Path(__file__).parent.parent / ".env"
INDEX_NAME = "wix_knowledge_base"
SEARCH_TOOL_ID = "wix-knowledge-search"
GROUND_TRUTH_COLUMN = "gt_customer_support_wix_knowledge_base"
WIX_QA_DATASET_PATH_ENV = "WIX_QA_DATASET_PATH"
WIX_KNOWLEDGE_BASE_PATH_ENV = "WIX_KNOWLEDGE_BASE_PATH"
AGENT_ID = "wix-eval-agent"
AGENT_NAME = "Wix Agent"
AGENT_INSTRUCTIONS = "Answer questions using the Wix knowledge base."
SEARCH_TOOL_DESCRIPTION = "Search the Wix knowledge base articles."


def get_dataset_path(variable: str) -> str:
    """Read a GCS dataset path from the environment. Only needed when USE_GCP is True."""
    value = os.getenv(variable)
    if not value:
        raise ValueError(f"Set {variable} in .env to load the Wix data from GCS, or run with USE_GCP = False.")
    return value


def _parse_relevant_doc_ids(value: Any) -> list[str]:
    if value is None:
        return []

    parsed = value
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Wix document ground truth must be a dictionary") from exc

    if not isinstance(parsed, dict):
        return []

    return [str(document_id) for document_id, relevant in parsed.items() if relevant]


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str)))


def _reference_id(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    document_id = value.get("id")
    return document_id if isinstance(document_id, str) else None


def _extract_retrieved_doc_ids(output: Any, *, tool_id: str) -> list[str]:
    if not isinstance(output, dict):
        return []

    retrieved: list[str] = []
    steps = output.get("steps")
    if not isinstance(steps, list):
        return retrieved

    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") != "tool_call" or step.get("tool_id") != tool_id:
            continue

        results = step.get("results")
        if not isinstance(results, list):
            continue

        for result in results:
            if not isinstance(result, dict):
                continue
            data = result.get("data")
            if not isinstance(data, dict):
                continue

            direct_id = _reference_id(data.get("reference"))
            if direct_id:
                retrieved.append(direct_id)

            resources = data.get("resources")
            if not isinstance(resources, list):
                continue
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                resource_id = _reference_id(resource.get("reference"))
                if resource_id:
                    retrieved.append(resource_id)

    return list(dict.fromkeys(retrieved))
