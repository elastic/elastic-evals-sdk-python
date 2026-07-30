# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared headers for Kibana API calls."""

from __future__ import annotations

from elastic_evals.api.constants import EVALS_API_VERSION


def build_kibana_headers(
    api_key: str | None,
    api_version: str = EVALS_API_VERSION,
) -> dict[str, str]:
    """Build common Kibana headers with optional API-key auth."""
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "x-elastic-internal-origin": "true",
        "Elastic-Api-Version": api_version,
    }

    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    return headers
