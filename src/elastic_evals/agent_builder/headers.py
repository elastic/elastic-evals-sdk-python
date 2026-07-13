"""Shared headers for Kibana Agent Builder public API calls."""

from __future__ import annotations

import base64
import os

from elastic_evals.agent_builder.constants import AGENT_BUILDER_API_VERSION


def build_agent_builder_headers(api_key: str | None) -> dict[str, str]:
    """Build Agent Builder public API headers with optional API-key auth."""
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "elastic-api-version": AGENT_BUILDER_API_VERSION,
        "x-elastic-internal-origin": "true",
    }

    #headers = {
        #    "kbn-xsrf": "true",
        #    "Content-Type": "application/json",
        #    "x-elastic-internal-origin": "true",
        #    **propagated_headers(),
        #}

    if api_key:
        token = base64.b64encode(
            f"{os.environ['KIBANA_USERNAME']}:{os.environ['KIBANA_PASSWORD']}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {token}"
        # headers["Authorization"] = f"ApiKey {api_key}"

    return headers
