"""Shared headers for Kibana internal evals API calls."""

from __future__ import annotations

from elastic_evals.api.constants import EVALS_API_VERSION


def build_kibana_headers(api_key: str | None) -> dict[str, str]:
    """Build common Kibana evals headers with optional API-key auth."""
    headers = {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "x-elastic-internal-origin": "true",
        "Elastic-Api-Version": EVALS_API_VERSION,
    }

    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    return headers
