# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Shared headers for Kibana Agent Builder public API calls."""

from __future__ import annotations

from elastic_evals._internal.agent_builder.constants import AGENT_BUILDER_API_VERSION
from elastic_evals.api.headers import build_kibana_headers


def build_agent_builder_headers(api_key: str | None) -> dict[str, str]:
    """Build Agent Builder public API headers with optional API-key auth."""
    return build_kibana_headers(api_key, api_version=AGENT_BUILDER_API_VERSION)
