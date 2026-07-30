# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Agent Builder task implementation."""

from __future__ import annotations

from typing import Any

import httpx

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.types import Example


async def agent_builder_task(example: Example, config: ElasticEvalsConfig) -> dict[str, Any]:
    """Call Agent Builder API and return response."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.kibana_url}/api/agent_builder/converse",
            json={
                "connector_id": config.connector_id,
                "input": example.input.get("question"),
            },
            headers={
                "kbn-xsrf": "true",
                "elastic-api-version": "2023-10-31",
            },
        )
        response.raise_for_status()
        data = response.json()

    response_payload = data.get("response", {})
    message = response_payload.get("message")
    trace_id = data.get("trace_id") or data.get("traceId")
    if not trace_id:
        raise ValueError("Agent Builder response is missing a trace ID")
    return {
        "messages": [{"message": message}] if message is not None else [],
        "steps": data.get("steps", []),
        "traceId": trace_id,
        "_interaction_trace_id": trace_id,
        "conversation_id": data.get("conversation_id"),
    }
