# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Chatbot RAG task implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import httpx

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.tracing import propagated_headers
from elastic_evals.types import Example

CHATBOT_APP_URL = os.environ.get("CHATBOT_APP_URL", "http://localhost:4000")


async def chatbot_rag_task(example: Example, config: ElasticEvalsConfig) -> dict[str, Any]:
    """Call the chatbot RAG app chat endpoint and parse SSE response."""
    del config

    request_session_id = str(uuid4())
    answer_chunks: list[str] = []
    source_docs: list[dict[str, Any]] = []
    parsed_session_id: str | None = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{CHATBOT_APP_URL}/api/chat?session_id={request_session_id}",
            json={"question": example.input.get("question")},
            headers=propagated_headers(),
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                payload = line[6:]
                if payload == "[DONE]":
                    break

                if payload.startswith("[SESSION_ID] "):
                    parsed_session_id = payload[len("[SESSION_ID] ") :].strip() or None
                    continue

                if payload.startswith("[SOURCE] "):
                    source_payload = payload[len("[SOURCE] ") :]
                    source_doc = json.loads(source_payload)
                    if isinstance(source_doc, dict):
                        source_docs.append(source_doc)
                    continue

                answer_chunks.append(payload)

    return {
        "answer": "".join(answer_chunks),
        "sources": [
            source_name
            for source_name in (source.get("name") for source in source_docs)
            if isinstance(source_name, str)
        ],
        "source_docs": source_docs,
        "session_id": parsed_session_id,
    }
