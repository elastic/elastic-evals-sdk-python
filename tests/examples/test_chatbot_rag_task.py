# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.types import EvaluatorParams, Example
from examples.chatbot_rag_app.evaluators.source_citation import (
    create_source_citation_evaluator,
)
from examples.chatbot_rag_app.tasks import chatbot_rag as chatbot_rag_module


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_chatbot_rag_task_parses_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    source_doc_one = {"name": "Work From Home Policy", "id": "doc-1"}
    source_doc_two = {"name": "Equipment Guide", "id": "doc-2"}
    lines = [
        "event: message",
        "data: [SESSION_ID] session-abc-123",
        f"data: [SOURCE] {json.dumps(source_doc_one)}",
        "data: The policy allows hybrid work. ",
        "data: Employees should keep a dedicated workspace. ",
        f"data: [SOURCE] {json.dumps(source_doc_two)}",
        "data: SOURCES: Work From Home Policy, Equipment Guide",
        "data: [DONE]",
        "data: ignored after done",
    ]

    def fake_stream(
        self: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> _FakeStreamResponse:
        del self, headers
        assert method == "POST"
        assert url.startswith(
            f"{chatbot_rag_module.CHATBOT_APP_URL}/api/chat?session_id="
        )
        assert json == {"question": "What is our working from home policy?"}
        return _FakeStreamResponse(lines)

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    result = await chatbot_rag_module.chatbot_rag_task(
        Example(input={"question": "What is our working from home policy?"}),
        ElasticEvalsConfig(
            connector_id="connector-id",
        ),
    )

    assert result["answer"] == (
        "The policy allows hybrid work. "
        "Employees should keep a dedicated workspace. "
        "SOURCES: Work From Home Policy, Equipment Guide"
    )
    assert result["sources"] == ["Work From Home Policy", "Equipment Guide"]
    assert result["source_docs"] == [source_doc_one, source_doc_two]
    assert result["session_id"] == "session-abc-123"


@pytest.mark.asyncio
async def test_source_citation_evaluator_behaviors() -> None:
    evaluator = create_source_citation_evaluator()

    partial_recall = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "What is our working from home policy?"},
            output={"sources": ["Work From Home Policy"]},
            expected={"expected_sources": ["Work From Home Policy", "Equipment Guide"]},
            metadata={"query_intent": "Factual"},
        )
    )
    assert partial_recall.score == 0.5
    assert partial_recall.label == "FAIL"
    assert partial_recall.metadata == {
        "expected_sources": ["Work From Home Policy", "Equipment Guide"],
        "retrieved_sources": ["Work From Home Policy"],
    }

    negative_pass = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "What's the NASA sales team?"},
            output={"sources": []},
            expected={"expected_sources": []},
            metadata={"query_intent": "Factual"},
        )
    )
    assert negative_pass.score == 1.0
    assert negative_pass.label == "PASS"

    negative_fail = await evaluator.evaluate(
        EvaluatorParams(
            input={"question": "What's the NASA sales team?"},
            output={"sources": ["Fabricated Source"]},
            expected={"expected_sources": []},
            metadata={"query_intent": "Factual"},
        )
    )
    assert negative_fail.score == 0.0
    assert negative_fail.label == "FAIL"
