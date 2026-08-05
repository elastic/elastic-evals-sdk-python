# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tenacity import wait_none

from elastic_evals.api.errors import KibanaAPIError
from elastic_evals.integrations.agent_builder import (
    AgentBuilderClient,
    AgentBuilderError,
    AgentConfiguration,
    ConverseResponse,
    CreateAgentRequest,
    CreateToolRequest,
    IndexSearchToolConfig,
    ToolSelection,
    UpdateAgentRequest,
    UpdateToolRequest,
)


class _RecordingAsyncClient:
    responses: list[httpx.Response | BaseException] = []
    requests: list[dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        return self._record("GET", url, headers=headers)

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        return self._record("POST", url, json=json, headers=headers)

    async def put(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        return self._record("PUT", url, json=json, headers=headers)

    def _record(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"method": method, "url": url, "timeout": self.timeout, **kwargs})
        outcome = self.responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @classmethod
    def configure(cls, outcomes: Sequence[httpx.Response | BaseException]) -> None:
        cls.responses = list(outcomes)
        cls.requests = []


@pytest.fixture(autouse=True)
def _mock_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "elastic_evals.integrations.agent_builder.client.httpx.AsyncClient",
        _RecordingAsyncClient,
    )
    monkeypatch.setattr(AgentBuilderClient._create_tool.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient._create_agent.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient.get_tool.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient.get_agent.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient.update_tool.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient.update_agent.retry, "wait", wait_none())
    monkeypatch.setattr(AgentBuilderClient.call_converse.retry, "wait", wait_none())


def _tool_request() -> CreateToolRequest:
    return CreateToolRequest(
        id="search-documents",
        type="index_search",
        description="Search documents",
        tags=["evaluation"],
        configuration=IndexSearchToolConfig(pattern="documents-*", row_limit=10),
    )


def _agent_request() -> CreateAgentRequest:
    return CreateAgentRequest(
        id="document-agent",
        name="Document agent",
        description="Answers questions about documents",
        configuration=AgentConfiguration(
            instructions="Use the search tool.",
            tools=[ToolSelection(tool_ids=["search-documents"])],
        ),
        labels=["evaluation"],
    )


def test_agent_builder_error_is_kibana_api_error() -> None:
    assert isinstance(AgentBuilderError("request failed"), KibanaAPIError)


@pytest.mark.asyncio
async def test_create_tool_creates_missing_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(404, json={"message": "not found"}),
            httpx.Response(
                201,
                json={
                    "id": "search-documents",
                    "type": "index_search",
                    "description": "Search documents",
                    "readonly": False,
                    "tags": ["evaluation"],
                    "configuration": {
                        "pattern": "documents-*",
                        "row_limit": 10,
                    },
                    "experimental": False,
                    "schema": {"type": "object"},
                },
            ),
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_tool(_tool_request())

    assert result.id == "search-documents"
    assert result.configuration == {"pattern": "documents-*", "row_limit": 10}
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "POST",
    ]
    assert "Created Agent Builder tool 'search-documents'" in caplog.text


@pytest.mark.asyncio
async def test_create_tool_reuses_existing_tool_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={"id": "search-documents", "type": "index_search"},
            )
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_tool(_tool_request())

    assert result.id == "search-documents"
    assert len(_RecordingAsyncClient.requests) == 1
    assert "already exists; using it" in caplog.text


@pytest.mark.asyncio
async def test_create_tool_updates_existing_tool_when_requested() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={"id": "search-documents", "type": "index_search"},
            ),
            httpx.Response(
                200,
                json={
                    "id": "search-documents",
                    "type": "index_search",
                    "description": "Search documents",
                },
            ),
        ]
    )

    await AgentBuilderClient("http://kibana:5601").create_tool(
        _tool_request(),
        update_if_exists=True,
    )

    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "PUT",
    ]
    assert _RecordingAsyncClient.requests[1]["url"] == ("http://kibana:5601/api/agent_builder/tools/search-documents")
    assert _RecordingAsyncClient.requests[1]["json"] == {
        "description": "Search documents",
        "tags": ["evaluation"],
        "configuration": {
            "pattern": "documents-*",
            "row_limit": 10,
        },
    }


@pytest.mark.asyncio
async def test_create_tool_rejects_type_change() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={"id": "search-documents", "type": "esql"},
            )
        ]
    )

    with pytest.raises(ValueError, match="from type 'esql' to 'index_search'"):
        await AgentBuilderClient("http://kibana:5601").create_tool(
            _tool_request(),
            update_if_exists=True,
        )

    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_create_agent_updates_existing_agent_and_returns_metadata() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "id": "document-agent",
                    "type": "chat",
                    "name": "Old name",
                    "readonly": False,
                    "configuration": {"tools": []},
                    "permissions": {
                        "update_agent": True,
                        "update_access_control": True,
                    },
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "document-agent",
                    "type": "chat",
                    "name": "Document agent",
                    "description": "Answers questions about documents",
                    "readonly": False,
                    "configuration": {
                        "instructions": "Use the search tool.",
                        "tools": [{"tool_ids": ["search-documents"]}],
                    },
                    "permissions": {
                        "update_agent": True,
                        "update_access_control": True,
                    },
                },
            ),
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_agent(
        _agent_request(),
        update_if_exists=True,
    )

    assert result.type == "chat"
    assert result.configuration is not None
    assert result.configuration.instructions == "Use the search tool."
    assert result.permissions == {
        "update_agent": True,
        "update_access_control": True,
    }
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "PUT",
    ]


@pytest.mark.asyncio
async def test_call_converse_uses_default_agent_and_maps_response() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "response": {
                        "message": "The answer",
                        "structured_output": {"answer": 42},
                    },
                    "steps": [
                        {
                            "type": "tool_call",
                            "tool_id": "search-documents",
                            "custom": "preserved",
                        }
                    ],
                    "conversation_id": "8bcbde74-e394-4d5a-a702-5dd8e524dd64",
                    "trace_id": "trace-1",
                },
            )
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").call_converse("What is the answer?")

    assert result == ConverseResponse(
        message="The answer",
        steps=[
            {
                "type": "tool_call",
                "tool_id": "search-documents",
                "custom": "preserved",
            }
        ],
        structured_output={"answer": 42},
        conversation_id="8bcbde74-e394-4d5a-a702-5dd8e524dd64",
        trace_id="trace-1",
    )
    request = _RecordingAsyncClient.requests[0]
    assert request["url"] == "http://kibana:5601/api/agent_builder/converse"
    assert request["json"] == {
        "input": "What is the answer?",
        "_execution_mode": "local",
    }


@pytest.mark.asyncio
async def test_call_converse_forwards_agent_connector_and_conversation() -> None:
    _RecordingAsyncClient.configure([httpx.Response(200, json={"response": {"message": "Follow-up"}})])

    await AgentBuilderClient("http://kibana:5601").call_converse(
        "Follow up",
        agent_id="document-agent",
        connector_id="connector-1",
        conversation_id="8bcbde74-e394-4d5a-a702-5dd8e524dd64",
    )

    assert _RecordingAsyncClient.requests[0]["json"] == {
        "input": "Follow up",
        "_execution_mode": "local",
        "agent_id": "document-agent",
        "connector_id": "connector-1",
        "conversation_id": "8bcbde74-e394-4d5a-a702-5dd8e524dd64",
    }


@pytest.mark.parametrize(
    ("status_code", "retryable", "attempts"),
    [
        (401, False, 1),
        (403, False, 1),
        (429, True, 3),
        (500, True, 3),
    ],
)
@pytest.mark.asyncio
async def test_get_tool_surfaces_http_errors(
    status_code: int,
    retryable: bool,
    attempts: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure(
        [httpx.Response(status_code, json={"message": "request failed"}) for _ in range(attempts)]
    )

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").get_tool("search-documents")

    assert exc_info.value.status_code == status_code
    assert exc_info.value.body == {"message": "request failed"}
    assert exc_info.value.retryable is retryable
    assert len(_RecordingAsyncClient.requests) == attempts
    assert (
        f"Agent Builder request failed (get tool 'search-documents') with {status_code}: "
        '{"message": "request failed"}'
    ) in caplog.text


@pytest.mark.asyncio
async def test_get_agent_retries_transient_error() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(503, json={"message": "temporarily unavailable"}),
            httpx.Response(
                200,
                json={
                    "id": "document-agent",
                    "type": "chat",
                    "name": "Document agent",
                    "configuration": {
                        "instructions": "Use the search tool.",
                        "tools": [{"tool_ids": ["search-documents"]}],
                    },
                },
            ),
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").get_agent("document-agent")

    assert result is not None
    assert result.id == "document-agent"
    assert len(_RecordingAsyncClient.requests) == 2


@pytest.mark.asyncio
async def test_create_tool_surfaces_validation_error() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(404, json={"message": "not found"}),
            httpx.Response(400, json={"message": "invalid tool configuration"}),
        ]
    )

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").create_tool(_tool_request())

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"message": "invalid tool configuration"}
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "POST",
    ]


@pytest.mark.asyncio
async def test_update_tool_surfaces_validation_error() -> None:
    _RecordingAsyncClient.configure([httpx.Response(400, json={"message": "invalid tool configuration"})])

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").update_tool(
            "search-documents",
            UpdateToolRequest(configuration={"row_limit": "invalid"}),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"message": "invalid tool configuration"}
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_create_tool_recovers_from_already_exists_race() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(404, json={"message": "not found"}),
            httpx.Response(400, json={"message": "Tool already exists"}),
            httpx.Response(
                200,
                json={"id": "search-documents", "type": "index_search"},
            ),
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_tool(_tool_request())

    assert result.id == "search-documents"
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "POST",
        "GET",
    ]


@pytest.mark.asyncio
async def test_create_agent_reuses_existing_agent_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={"id": "document-agent", "type": "chat", "name": "Existing agent"},
            )
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_agent(_agent_request())

    assert result.id == "document-agent"
    assert len(_RecordingAsyncClient.requests) == 1
    assert "already exists; using it" in caplog.text


@pytest.mark.asyncio
async def test_create_agent_creates_missing_agent() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(404, json={"message": "not found"}),
            httpx.Response(
                202,
                json={
                    "id": "document-agent",
                    "type": "chat",
                    "name": "Document agent",
                    "configuration": {
                        "instructions": "Use the search tool.",
                        "tools": [{"tool_ids": ["search-documents"]}],
                    },
                },
            ),
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").create_agent(_agent_request())

    assert result.id == "document-agent"
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "POST",
    ]


@pytest.mark.asyncio
async def test_update_missing_agent_surfaces_not_found() -> None:
    _RecordingAsyncClient.configure([httpx.Response(404, json={"message": "Agent document-agent not found"})])

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").update_agent(
            "document-agent",
            UpdateAgentRequest(name="Document agent"),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == {"message": "Agent document-agent not found"}
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.parametrize(
    ("status_code", "content"),
    [(200, b"not-json"), (204, b"")],
)
@pytest.mark.asyncio
async def test_get_tool_logs_invalid_success_body(
    status_code: int,
    content: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure([httpx.Response(status_code, content=content)])

    with pytest.raises(json.JSONDecodeError):
        await AgentBuilderClient("http://kibana:5601").get_tool("search-documents")

    assert f"Agent Builder request succeeded (get tool 'search-documents') with {status_code}" in caplog.text
    assert "returned an invalid response body" in caplog.text


@pytest.mark.asyncio
async def test_get_agent_rejects_incomplete_response() -> None:
    _RecordingAsyncClient.configure([httpx.Response(200, json={"type": "chat"})])

    with pytest.raises(ValidationError, match="id"):
        await AgentBuilderClient("http://kibana:5601").get_agent("document-agent")


@pytest.mark.parametrize(
    ("status_code", "error_message", "arguments"),
    [
        (
            400,
            "No connector available for chat execution",
            {"connector_id": "missing-connector"},
        ),
        (
            404,
            'Agent "missing-agent" not found or not available',
            {"agent_id": "missing-agent"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_call_converse_surfaces_missing_dependency(
    status_code: int,
    error_message: str,
    arguments: dict[str, str],
) -> None:
    _RecordingAsyncClient.configure([httpx.Response(status_code, json={"message": error_message})])

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").call_converse(
            "What is the answer?",
            **arguments,
        )

    assert exc_info.value.status_code == status_code
    assert exc_info.value.body == {"message": error_message}
    assert len(_RecordingAsyncClient.requests) == 1


@pytest.mark.asyncio
async def test_create_agent_surfaces_missing_tool() -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(404, json={"message": "Agent document-agent not found"}),
            httpx.Response(400, json={"message": "Tool search-documents not found"}),
        ]
    )

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").create_agent(_agent_request())

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {"message": "Tool search-documents not found"}
    assert [request["method"] for request in _RecordingAsyncClient.requests] == [
        "GET",
        "POST",
    ]


@pytest.mark.parametrize(("status_code", "attempts"), [(400, 1), (500, 3)])
@pytest.mark.asyncio
async def test_call_converse_surfaces_failure(status_code: int, attempts: int) -> None:
    _RecordingAsyncClient.configure(
        [httpx.Response(status_code, json={"message": "Converse failed"}) for _ in range(attempts)]
    )

    with pytest.raises(AgentBuilderError) as exc_info:
        await AgentBuilderClient("http://kibana:5601").call_converse("What is the answer?")

    assert exc_info.value.status_code == status_code
    assert exc_info.value.retryable is (status_code == 500)
    assert len(_RecordingAsyncClient.requests) == attempts


@pytest.mark.asyncio
async def test_call_converse_maps_empty_response() -> None:
    _RecordingAsyncClient.configure([httpx.Response(200, json={})])

    result = await AgentBuilderClient("http://kibana:5601").call_converse("What is the answer?")

    assert result == ConverseResponse()


@pytest.mark.asyncio
async def test_call_converse_warns_about_unexpected_response_shapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _RecordingAsyncClient.configure(
        [
            httpx.Response(
                200,
                json={
                    "response": "unexpected",
                    "steps": {"type": "tool_call"},
                },
            )
        ]
    )

    result = await AgentBuilderClient("http://kibana:5601").call_converse("What is the answer?")

    assert result == ConverseResponse()
    assert "Expected Agent Builder converse response to be a dict, got str" in caplog.text
    assert "Expected Agent Builder converse steps to be a list, got dict" in caplog.text
