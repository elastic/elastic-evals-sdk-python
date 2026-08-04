# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import AsyncExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import quote

import httpx
import pytest

from elastic_evals.integrations.agent_builder import (
    AgentBuilderClient,
    AgentConfiguration,
    CreateAgentRequest,
    CreateToolRequest,
    EsqlToolConfig,
    ToolSelection,
    build_agent_builder_headers,
)

_RUN_ACCEPTANCE_TESTS = os.environ.get("RUN_AGENT_BUILDER_ACCEPTANCE", "").lower() in {
    "1",
    "true",
    "yes",
}

pytestmark = pytest.mark.skipif(
    not _RUN_ACCEPTANCE_TESTS,
    reason="Set RUN_AGENT_BUILDER_ACCEPTANCE=1 to run Agent Builder acceptance tests",
)

_DIRECT_ANSWER = "AGENT_BUILDER_ACCEPTANCE_OK"


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _is_title_request(body: dict[str, Any]) -> bool:
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name") == "set_title":
            return True

    for message in body.get("messages", []):
        if (
            isinstance(message, dict)
            and message.get("role") == "system"
            and "title-generation utility" in str(message.get("content", ""))
        ):
            return True
    return False


class _LlmScenario:
    def __init__(
        self,
        *,
        answer: str,
        tool_id: str | None = None,
        expected_tool_value: str | None = None,
    ) -> None:
        self.answer = answer
        self.tool_id = tool_id
        self.expected_tool_value = expected_tool_value

    def respond(self, body: dict[str, Any]) -> str | dict[str, Any]:
        if _is_title_request(body):
            return _tool_call("set_title", {"title": "Agent Builder acceptance test"})

        if self.tool_id is None:
            return self.answer

        messages = body.get("messages", [])
        if any(isinstance(message, dict) and message.get("role") == "tool" for message in messages):
            serialized_messages = json.dumps(messages)
            if self.expected_tool_value and self.expected_tool_value in serialized_messages:
                return self.answer
            return "EXPECTED_TOOL_RESULT_WAS_NOT_RECEIVED"

        tools = body.get("tools", [])
        available_names = [
            function["name"]
            for tool in tools
            if isinstance(tool, dict)
            and isinstance((function := tool.get("function")), dict)
            and isinstance(function.get("name"), str)
        ]
        tool_name = next(
            (name for name in available_names if name == self.tool_id or self.tool_id in name),
            None,
        )
        if tool_name is None:
            return "EXPECTED_TOOL_WAS_NOT_AVAILABLE"
        return _tool_call(tool_name, {})


def _openai_response(message: str | dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    content = message.get("content", "") if isinstance(message, dict) else message
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _openai_chunks(message: str | dict[str, Any]) -> list[str]:
    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    content = message.get("content", "") if isinstance(message, dict) else message
    delta: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        delta["tool_calls"] = [
            {
                **tool_call,
                "index": index,
            }
            for index, tool_call in enumerate(tool_calls)
        ]

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    chunks = [
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        },
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
        },
    ]
    return [*(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks), "data: [DONE]\n\n"]


class _LlmProxy:
    def __init__(self, scenario: _LlmScenario) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0"))
                request_body = json.loads(self.rfile.read(content_length))
                message = scenario.respond(request_body)

                if request_body.get("stream"):
                    payload = "".join(_openai_chunks(message)).encode()
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(_openai_response(message)).encode()
                    content_type = "application/json"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, message_format: str, *args: Any) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        advertised_host = os.environ.get(
            "AGENT_BUILDER_ACCEPTANCE_PROXY_HOST",
            "127.0.0.1",
        )
        return f"http://{advertised_host}:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _AcceptanceEnvironment:
    def __init__(self) -> None:
        self.kibana_url = os.environ.get("KIBANA_URL", "http://localhost:5601").rstrip("/")
        self.kibana_api_key = os.environ.get("KIBANA_API_KEY")
        self.elasticsearch_url = (
            os.environ.get("ELASTICSEARCH_URL") or os.environ.get("ES_URL") or "http://localhost:9200"
        ).rstrip("/")
        self.elasticsearch_api_key = os.environ.get("ELASTICSEARCH_API_KEY")
        username = os.environ.get("ELASTICSEARCH_USERNAME")
        password = os.environ.get("ELASTICSEARCH_PASSWORD")
        self.elasticsearch_auth = (
            httpx.BasicAuth(username, password)
            if not self.elasticsearch_api_key and username is not None and password is not None
            else None
        )

    @property
    def agent_builder_client(self) -> AgentBuilderClient:
        return AgentBuilderClient(
            self.kibana_url,
            api_key=self.kibana_api_key,
            timeout=120,
        )

    async def create_connector(self, *, name: str, proxy_url: str) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.kibana_url}/api/actions/connector",
                headers=build_agent_builder_headers(self.kibana_api_key),
                json={
                    "name": name,
                    "config": {
                        "apiProvider": "OpenAI",
                        "apiUrl": proxy_url,
                        "defaultModel": "gpt-4",
                    },
                    "secrets": {"apiKey": "acceptance-test-key"},
                    "connector_type_id": ".gen-ai",
                },
            )
        response.raise_for_status()
        connector_id = response.json().get("id")
        if not isinstance(connector_id, str):
            raise RuntimeError("Kibana did not return a connector ID")
        return connector_id

    async def delete_connector(self, connector_id: str) -> None:
        await self._delete_kibana(f"/api/actions/connector/{quote(connector_id, safe='')}")

    async def delete_conversation(self, conversation_id: str) -> None:
        await self._delete_kibana(f"/api/agent_builder/conversations/{quote(conversation_id, safe='')}")

    async def delete_agent(self, agent_id: str) -> None:
        await self._delete_kibana(f"/api/agent_builder/agents/{quote(agent_id, safe='')}")

    async def delete_tool(self, tool_id: str) -> None:
        await self._delete_kibana(
            f"/api/agent_builder/tools/{quote(tool_id, safe='')}",
            params={"force": "true"},
        )

    async def _delete_kibana(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.delete(
                f"{self.kibana_url}{path}",
                params=params,
                headers=build_agent_builder_headers(self.kibana_api_key),
            )
        if response.status_code != 404:
            response.raise_for_status()

    async def create_index(self, index_name: str, *, answer: str) -> None:
        async with httpx.AsyncClient(
            timeout=30,
            auth=self.elasticsearch_auth,
        ) as client:
            create_response = await client.put(
                f"{self.elasticsearch_url}/{quote(index_name, safe='')}",
                headers=self._elasticsearch_headers(),
                json={"mappings": {"properties": {"answer": {"type": "text"}}}},
            )
            create_response.raise_for_status()
            index_response = await client.post(
                f"{self.elasticsearch_url}/{quote(index_name, safe='')}/_doc/acceptance",
                params={"refresh": "true"},
                headers=self._elasticsearch_headers(),
                json={"answer": answer},
            )
            index_response.raise_for_status()

    async def delete_index(self, index_name: str) -> None:
        async with httpx.AsyncClient(
            timeout=30,
            auth=self.elasticsearch_auth,
        ) as client:
            response = await client.delete(
                f"{self.elasticsearch_url}/{quote(index_name, safe='')}",
                headers=self._elasticsearch_headers(),
            )
        if response.status_code != 404:
            response.raise_for_status()

    def _elasticsearch_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.elasticsearch_api_key:
            headers["Authorization"] = f"ApiKey {self.elasticsearch_api_key}"
        return headers


@pytest.mark.asyncio
async def test_agent_builder_converse_returns_expected_answer() -> None:
    environment = _AcceptanceEnvironment()
    suffix = uuid.uuid4().hex[:12]
    agent_id = f"acceptance-converse-{suffix}"
    proxy = _LlmProxy(_LlmScenario(answer=_DIRECT_ANSWER))

    async with AsyncExitStack() as stack:
        stack.callback(proxy.close)
        connector_id = await environment.create_connector(
            name=f"acceptance-converse-{suffix}",
            proxy_url=proxy.url,
        )
        stack.push_async_callback(environment.delete_connector, connector_id)

        client = environment.agent_builder_client
        await client.create_agent(
            CreateAgentRequest(
                id=agent_id,
                name="Acceptance converse agent",
                description="Returns a deterministic acceptance-test response.",
                configuration=AgentConfiguration(
                    instructions="Answer the user directly.",
                    tools=[],
                ),
            )
        )
        stack.push_async_callback(environment.delete_agent, agent_id)

        result = await client.call_converse(
            "Return the acceptance-test response.",
            agent_id=agent_id,
            connector_id=connector_id,
        )
        if result.conversation_id:
            stack.push_async_callback(
                environment.delete_conversation,
                result.conversation_id,
            )

        assert result.message == _DIRECT_ANSWER


@pytest.mark.asyncio
async def test_agent_builder_tool_grounded_converse_returns_document_answer() -> None:
    environment = _AcceptanceEnvironment()
    suffix = uuid.uuid4().hex[:12]
    index_name = f"agent_builder_acceptance_{suffix}"
    tool_id = f"acceptance-esql-{suffix}"
    agent_id = f"acceptance-tool-agent-{suffix}"
    indexed_value = f"cobalt-{suffix}"
    expected_answer = f"The indexed acceptance value is {indexed_value}."
    proxy = _LlmProxy(
        _LlmScenario(
            answer=expected_answer,
            tool_id=tool_id,
            expected_tool_value=indexed_value,
        )
    )

    async with AsyncExitStack() as stack:
        stack.callback(proxy.close)
        connector_id = await environment.create_connector(
            name=f"acceptance-tool-{suffix}",
            proxy_url=proxy.url,
        )
        stack.push_async_callback(environment.delete_connector, connector_id)

        await environment.create_index(index_name, answer=indexed_value)
        stack.push_async_callback(environment.delete_index, index_name)

        client = environment.agent_builder_client
        await client.create_tool(
            CreateToolRequest(
                id=tool_id,
                type="esql",
                description="Read the acceptance-test value.",
                configuration=EsqlToolConfig(
                    query=f"FROM {index_name} | KEEP answer | LIMIT 1",
                ),
            )
        )
        stack.push_async_callback(environment.delete_tool, tool_id)

        await client.create_agent(
            CreateAgentRequest(
                id=agent_id,
                name="Acceptance tool agent",
                description="Reads the acceptance-test value with ES|QL.",
                configuration=AgentConfiguration(
                    instructions="Use the configured tool to answer the question.",
                    tools=[ToolSelection(tool_ids=[tool_id])],
                ),
            )
        )
        stack.push_async_callback(environment.delete_agent, agent_id)

        result = await client.call_converse(
            "What is the indexed acceptance value?",
            agent_id=agent_id,
            connector_id=connector_id,
        )
        if result.conversation_id:
            stack.push_async_callback(
                environment.delete_conversation,
                result.conversation_id,
            )

        assert result.message == expected_answer
