"""Client for the Kibana Agent Builder public API."""

from __future__ import annotations

import json
from typing import Any, NoReturn

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from elastic_evals.agent_builder.constants import (
    AGENT_URL,
    AGENTS_URL,
    TOOL_URL,
    TOOLS_URL,
)
from elastic_evals.agent_builder.errors import AgentBuilderError
from elastic_evals.agent_builder.headers import build_agent_builder_headers
from elastic_evals.agent_builder.models import (
    AgentResponse,
    CreateAgentRequest,
    CreateToolRequest,
    ToolResponse,
)


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in {408, 429} or status_code >= 500


def _is_retryable_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return _is_retryable_status_code(error.response.status_code)
    if isinstance(error, httpx.HTTPError):
        return True
    if isinstance(error, AgentBuilderError):
        return error.retryable
    return False


def _is_already_exists(error: AgentBuilderError) -> bool:
    return error.status_code == 400 and "already exists" in error.message.lower()


class AgentBuilderClient:
    def __init__(
        self,
        kibana_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.kibana_url = kibana_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def create_tool(self, request: CreateToolRequest) -> ToolResponse:
        """Create a tool via POST /api/agent_builder/tools."""
        url = f"{self.kibana_url}{TOOLS_URL}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, json=request.model_dump(exclude_none=True), headers=headers
            )

        if 200 <= response.status_code < 300:
            return ToolResponse.model_validate(response.json())

        self._raise_error(response, context=f"create tool '{request.id}'")

    async def get_tool(self, tool_id: str) -> ToolResponse | None:
        """Fetch a tool by id, or None if it does not exist."""
        url = f"{self.kibana_url}{TOOL_URL.format(tool_id=tool_id)}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)

        if response.status_code == 200:
            return ToolResponse.model_validate(response.json())
        if response.status_code == 404:
            return None

        self._raise_error(response, context=f"get tool '{tool_id}'")

    async def get_or_create_tool(self, request: CreateToolRequest) -> ToolResponse:
        """Return the existing tool with request.id, or create it."""
        existing = await self.get_tool(request.id)
        if existing is not None:
            return existing
        try:
            return await self.create_tool(request)
        except AgentBuilderError as error:
            if _is_already_exists(error):
                fetched = await self.get_tool(request.id)
                if fetched is not None:
                    return fetched
            raise

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def create_agent(self, request: CreateAgentRequest) -> AgentResponse:
        """Create an agent via POST /api/agent_builder/agents."""
        url = f"{self.kibana_url}{AGENTS_URL}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, json=request.model_dump(exclude_none=True), headers=headers
            )

        if 200 <= response.status_code < 300:
            return AgentResponse.model_validate(response.json())

        self._raise_error(response, context=f"create agent '{request.id}'")

    async def get_agent(self, agent_id: str) -> AgentResponse | None:
        """Fetch an agent by id, or None if it does not exist."""
        url = f"{self.kibana_url}{AGENT_URL.format(agent_id=agent_id)}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)

        if response.status_code == 200:
            return AgentResponse.model_validate(response.json())
        if response.status_code == 404:
            return None

        self._raise_error(response, context=f"get agent '{agent_id}'")

    async def get_or_create_agent(self, request: CreateAgentRequest) -> AgentResponse:
        """Return the existing agent with request.id, or create it."""
        existing = await self.get_agent(request.id)
        if existing is not None:
            return existing
        try:
            return await self.create_agent(request)
        except AgentBuilderError as error:
            if _is_already_exists(error):
                fetched = await self.get_agent(request.id)
                if fetched is not None:
                    return fetched
            raise

    def _raise_error(self, response: httpx.Response, *, context: str) -> NoReturn:
        status_code = response.status_code
        body: Any
        try:
            body = response.json()
            body_text = json.dumps(body, ensure_ascii=True)
        except (ValueError, json.JSONDecodeError):
            body = response.text
            body_text = response.text

        message = f"Agent Builder request failed ({context}) with {status_code}"
        if body_text:
            message = f"{message}: {body_text}"

        raise AgentBuilderError(
            message=message,
            status_code=status_code,
            body=body,
            retryable=_is_retryable_status_code(status_code),
        )
