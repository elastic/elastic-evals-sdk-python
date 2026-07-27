# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for the Kibana Agent Builder public API."""

from __future__ import annotations

import json
from typing import Any, NoReturn

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from elastic_evals.agent_builder.constants import (
    AGENT_URL,
    AGENTS_URL,
    CONVERSE_URL,
    TOOL_URL,
    TOOLS_URL,
)
from elastic_evals.agent_builder.errors import AgentBuilderError
from elastic_evals.agent_builder.headers import build_agent_builder_headers
from elastic_evals.agent_builder.models import (
    AgentResponse,
    ConverseResponse,
    CreateAgentRequest,
    CreateToolRequest,
    ToolResponse,
    UpdateAgentConfiguration,
    UpdateAgentRequest,
    UpdateToolRequest,
)
from elastic_evals.utils.logging import log

logger = log.getChild(__name__)


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

    async def create_tool(
        self,
        request: CreateToolRequest,
        *,
        update_if_exists: bool = False,
    ) -> ToolResponse:
        """Create a tool, optionally updating an existing tool with the same ID."""
        existing = await self.get_tool(request.id)
        if existing is not None:
            return await self._use_or_update_tool(
                existing,
                request,
                update_if_exists=update_if_exists,
            )

        try:
            created = await self._create_tool(request)
        except AgentBuilderError as error:
            if not _is_already_exists(error):
                raise
            existing = await self.get_tool(request.id)
            if existing is None:
                raise
            return await self._use_or_update_tool(
                existing,
                request,
                update_if_exists=update_if_exists,
            )

        logger.info("Created Agent Builder tool '%s'", request.id)
        return created

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def _create_tool(self, request: CreateToolRequest) -> ToolResponse:
        url = f"{self.kibana_url}{TOOLS_URL}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request.model_dump(exclude_none=True), headers=headers)

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

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def update_tool(
        self,
        tool_id: str,
        request: UpdateToolRequest,
    ) -> ToolResponse:
        """Update an existing tool."""
        url = f"{self.kibana_url}{TOOL_URL.format(tool_id=tool_id)}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(url, json=request.model_dump(exclude_none=True), headers=headers)

        if response.status_code == 200:
            return ToolResponse.model_validate(response.json())

        self._raise_error(response, context=f"update tool '{tool_id}'")

    async def _use_or_update_tool(
        self,
        existing: ToolResponse,
        request: CreateToolRequest,
        *,
        update_if_exists: bool,
    ) -> ToolResponse:
        if not update_if_exists:
            logger.info("Agent Builder tool '%s' already exists; using it", request.id)
            return existing
        if existing.type is not None and existing.type != request.type:
            raise ValueError(
                f"Cannot update Agent Builder tool '{request.id}' from type '{existing.type}' to '{request.type}'"
            )
        logger.info("Agent Builder tool '%s' already exists; updating it", request.id)
        return await self.update_tool(
            request.id,
            UpdateToolRequest(
                description=request.description,
                tags=request.tags,
                configuration=request.configuration,
            ),
        )

    async def create_agent(
        self,
        request: CreateAgentRequest,
        *,
        update_if_exists: bool = False,
    ) -> AgentResponse:
        """Create an agent, optionally updating an existing agent with the same ID."""
        existing = await self.get_agent(request.id)
        if existing is not None:
            return await self._use_or_update_agent(
                existing,
                request,
                update_if_exists=update_if_exists,
            )

        try:
            created = await self._create_agent(request)
        except AgentBuilderError as error:
            if not _is_already_exists(error):
                raise
            existing = await self.get_agent(request.id)
            if existing is None:
                raise
            return await self._use_or_update_agent(
                existing,
                request,
                update_if_exists=update_if_exists,
            )

        logger.info("Created Agent Builder agent '%s'", request.id)
        return created

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def _create_agent(self, request: CreateAgentRequest) -> AgentResponse:
        url = f"{self.kibana_url}{AGENTS_URL}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request.model_dump(exclude_none=True), headers=headers)

        if 200 <= response.status_code < 300:
            return AgentResponse.model_validate(response.json())

        self._raise_error(response, context=f"create agent '{request.id}'")

    async def get_agent(self, agent_id: str) -> AgentResponse | None:
        """Fetch an agent and its metadata by ID, or None if it does not exist."""
        url = f"{self.kibana_url}{AGENT_URL.format(agent_id=agent_id)}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)

        if response.status_code == 200:
            return AgentResponse.model_validate(response.json())
        if response.status_code == 404:
            return None

        self._raise_error(response, context=f"get agent '{agent_id}'")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def update_agent(
        self,
        agent_id: str,
        request: UpdateAgentRequest,
    ) -> AgentResponse:
        """Update an existing agent."""
        url = f"{self.kibana_url}{AGENT_URL.format(agent_id=agent_id)}"
        headers = build_agent_builder_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(url, json=request.model_dump(exclude_none=True), headers=headers)

        if response.status_code == 200:
            return AgentResponse.model_validate(response.json())

        self._raise_error(response, context=f"update agent '{agent_id}'")

    async def _use_or_update_agent(
        self,
        existing: AgentResponse,
        request: CreateAgentRequest,
        *,
        update_if_exists: bool,
    ) -> AgentResponse:
        if not update_if_exists:
            logger.info("Agent Builder agent '%s' already exists; using it", request.id)
            return existing
        logger.info("Agent Builder agent '%s' already exists; updating it", request.id)
        return await self.update_agent(
            request.id,
            UpdateAgentRequest(
                name=request.name,
                description=request.description,
                configuration=UpdateAgentConfiguration.model_validate(
                    request.configuration.model_dump(exclude_none=True)
                ),
                avatar_color=request.avatar_color,
                avatar_symbol=request.avatar_symbol,
                labels=request.labels,
            ),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    async def call_converse(
        self,
        input_text: str,
        *,
        agent_id: str | None = None,
        connector_id: str | None = None,
        conversation_id: str | None = None,
    ) -> ConverseResponse:
        """Send a message to an Agent Builder agent."""
        url = f"{self.kibana_url}{CONVERSE_URL}"
        headers = build_agent_builder_headers(self.api_key)
        request_body = {
            "input": input_text,
            "_execution_mode": "local",
            **({"agent_id": agent_id} if agent_id is not None else {}),
            **({"connector_id": connector_id} if connector_id is not None else {}),
            **({"conversation_id": conversation_id} if conversation_id is not None else {}),
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=request_body, headers=headers)

        if response.status_code != 200:
            self._raise_error(response, context="call converse")

        payload = response.json()
        response_payload = payload.get("response")
        if not isinstance(response_payload, dict):
            response_payload = {}
        steps = payload.get("steps")
        if not isinstance(steps, list):
            steps = []

        return ConverseResponse(
            message=response_payload.get("message") or "",
            steps=steps,
            structured_output=response_payload.get("structured_output"),
            conversation_id=payload.get("conversation_id"),
            trace_id=payload.get("trace_id"),
        )

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
