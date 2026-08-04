# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Kibana Agent Builder client, models, and helpers."""

from elastic_evals.integrations.agent_builder.client import AgentBuilderClient
from elastic_evals.integrations.agent_builder.constants import (
    AGENT_BUILDER_API_VERSION,
    AGENT_URL,
    AGENTS_URL,
    CONVERSE_URL,
    TOOL_URL,
    TOOLS_URL,
)
from elastic_evals.integrations.agent_builder.errors import AgentBuilderError
from elastic_evals.integrations.agent_builder.headers import build_agent_builder_headers
from elastic_evals.integrations.agent_builder.models import (
    AgentConfiguration,
    AgentResponse,
    ConverseResponse,
    ConverseStep,
    CreateAgentRequest,
    CreateToolRequest,
    EsqlToolConfig,
    EsqlToolParam,
    IndexSearchToolConfig,
    ToolResponse,
    ToolSelection,
    UpdateAgentConfiguration,
    UpdateAgentRequest,
    UpdateToolRequest,
)

__all__ = [
    "AGENT_BUILDER_API_VERSION",
    "AGENT_URL",
    "AGENTS_URL",
    "AgentBuilderClient",
    "AgentBuilderError",
    "AgentConfiguration",
    "AgentResponse",
    "CONVERSE_URL",
    "ConverseResponse",
    "ConverseStep",
    "CreateAgentRequest",
    "CreateToolRequest",
    "EsqlToolConfig",
    "EsqlToolParam",
    "IndexSearchToolConfig",
    "ToolResponse",
    "ToolSelection",
    "UpdateAgentConfiguration",
    "UpdateAgentRequest",
    "UpdateToolRequest",
    "TOOL_URL",
    "TOOLS_URL",
    "build_agent_builder_headers",
]
