# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""DTOs for the Kibana Agent Builder public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EsqlToolParamType = Literal["string", "integer", "float", "boolean", "date", "array"]


class IndexSearchToolConfig(BaseModel):
    pattern: str
    row_limit: int | None = None
    custom_instructions: str | None = None


class EsqlToolParam(BaseModel):
    type: EsqlToolParamType
    description: str
    optional: bool | None = None
    defaultValue: Any | None = None


class EsqlToolConfig(BaseModel):
    query: str
    params: dict[str, EsqlToolParam] = Field(default_factory=dict)


class CreateToolRequest(BaseModel):
    id: str
    type: Literal["index_search", "esql"]
    description: str
    tags: list[str] = Field(default_factory=list)
    configuration: IndexSearchToolConfig | EsqlToolConfig


class UpdateToolRequest(BaseModel):
    description: str | None = None
    tags: list[str] | None = None
    configuration: IndexSearchToolConfig | EsqlToolConfig | dict[str, Any] | None = None


class ToolSelection(BaseModel):
    tool_ids: list[str]


class AgentConfiguration(BaseModel):
    instructions: str | None = None
    tools: list[ToolSelection] = Field(default_factory=list)
    skill_ids: list[str] | None = None
    enable_elastic_capabilities: bool | None = None
    workflow_ids: list[str] | None = None
    plugin_ids: list[str] | None = None
    connector_ids: list[str] | None = None


class UpdateAgentConfiguration(BaseModel):
    instructions: str | None = None
    tools: list[ToolSelection] | None = None
    skill_ids: list[str] | None = None
    enable_elastic_capabilities: bool | None = None
    workflow_ids: list[str] | None = None
    plugin_ids: list[str] | None = None
    connector_ids: list[str] | None = None


class CreateAgentRequest(BaseModel):
    id: str
    name: str
    description: str
    configuration: AgentConfiguration
    avatar_color: str | None = None
    avatar_symbol: str | None = None
    labels: list[str] | None = None


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    configuration: UpdateAgentConfiguration | None = None
    avatar_color: str | None = None
    avatar_symbol: str | None = None
    labels: list[str] | None = None


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    description: str | None = None
    readonly: bool | None = None
    tags: list[str] = Field(default_factory=list)
    configuration: dict[str, Any] | None = None
    experimental: bool | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    name: str | None = None
    description: str | None = None
    readonly: bool | None = None
    access_control: dict[str, Any] | None = None
    created_by: dict[str, Any] | None = None
    labels: list[str] | None = None
    avatar_icon: str | None = None
    avatar_color: str | None = None
    avatar_symbol: str | None = None
    configuration: AgentConfiguration | None = None
    permissions: dict[str, bool] | None = None


class ConverseStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    tool_id: str | None = None
    tool_call_id: str | None = None
    params: dict[str, Any] | None = None
    results: list[Any] | None = None


class ConverseResponse(BaseModel):
    message: str = ""
    steps: list[ConverseStep] = Field(default_factory=list)
    structured_output: Any | None = None
    conversation_id: str | None = None
    trace_id: str | None = None
