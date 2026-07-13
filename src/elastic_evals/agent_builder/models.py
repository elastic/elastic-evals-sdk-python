"""DTOs for the Kibana Agent Builder public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


class ToolSelection(BaseModel):
    tool_ids: list[str]


class AgentConfiguration(BaseModel):
    instructions: str | None = None
    tools: list[ToolSelection] = Field(default_factory=list)
    connector_ids: list[str] | None = None


class CreateAgentRequest(BaseModel):
    id: str
    name: str
    description: str
    configuration: AgentConfiguration
    avatar_color: str | None = None
    avatar_symbol: str | None = None
    labels: list[str] | None = None


class ToolResponse(BaseModel):
    id: str
    type: str | None = None


class AgentResponse(BaseModel):
    id: str
    name: str | None = None
