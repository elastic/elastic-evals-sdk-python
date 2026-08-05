# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""DTOs for Kibana evaluator APIs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from elastic_evals.api.constants import MAX_EVALUATORS_PER_REQUEST

EvaluatorKind = Literal["llm", "code"]
EvaluationMode = Literal["single-turn", "multi-turn"]
InstrumentationProfile = Literal[
    "elastic-inference",
    "otel-genai-events",
    "otel-genai-attributes",
    "claude-code",
]
EvidenceStatus = Literal["found", "not_found", "content_redacted"]
EvaluationResultStatus = Literal["ok", "error"]
EvaluationErrorCode = Literal["evidence_unmet"]


class EvaluatorDefinition(BaseModel):
    name: str
    version: str
    kind: EvaluatorKind
    description: str
    reference_data_schema: dict[str, Any] | None = None
    evidence_schema: dict[str, Any] | None = None


class ListEvaluatorsResponse(BaseModel):
    evaluators: list[EvaluatorDefinition]


class ResolveInstrumentationRequest(BaseModel):
    trace_id: str


class EvidenceProbe(BaseModel):
    status: EvidenceStatus
    field: str | None = None
    sample: str | None = None


class ProfileEvidence(BaseModel):
    user_query: EvidenceProbe
    agent_response: EvidenceProbe
    tool_calls: EvidenceProbe


class InstrumentationProfileResult(BaseModel):
    profile: str
    evidence: ProfileEvidence


class RecommendedInstrumentation(BaseModel):
    profile: str


class ResolveInstrumentationResponse(BaseModel):
    profiles: list[InstrumentationProfileResult]
    recommended_instrumentation: RecommendedInstrumentation | None


class EvaluationTrace(BaseModel):
    trace_id: str
    reference_data: dict[str, Any] | None = None


class EvaluationInstrumentation(BaseModel):
    profile: InstrumentationProfile = "elastic-inference"


class EvaluationSubject(BaseModel):
    mode: EvaluationMode = "single-turn"
    traces: Annotated[list[EvaluationTrace], Field(min_length=1, max_length=1)]
    instrumentation: EvaluationInstrumentation | None = None


class ValidateEvaluatorConfig(BaseModel):
    name: str
    version: str | None = None


class EvaluateEvaluatorConfig(ValidateEvaluatorConfig):
    connector_id: str | None = None


class ValidateEvaluatorsRequest(BaseModel):
    subject: EvaluationSubject
    evaluators: Annotated[
        list[ValidateEvaluatorConfig],
        Field(min_length=1, max_length=MAX_EVALUATORS_PER_REQUEST),
    ]


class EvaluateRequest(BaseModel):
    subject: EvaluationSubject
    evaluators: Annotated[
        list[EvaluateEvaluatorConfig],
        Field(min_length=1, max_length=MAX_EVALUATORS_PER_REQUEST),
    ]


class EvaluatorIdentity(BaseModel):
    name: str
    version: str
    kind: EvaluatorKind


class EvaluatorScore(BaseModel):
    name: str
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] | None = None


class EvaluatorError(BaseModel):
    code: EvaluationErrorCode | None = None
    message: str


class EvaluateResult(BaseModel):
    status: EvaluationResultStatus
    evaluator: EvaluatorIdentity
    scores: list[EvaluatorScore] | None = None
    error: EvaluatorError | None = None


class EvaluateResponse(BaseModel):
    results: list[EvaluateResult]


class ValidateEvaluatorResult(BaseModel):
    name: str
    version: str
    ready: bool
    unmet: list[str]
    remediation: str | None = None


class ValidateEvaluatorsResponse(BaseModel):
    evaluators: list[ValidateEvaluatorResult]
