# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Client for Kibana evaluator APIs."""

from __future__ import annotations

from typing import Any, Literal

import httpx

from elastic_evals.api.constants import (
    EVALS_EVALUATE_URL,
    EVALS_EVALUATORS_URL,
    EVALS_RESOLVE_INSTRUMENTATION_URL,
    EVALS_VALIDATE_URL,
)
from elastic_evals.api.errors import KibanaEvaluatorsError
from elastic_evals.api.evaluators_models import (
    EvaluateRequest,
    EvaluateResponse,
    ListEvaluatorsResponse,
    ResolveInstrumentationRequest,
    ResolveInstrumentationResponse,
    ValidateEvaluatorsRequest,
    ValidateEvaluatorsResponse,
)
from elastic_evals.api.headers import build_kibana_headers
from elastic_evals.api.response import raise_kibana_error
from elastic_evals.api.retry import retry_kibana_api_call


class KibanaEvaluatorsClient:
    """Async client for Kibana's raw evaluator API surface."""

    def __init__(
        self,
        kibana_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.kibana_url = kibana_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @retry_kibana_api_call
    async def list_evaluators(self) -> ListEvaluatorsResponse:
        response = await self._request("GET", EVALS_EVALUATORS_URL)
        return ListEvaluatorsResponse.model_validate(response.json())

    @retry_kibana_api_call
    async def resolve_instrumentation(self, trace_id: str) -> ResolveInstrumentationResponse:
        payload = ResolveInstrumentationRequest(trace_id=trace_id)
        response = await self._request(
            "POST",
            EVALS_RESOLVE_INSTRUMENTATION_URL,
            payload.model_dump(exclude_none=True),
        )
        return ResolveInstrumentationResponse.model_validate(response.json())

    @retry_kibana_api_call
    async def validate(self, payload: ValidateEvaluatorsRequest) -> ValidateEvaluatorsResponse:
        response = await self._request(
            "POST",
            EVALS_VALIDATE_URL,
            payload.model_dump(exclude_none=True),
        )
        return ValidateEvaluatorsResponse.model_validate(response.json())

    @retry_kibana_api_call
    async def evaluate(self, payload: EvaluateRequest) -> EvaluateResponse:
        response = await self._request(
            "POST",
            EVALS_EVALUATE_URL,
            payload.model_dump(exclude_none=True),
        )
        return EvaluateResponse.model_validate(response.json())

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        request_body: dict[str, Any] | None = None,
    ) -> httpx.Response:
        url = f"{self.kibana_url}{path}"
        headers = build_kibana_headers(self.api_key)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if method == "GET":
                response = await client.get(url, headers=headers)
            else:
                response = await client.post(url, json=request_body, headers=headers)

        if response.status_code == 200:
            return response

        raise_kibana_error(
            response,
            error_cls=KibanaEvaluatorsError,
            context="Kibana evaluators request",
        )
