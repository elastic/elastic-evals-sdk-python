# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Elasticsearch client for querying indexed evaluation traces."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel


class EsqlColumn(BaseModel):
    name: str
    type: str


class EsqlResponse(BaseModel):
    columns: list[EsqlColumn]
    values: list[list[Any]]


class ElasticsearchTraceClient:
    def __init__(
        self,
        elasticsearch_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.elasticsearch_url = elasticsearch_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def query(self, query: str) -> EsqlResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.elasticsearch_url}/_query?format=json",
                json={"query": query},
                headers=headers,
            )
            response.raise_for_status()

        return EsqlResponse.model_validate(response.json())
