# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from elastic_evals.integrations.agent_builder import build_agent_builder_headers


def test_build_agent_builder_headers_without_api_key() -> None:
    headers = build_agent_builder_headers(None)

    assert headers == {
        "Content-Type": "application/json",
        "kbn-xsrf": "true",
        "x-elastic-internal-origin": "true",
        "Elastic-Api-Version": "2023-10-31",
    }


def test_build_agent_builder_headers_with_api_key() -> None:
    headers = build_agent_builder_headers("key-123")

    assert headers["Authorization"] == "ApiKey key-123"


def test_build_agent_builder_headers_with_empty_api_key() -> None:
    headers = build_agent_builder_headers("")

    assert "Authorization" not in headers
