# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

from elastic_evals.config import ElasticEvalsConfig


def test_config_constructs_without_connector_or_kibana_credentials() -> None:
    config = ElasticEvalsConfig()

    assert config.connector_id is None
    assert config.kibana_api_key is None


def test_from_env_no_longer_requires_connector_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONNECTOR_ID", raising=False)

    config = ElasticEvalsConfig.from_env()

    assert config.connector_id is None


def test_from_env_treats_empty_connector_id_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTOR_ID", "")

    assert ElasticEvalsConfig.from_env().connector_id is None


def test_from_env_still_reads_connector_id_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONNECTOR_ID", "conn-1")

    assert ElasticEvalsConfig.from_env().connector_id == "conn-1"


def test_tracing_is_on_unless_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ElasticEvalsConfig().tracing.enabled is True

    monkeypatch.delenv("ELASTIC_EVALS_TRACING_ENABLED", raising=False)
    assert ElasticEvalsConfig.from_env().tracing.enabled is True

    monkeypatch.setenv("ELASTIC_EVALS_TRACING_ENABLED", "false")
    assert ElasticEvalsConfig.from_env().tracing.enabled is False
