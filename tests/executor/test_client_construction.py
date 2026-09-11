# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

from elastic_evals.api import KibanaEvaluatorsClient
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.datasets import InMemoryDatasetStore, KibanaDatasetStore
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.export import InMemoryScoreSink, KibanaScoreSink
from elastic_evals.tracing import TracingConfig


def _config(**overrides: object) -> ElasticEvalsConfig:
    return ElasticEvalsConfig(tracing=TracingConfig(enabled=False), **overrides)  # type: ignore[arg-type]


def test_local_client_needs_no_kibana_credentials_or_connector() -> None:
    client = ElasticEvalsClient.local(_config())

    assert isinstance(client.dataset_store, InMemoryDatasetStore)
    assert isinstance(client.score_sink, InMemoryScoreSink)


def test_injected_store_and_sink_are_returned_as_given() -> None:
    store = InMemoryDatasetStore()
    sink = InMemoryScoreSink()

    client = ElasticEvalsClient(_config(), dataset_store=store, score_sink=sink)

    assert client.dataset_store is store
    assert client.score_sink is sink


def test_defaults_are_kibana_backed_and_built_once() -> None:
    client = ElasticEvalsClient(_config(kibana_url="http://kibana:5601", kibana_api_key="k"))

    assert isinstance(client.dataset_store, KibanaDatasetStore)
    assert isinstance(client.score_sink, KibanaScoreSink)
    assert client.dataset_store is client.dataset_store
    assert client.score_sink is client.score_sink


def test_evaluators_client_is_lazy_and_cached() -> None:
    client = ElasticEvalsClient(_config())

    first = client.get_evaluators_client()

    assert isinstance(first, KibanaEvaluatorsClient)
    assert client.get_evaluators_client() is first


def test_inference_client_requires_a_connector() -> None:
    client = ElasticEvalsClient.local(_config())

    with pytest.raises(ValueError, match="connector"):
        client.get_inference_client()


def test_inference_client_prefers_evaluator_connector() -> None:
    client = ElasticEvalsClient(_config(connector_id="task", evaluator_connector_id="judge"))

    assert client.get_inference_client().connector_id == "judge"
