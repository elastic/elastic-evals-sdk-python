# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

import logging

import httpx
import pytest

from elastic_evals.api.errors import KibanaEvaluatorsError
from elastic_evals.api.response import raise_kibana_error


def test_raise_kibana_error_parses_json_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger(__name__)

    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        raise_kibana_error(
            httpx.Response(503, json={"message": "temporarily unavailable"}),
            error_cls=KibanaEvaluatorsError,
            context="Kibana evaluators request",
            logger=logger,
        )

    assert exc_info.value.message == (
        'Kibana evaluators request failed with 503: {"message": "temporarily unavailable"}'
    )
    assert exc_info.value.status_code == 503
    assert exc_info.value.body == {"message": "temporarily unavailable"}
    assert exc_info.value.retryable is True
    assert exc_info.value.message in caplog.text


def test_raise_kibana_error_preserves_text_body() -> None:
    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        raise_kibana_error(
            httpx.Response(400, text="invalid request"),
            error_cls=KibanaEvaluatorsError,
            context="Kibana evaluators request",
        )

    assert exc_info.value.message == "Kibana evaluators request failed with 400: invalid request"
    assert exc_info.value.body == "invalid request"
    assert exc_info.value.retryable is False


def test_raise_kibana_error_includes_operation() -> None:
    with pytest.raises(KibanaEvaluatorsError) as exc_info:
        raise_kibana_error(
            httpx.Response(400, text="invalid request"),
            error_cls=KibanaEvaluatorsError,
            context="Kibana evaluators request",
            operation="evaluate",
        )

    assert exc_info.value.message == "Kibana evaluators request failed (evaluate) with 400: invalid request"
