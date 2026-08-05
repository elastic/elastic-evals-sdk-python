# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Input token evaluator."""

from __future__ import annotations

import logging

from elastic_evals.api import InstrumentationProfile, KibanaEvaluatorsClient
from elastic_evals.evaluators.kibana import KibanaEvaluatorConfig, kibana_evaluators
from elastic_evals.types import Evaluator


def create_input_tokens_evaluator(
    *,
    client: KibanaEvaluatorsClient,
    instrumentation_profile: InstrumentationProfile = "elastic-inference",
    log: logging.Logger | None = None,
) -> Evaluator:
    return kibana_evaluators(
        [KibanaEvaluatorConfig(name="input_tokens", kind="CODE")],
        client=client,
        instrumentation_profile=instrumentation_profile,
        log=log,
    )[0]
