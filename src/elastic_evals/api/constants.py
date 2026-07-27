# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Constants for Kibana evals internal APIs."""

from __future__ import annotations

import uuid

EVALS_INTERNAL_URL = "/internal/evals"
EVALS_SCORES_URL = f"{EVALS_INTERNAL_URL}/scores"
EVALS_DATASETS_URL = f"{EVALS_INTERNAL_URL}/datasets"
EVALS_DATASET_URL = f"{EVALS_DATASETS_URL}/{{dataset_id}}"
EVALS_DATASET_UPSERT_URL = f"{EVALS_DATASETS_URL}/_upsert"
EVALS_EVALUATORS_URL = f"{EVALS_INTERNAL_URL}/evaluators"
EVALS_VALIDATE_URL = f"{EVALS_EVALUATORS_URL}/_validate"
EVALS_EVALUATE_URL = f"{EVALS_INTERNAL_URL}/_evaluate"
EVALS_RESOLVE_INSTRUMENTATION_URL = f"{EVALS_INTERNAL_URL}/traces/_resolve_instrumentation"

DATASET_UUID_NAMESPACE = uuid.UUID("f77b3ee3-7bc6-4bf8-9e43-d7fca9e69ae0")

MAX_INGEST_BATCH_SIZE = 1000
SCORE_INGEST_PAYLOAD_CAP_BYTES = 5 * 1024 * 1024
MAX_EXAMPLES_PER_DATASET = 10_000
MAX_EVALUATORS_PER_REQUEST = 20

EVALS_API_VERSION = "1"
