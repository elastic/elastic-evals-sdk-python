"""Constants for Kibana evals internal APIs."""

from __future__ import annotations

import uuid

EVALS_INTERNAL_URL = "/internal/evals"
EVALS_SCORES_URL = "/internal/evals/scores"
EVALS_DATASETS_URL = "/internal/evals/datasets"
EVALS_DATASET_URL = "/internal/evals/datasets/{dataset_id}"
EVALS_DATASET_UPSERT_URL = "/internal/evals/datasets/_upsert"

DATASET_UUID_NAMESPACE = uuid.UUID("f77b3ee3-7bc6-4bf8-9e43-d7fca9e69ae0")

MAX_INGEST_BATCH_SIZE = 1000
SCORE_INGEST_PAYLOAD_CAP_BYTES = 5 * 1024 * 1024
MAX_EXAMPLES_PER_DATASET = 10_000

EVALS_API_VERSION = "1"
