"""Index the Wix knowledge base into Elasticsearch using orca's real pipeline.

This mirrors orca's own ``orca.etl.indexing.cli`` so it runs the exact code the
orca team uses: ``load_dataset_config`` -> ``get_elasticsearch_client`` ->
``index_dataset``, reading the cleaned corpus from GCS. Compare against
``run.py``'s step 2, which indexes the same corpus with the SDK-native
Elasticsearch client instead of orca's pipeline.

Prerequisites:
- Install orca into this environment, e.g.:
    uv pip install -e ../../../../orca-framework/orca
- Elasticsearch auth: set CLOUD_ID + ELASTICSEARCH_API_KEY (or ES_HOST).
- GCS auth for gcsfs (application default credentials).
- The semantic_text mappings require the referenced inference endpoints
  (ELSER and ``.jina-embeddings-v3``) to exist in the cluster.

Run:
    python examples/opik_vs_elastic/orca/scripts/index_kb_with_orca.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from orca.clients.elasticsearch_client import get_elasticsearch_client
from orca.config_parser import load_dataset_config
from orca.etl.indexing.indexer import index_dataset
from orca.etl.indexing.types import IndexingConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "wix_knowledge_base.yml"
GCS_ROOT = "gs://agent-builder-data-science-datasets/knowledge_bases/cleaned"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def main() -> int:
    load_dotenv(ENV_PATH)

    dataset_config = load_dataset_config(CONFIG_PATH)

    es_client = get_elasticsearch_client()
    if es_client is None:
        raise RuntimeError(
            "Could not connect to Elasticsearch. Set CLOUD_ID + ELASTICSEARCH_API_KEY "
            "or ES_HOST."
        )

    indexing_config = IndexingConfig(gcs_root=GCS_ROOT, recreate_indices=True)

    summary = index_dataset(dataset_config, indexing_config, es_client)
    print(summary.to_dict())

    return 0 if summary.files_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
