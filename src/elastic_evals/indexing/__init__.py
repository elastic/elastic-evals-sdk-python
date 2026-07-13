"""Elasticsearch indexing client for knowledge-base corpora."""

from elastic_evals.indexing.client import (
    ElasticsearchClient,
    get_elasticsearch_client,
)

__all__ = [
    "ElasticsearchClient",
    "get_elasticsearch_client",
]
