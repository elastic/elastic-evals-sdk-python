"""Client for creating indices and bulk-ingesting documents into Elasticsearch."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def _clean(value: Any) -> Any:
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_actions(
    index_name: str,
    records: Iterable[Mapping[str, Any]],
    id_field: str | None,
) -> Iterator[dict[str, Any]]:
    for record in records:
        source = {key: _clean(value) for key, value in record.items()}
        action: dict[str, Any] = {"_index": index_name, "_source": source}
        if id_field:
            doc_id = source.pop(id_field, None)
            if doc_id is not None:
                action["_id"] = str(doc_id)
        yield action


class ElasticsearchClient:
    def __init__(self, es: Elasticsearch) -> None:
        self.es = es

    def create_index(
        self,
        index_name: str,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        recreate: bool = False,
    ) -> bool:
        """Create an index, returning True if created and False if it already exists."""
        if recreate and self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        if self.es.indices.exists(index=index_name):
            return False
        self.es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return True

    def index_records(
        self,
        index_name: str,
        records: Iterable[Mapping[str, Any]],
        id_field: str | None = None,
        batch_size: int = 500,
    ) -> tuple[int, list[Any]]:
        """Bulk-ingest records, using id_field as the document _id when present."""
        success, errors = bulk(
            self.es,
            _to_actions(index_name, records, id_field),
            chunk_size=batch_size,
            raise_on_error=False,
        )
        return success, errors if isinstance(errors, list) else []


def get_elasticsearch_client() -> ElasticsearchClient:
    """Build a client from CLOUD_ID/ELASTICSEARCH_API_KEY or ES_URL/ES_HOST env vars."""
    cloud_id = os.getenv("CLOUD_ID")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    host = (
        os.getenv("ES_URL")
        or os.getenv("ELASTICSEARCH_ENDPOINT")
        or os.getenv("ES_HOST")
    )

    if cloud_id and api_key:
        es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)
    elif host and api_key:
        es = Elasticsearch(hosts=[host], api_key=api_key)
    elif host:
        es = Elasticsearch(hosts=[host])
    else:
        raise ValueError(
            "Set CLOUD_ID + ELASTICSEARCH_API_KEY, or ES_URL/ES_HOST, to connect to Elasticsearch."
        )

    return ElasticsearchClient(es=es)
