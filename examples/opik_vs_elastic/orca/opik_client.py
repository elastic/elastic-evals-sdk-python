"""Opik dataset client, mirroring and extending orca's opik_client.py.

orca's ``clients/opik_client.py`` only wraps ``get_dataset`` and
``get_or_create_dataset``. This module adds the remaining dataset operations
that the Opik SDK already supports (list, add rows, retrieve rows, URLs) but
that orca's wrapper does not expose.
"""

from __future__ import annotations

import ast
from typing import Any

import opik
import pandas as pd
from opik import url_helpers
from opik.api_objects.dataset.dataset import Dataset, DatasetVersion
from opik.config import OpikConfig


class OpikDatasetsClient:
    """Thin wrapper around the Opik SDK client for dataset management.

    Args:
        sdk_client: An ``opik.Opik`` instance (injected for testing).
    """

    def __init__(self, sdk_client: opik.Opik) -> None:
        self._sdk_client = sdk_client

    def get_dataset(
        self,
        name: str,
        version: str | None = None,
    ) -> Dataset | DatasetVersion:
        """Retrieve an existing Opik dataset, optionally pinned to a version."""
        dataset = self._sdk_client.get_dataset(name=name)
        if version:
            return dataset.get_version_view(version)
        return dataset

    def get_or_create_dataset(
        self,
        name: str,
        description: str | None = None,
        project_name: str | None = None,
    ) -> Dataset:
        """Retrieve an existing Opik dataset or create it if absent."""
        return self._sdk_client.get_or_create_dataset(
            name=name,
            description=description,
            project_name=project_name,
        )

    def list_datasets(
        self,
        project_name: str | None = None,
        max_results: int = 100,
    ) -> list[Dataset]:
        """List datasets, up to ``max_results``.

        Falls back to the client's default project when ``project_name`` is
        omitted.
        """
        return self._sdk_client.get_datasets(
            max_results=max_results, project_name=project_name
        )

    def add_rows(
        self,
        dataset: Dataset,
        df: pd.DataFrame,
        keys_mapping: dict[str, str] | None = None,
        ignore_keys: list[str] | None = None,
    ) -> None:
        """Insert dataframe rows into a dataset, renaming/skipping columns."""
        dataset.insert_from_pandas(
            df, keys_mapping=keys_mapping, ignore_keys=ignore_keys
        )

    def get_rows(
        self,
        dataset: Dataset,
        nb_samples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve dataset rows, optionally limited to ``nb_samples``."""
        return dataset.get_items(nb_samples=nb_samples)

    def dataset_url(self, dataset: Dataset) -> str:
        """Build a UI URL for viewing this dataset in Opik."""
        return url_helpers.get_dataset_url_by_id(
            dataset_id=dataset.id, url_override=OpikConfig().url_override
        )

    def project_url(self, project_name: str | None = None) -> str:
        """Build a UI URL for viewing a project in Opik."""
        return self._sdk_client.get_project_url(project_name=project_name)


def get_opik_datasets_client(project_name: str | None = None) -> OpikDatasetsClient:
    """Instantiate an :class:`OpikDatasetsClient` using default SDK configuration.

    Opik reads its configuration (URL, API key, workspace) from environment
    variables or ``~/.opik.config``.

    Args:
        project_name: Opik project name to associate the client with. When
            ``None``, Opik falls back to the ``OPIK_PROJECT_NAME`` environment
            variable, or the default project.

    Returns:
        A ready-to-use :class:`OpikDatasetsClient`.
    """
    sdk_client = opik.Opik(project_name=project_name)
    return OpikDatasetsClient(sdk_client=sdk_client)


def extract_relevant_doc_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Split a stringified ``{doc_id: 0|1}`` column into relevant/non-relevant IDs.

    Mirrors orca's ``transformers._extract_dict_keys``: parses ``source`` with
    :func:`ast.literal_eval` and splits its keys into ``relevant_doc_ids``
    (truthy values) and ``NOT_relevant_doc_ids`` (falsy values).

    Args:
        df: Input dataframe.
        source: Name of the column containing stringified ``{doc_id: 0|1}`` dicts.

    Returns:
        A copy of ``df`` with ``relevant_doc_ids`` and ``NOT_relevant_doc_ids`` added.
    """

    def _split(value: object) -> tuple[list[str], list[str]]:
        if pd.isna(value):  # type: ignore[arg-type]
            return [], []
        parsed: dict[str, Any] = ast.literal_eval(str(value))
        relevant = [doc_id for doc_id, flag in parsed.items() if flag]
        non_relevant = [doc_id for doc_id, flag in parsed.items() if not flag]
        return relevant, non_relevant

    df = df.copy()
    splits = df[source].apply(_split)
    df["relevant_doc_ids"] = splits.apply(lambda pair: pair[0])
    df["NOT_relevant_doc_ids"] = splits.apply(lambda pair: pair[1])
    return df
