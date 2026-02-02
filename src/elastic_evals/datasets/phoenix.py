"""Phoenix dataset loader for elastic-evals.

This module provides functions to load datasets from Arize Phoenix
for use in evaluations.

Requires the phoenix optional dependency:
    pip install elastic-evals[phoenix]
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from elastic_evals.types import EvaluationDataset, Example


class PhoenixDatasetConfig(BaseModel):
    """Configuration for connecting to Phoenix to load datasets."""

    base_url: str = "http://localhost:6006"
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "PhoenixDatasetConfig":
        """Create config from environment variables.

        Environment variables:
        - PHOENIX_BASE_URL or PHOENIX_COLLECTOR_ENDPOINT: Phoenix server URL
        - PHOENIX_API_KEY: API key for Phoenix Cloud
        """
        base_url = os.environ.get(
            "PHOENIX_BASE_URL",
            os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
        )
        api_key = os.environ.get("PHOENIX_API_KEY")
        return cls(base_url=base_url, api_key=api_key)


def _get_phoenix_client(config: PhoenixDatasetConfig | None = None):
    """Get a Phoenix client instance."""
    try:
        from phoenix.client import Client
    except ImportError as e:
        raise ImportError(
            "Phoenix client is required to load datasets from Phoenix. "
            "Install with: pip install elastic-evals[phoenix]"
        ) from e

    config = config or PhoenixDatasetConfig.from_env()
    return Client(base_url=config.base_url, api_key=config.api_key)


def _get_async_phoenix_client(config: PhoenixDatasetConfig | None = None):
    """Get an async Phoenix client instance."""
    try:
        from phoenix.client import AsyncClient
    except ImportError as e:
        raise ImportError(
            "Phoenix client is required to load datasets from Phoenix. "
            "Install with: pip install elastic-evals[phoenix]"
        ) from e

    config = config or PhoenixDatasetConfig.from_env()
    return AsyncClient(base_url=config.base_url, api_key=config.api_key)


def _convert_phoenix_dataset_to_evaluation_dataset(
    phoenix_dataset: Any,
    description: str | None = None,
) -> EvaluationDataset:
    """Convert a Phoenix dataset to an EvaluationDataset."""
    df = phoenix_dataset.to_dataframe()

    examples = []
    for _, row in df.iterrows():
        # Phoenix datasets have 'input', 'output', 'metadata' columns
        input_data = row.get("input", {})
        output_data = row.get("output")
        metadata = row.get("metadata")

        # Handle cases where input/output/metadata might be stored differently
        if isinstance(input_data, str):
            input_data = {"input": input_data}
        if not isinstance(input_data, dict):
            input_data = {"value": input_data}

        examples.append(
            Example(
                input=input_data,
                output=output_data,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )

    dataset_name = getattr(phoenix_dataset, "name", "phoenix-dataset")
    dataset_description = description or getattr(
        phoenix_dataset, "description", None
    ) or f"Loaded from Phoenix: {dataset_name}"

    return EvaluationDataset(
        name=dataset_name,
        description=dataset_description,
        examples=examples,
    )


def load_dataset_from_phoenix(
    dataset_name: str,
    config: PhoenixDatasetConfig | None = None,
    description: str | None = None,
) -> EvaluationDataset:
    """Load a dataset from Phoenix and convert to EvaluationDataset.

    Args:
        dataset_name: Name of the dataset in Phoenix
        config: Phoenix connection configuration (uses env vars if not provided)
        description: Optional description override for the dataset

    Returns:
        EvaluationDataset ready for use in evaluations

    Example:
        >>> from elastic_evals.datasets import load_dataset_from_phoenix
        >>> dataset = load_dataset_from_phoenix("customer-support-qa")
        >>> result = await client.run_experiment(
        ...     dataset=dataset,
        ...     task=my_task,
        ...     evaluators=my_evaluators,
        ... )
    """
    client = _get_phoenix_client(config)
    phoenix_dataset = client.datasets.get_dataset(dataset=dataset_name)
    return _convert_phoenix_dataset_to_evaluation_dataset(phoenix_dataset, description)


async def load_dataset_from_phoenix_async(
    dataset_name: str,
    config: PhoenixDatasetConfig | None = None,
    description: str | None = None,
) -> EvaluationDataset:
    """Async version of load_dataset_from_phoenix.

    Args:
        dataset_name: Name of the dataset in Phoenix
        config: Phoenix connection configuration (uses env vars if not provided)
        description: Optional description override for the dataset

    Returns:
        EvaluationDataset ready for use in evaluations
    """
    client = _get_async_phoenix_client(config)
    phoenix_dataset = await client.datasets.get_dataset(dataset=dataset_name)
    return _convert_phoenix_dataset_to_evaluation_dataset(phoenix_dataset, description)


def list_phoenix_datasets(
    config: PhoenixDatasetConfig | None = None,
) -> list[dict[str, Any]]:
    """List all datasets available in Phoenix.

    Args:
        config: Phoenix connection configuration (uses env vars if not provided)

    Returns:
        List of dataset metadata dictionaries with keys like 'name', 'example_count', etc.

    Example:
        >>> from elastic_evals.datasets import list_phoenix_datasets
        >>> datasets = list_phoenix_datasets()
        >>> for ds in datasets:
        ...     print(f"{ds['name']}: {ds['example_count']} examples")
    """
    client = _get_phoenix_client(config)
    return list(client.datasets.list())


async def list_phoenix_datasets_async(
    config: PhoenixDatasetConfig | None = None,
) -> list[dict[str, Any]]:
    """Async version of list_phoenix_datasets.

    Args:
        config: Phoenix connection configuration (uses env vars if not provided)

    Returns:
        List of dataset metadata dictionaries
    """
    client = _get_async_phoenix_client(config)
    datasets = await client.datasets.list()
    return list(datasets)
