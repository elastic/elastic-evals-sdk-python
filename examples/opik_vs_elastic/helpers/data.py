# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Load and normalize WixQA data from GCS or Hugging Face."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
from datasets import load_dataset

from examples.opik_vs_elastic.helpers.helpers import (
    GROUND_TRUTH_COLUMN,
    WIX_KNOWLEDGE_BASE_PATH_ENV,
    WIX_QA_DATASET_PATH_ENV,
    _parse_relevant_doc_ids,  # noqa: PLC2701
    get_dataset_path,
)

HUGGING_FACE_DATASET = "Wix/WixQA"
HUGGING_FACE_QA_CONFIG = "wixqa_expertwritten"
HUGGING_FACE_KB_CONFIG = "wix_kb_corpus"


def _require_columns(dataframe: pd.DataFrame, columns: set[str], *, source: str) -> None:
    missing = sorted(columns - set(dataframe.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _to_document_ids(value: Any) -> list[str]:
    if isinstance(value, str) or not isinstance(value, Iterable) or isinstance(value, Mapping):
        return []
    return list(dict.fromkeys(str(item) for item in value if item is not None))


def _load_hugging_face_config(config: str) -> pd.DataFrame:
    dataset = load_dataset(HUGGING_FACE_DATASET, config, split="train")
    return dataset.to_pandas()


def _normalize_gcs_qa(dataframe: pd.DataFrame) -> pd.DataFrame:
    required = {
        "meta_query_id",
        "input_question",
        "output_expected",
        GROUND_TRUTH_COLUMN,
    }
    _require_columns(dataframe, required, source="GCS WixQA dataset")

    normalized = dataframe.copy()
    normalized["relevant_doc_ids"] = normalized[GROUND_TRUTH_COLUMN].apply(_parse_relevant_doc_ids)
    return normalized


def _normalize_hugging_face_qa(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.reset_index(drop=True)
    _require_columns(
        dataframe,
        {"question", "answer", "article_ids"},
        source="Hugging Face WixQA dataset",
    )

    document_ids = dataframe["article_ids"].apply(_to_document_ids)
    return pd.DataFrame(
        {
            "meta_query_id": [f"wixqa_expertwritten_{index + 1}" for index in range(len(dataframe))],
            "input_question": dataframe["question"].astype(str),
            "output_expected": dataframe["answer"].astype(str),
            GROUND_TRUTH_COLUMN: document_ids.apply(lambda ids: {document_id: True for document_id in ids}),
            "relevant_doc_ids": document_ids,
        }
    )


def _normalize_knowledge_base(dataframe: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _require_columns(dataframe, {"id", "contents"}, source=source)
    normalized = dataframe.copy()
    normalized["id"] = normalized["id"].astype(str)
    return normalized


def load_wix_data(*, use_gcp: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WixQA examples and their knowledge-base corpus."""
    if use_gcp:
        qa = _normalize_gcs_qa(pd.read_csv(get_dataset_path(WIX_QA_DATASET_PATH_ENV)))
        knowledge_base = _normalize_knowledge_base(
            pd.read_csv(get_dataset_path(WIX_KNOWLEDGE_BASE_PATH_ENV)),
            source="GCS Wix knowledge base",
        )
    else:
        qa = _normalize_hugging_face_qa(_load_hugging_face_config(HUGGING_FACE_QA_CONFIG))
        knowledge_base = _normalize_knowledge_base(
            _load_hugging_face_config(HUGGING_FACE_KB_CONFIG),
            source="Hugging Face Wix knowledge base",
        )

    return qa, knowledge_base


def select_qa_examples(
    dataframe: pd.DataFrame,
    *,
    use_entire_dataset: bool,
    sample_size: int,
) -> pd.DataFrame:
    """Return the full QA dataset or its first configured examples."""
    if use_entire_dataset:
        return dataframe.reset_index(drop=True).copy()
    if sample_size < 1:
        raise ValueError("DATASET_SAMPLE_SIZE must be at least 1")
    return dataframe.head(sample_size).reset_index(drop=True).copy()
