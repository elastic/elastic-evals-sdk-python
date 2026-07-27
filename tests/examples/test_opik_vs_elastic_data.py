# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("datasets")

from examples.opik_vs_elastic.helpers import data  # noqa: E402


def test_load_wix_data_normalizes_hugging_face_data(monkeypatch: pytest.MonkeyPatch) -> None:
    qa_source = pd.DataFrame(
        {
            "question": ["How do I configure Wix?"],
            "answer": ["Open the site settings."],
            "article_ids": [["doc-1", "doc-2"]],
        }
    )
    knowledge_base_source = pd.DataFrame(
        {
            "id": ["doc-1"],
            "contents": ["Configuration instructions"],
            "article_type": ["article"],
        }
    )

    monkeypatch.setattr(
        data,
        "_load_hugging_face_config",
        lambda config: qa_source if config == data.HUGGING_FACE_QA_CONFIG else knowledge_base_source,
    )

    qa, knowledge_base = data.load_wix_data(use_gcp=False)

    assert qa.to_dict(orient="records") == [
        {
            "meta_query_id": "wixqa_expertwritten_1",
            "input_question": "How do I configure Wix?",
            "output_expected": "Open the site settings.",
            data.GROUND_TRUTH_COLUMN: {"doc-1": True, "doc-2": True},
            "relevant_doc_ids": ["doc-1", "doc-2"],
        }
    ]
    assert knowledge_base.to_dict(orient="records") == [
        {
            "id": "doc-1",
            "contents": "Configuration instructions",
            "article_type": "article",
        }
    ]


def test_load_wix_data_normalizes_gcs_ground_truth(monkeypatch: pytest.MonkeyPatch) -> None:
    qa_source = pd.DataFrame(
        {
            "meta_query_id": ["wix_kb_1"],
            "input_question": ["How do I configure Wix?"],
            "output_expected": ["Open the site settings."],
            data.GROUND_TRUTH_COLUMN: ["{'doc-1': True, 'doc-2': False}"],
        }
    )
    knowledge_base_source = pd.DataFrame(
        {
            "id": ["doc-1"],
            "contents": ["Configuration instructions"],
        }
    )

    monkeypatch.setattr(
        data.pd,
        "read_csv",
        lambda path: qa_source if path == data.WIX_QA_DATASET_PATH else knowledge_base_source,
    )

    qa, knowledge_base = data.load_wix_data(use_gcp=True)

    assert qa["relevant_doc_ids"].tolist() == [["doc-1"]]
    assert knowledge_base["id"].tolist() == ["doc-1"]


def test_select_qa_examples_supports_sample_and_entire_dataset() -> None:
    dataframe = pd.DataFrame({"id": [1, 2, 3]})

    sample = data.select_qa_examples(
        dataframe,
        use_entire_dataset=False,
        sample_size=2,
    )
    entire = data.select_qa_examples(
        dataframe,
        use_entire_dataset=True,
        sample_size=0,
    )

    assert sample["id"].tolist() == [1, 2]
    assert entire["id"].tolist() == [1, 2, 3]

    with pytest.raises(ValueError, match="DATASET_SAMPLE_SIZE must be at least 1"):
        data.select_qa_examples(
            dataframe,
            use_entire_dataset=False,
            sample_size=0,
        )
