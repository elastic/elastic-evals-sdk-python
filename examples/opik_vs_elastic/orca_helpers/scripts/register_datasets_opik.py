# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Register the full Wix QA golden dataset into Opik.

Mirrors orca_helpers's dataset-registration pattern (see
orca_helpers-framework/orca_helpers/src/orca_helpers/evaluation/datasets/registration.py and
transformers.py): read the golden CSV from GCS, derive ``relevant_doc_ids``
from the ground-truth column, then create/insert into an Opik dataset.

For a smaller dataset used to smoke-test the pipeline, see run.py's step 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# examples/opik_vs_elastic/orca_helpers/ holds opik_client.py; add it to sys.path and
# import it by bare module name rather than as `orca_helpers.opik_client` — `orca_helpers` is
# also the name of the real orca_helpers package, and importing `orca_helpers.<anything>` here
# would risk resolving to this local example folder instead of the installed
# orca_helpers package (or vice versa).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opik_client import extract_relevant_doc_ids, get_opik_datasets_client  # noqa: E402

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
DATASET_NAME = "wix_qa"
SOURCE_CSV = "gs://agent-builder-data-science-datasets/queries/wix_qa.csv"
GROUND_TRUTH_COLUMN = "gt_customer_support_wix_knowledge_base"


def main() -> None:
    load_dotenv(ENV_PATH)

    client = get_opik_datasets_client()

    print("Existing Opik datasets:")
    for existing in client.list_datasets():
        print(f"  - {existing.name}: {client.dataset_url(existing)}")

    df = pd.read_csv(SOURCE_CSV)
    df = extract_relevant_doc_ids(df, GROUND_TRUTH_COLUMN)

    dataset = client.get_or_create_dataset(
        name=DATASET_NAME,
        description="WixQA golden Q&A pairs.",
    )
    client.add_rows(
        dataset,
        df,
        keys_mapping={"input_question": "input", "output_expected": "expected_output"},
        ignore_keys=[GROUND_TRUTH_COLUMN, "NOT_relevant_doc_ids"],
    )
    print(f"\nInserted {len(df)} rows into dataset '{DATASET_NAME}'.")

    sample = client.get_rows(dataset, nb_samples=3)
    print(f"\nSample rows from '{DATASET_NAME}':")
    for row in sample:
        print(f"  {row}")

    print(f"\nView dataset: {client.dataset_url(dataset)}")
    print(f"View project: {client.project_url()}")


if __name__ == "__main__":
    main()
