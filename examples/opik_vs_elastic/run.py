# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from elastic_evals.agent_builder import (
    AgentBuilderClient,
    AgentConfiguration,
    CreateAgentRequest,
    CreateToolRequest,
    IndexSearchToolConfig,
    ToolSelection,
    build_agent_builder_headers,
)
from elastic_evals.api.datasets_models import UpsertDatasetExamplePayload
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.correctness import (
    create_correctness_analysis_evaluator,
    create_quantitative_correctness_evaluators,
)
from elastic_evals.evaluators.criteria import create_criteria_evaluator
from elastic_evals.evaluators.groundedness import (
    create_groundedness_analysis_evaluator,
    create_quantitative_groundedness_evaluator,
)
from elastic_evals.evaluators.input_tokens import create_input_tokens_evaluator
from elastic_evals.evaluators.latency import create_latency_evaluator
from elastic_evals.evaluators.output_tokens import create_output_tokens_evaluator
from elastic_evals.evaluators.tool_calls import create_tool_calls_evaluator
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.tracing import init_tracing
from elastic_evals.types import (
    EvaluationDataset,
    EvaluationResult,
    Evaluator,
    EvaluatorParams,
    Example,
)
from examples.opik_vs_elastic.helpers.data import load_wix_data, select_qa_examples
from examples.opik_vs_elastic.helpers.helpers import (
    AGENT_ID,
    AGENT_INSTRUCTIONS,
    AGENT_NAME,
    ENV_PATH,
    GROUND_TRUTH_COLUMN,
    INDEX_NAME,
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_ID,
    _extract_retrieved_doc_ids,  # noqa: PLC2701
    _to_string_list,  # noqa: PLC2701
)
from examples.opik_vs_elastic.helpers.indexing import get_elasticsearch_client

USE_ENTIRE_DATASET = False
DATASET_SAMPLE_SIZE = 10
USE_GCP = False

DATASET_NAME = "wix_qa_managed_workflow"
DATASET_DESCRIPTION = "WixQA golden Q&A pairs for the managed evaluation workflow."

WIX_RESPONSE_CRITERIA = [
    "The response directly addresses the user's Wix support question.",
    "The response provides clear and actionable guidance.",
]


def create_document_recall_evaluator(*, tool_id: str) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        expected = _to_string_list((params.metadata or {}).get("relevant_doc_ids"))
        retrieved = _extract_retrieved_doc_ids(params.output, tool_id=tool_id)

        if not expected:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation="No relevant document IDs available",
                metadata={
                    "expected_document_ids": expected,
                    "retrieved_document_ids": retrieved,
                },
            )

        retrieved_set = set(retrieved)
        matched = [document_id for document_id in expected if document_id in retrieved_set]
        missing = [document_id for document_id in expected if document_id not in retrieved_set]
        score = len(matched) / len(expected)
        label = "PASS" if score == 1.0 else "PARTIAL" if score > 0 else "FAIL"

        return EvaluationResult(
            score=score,
            label=label,
            metadata={
                "expected_document_ids": expected,
                "retrieved_document_ids": retrieved,
                "matched_document_ids": matched,
                "missing_document_ids": missing,
            },
        )

    return SimpleEvaluator(name="DocumentRecall", kind="CODE", evaluate=evaluate)


async def agent_builder_task(
    example: Example, config: ElasticEvalsConfig, agent_id: str | None = None
) -> dict[str, Any]:
    """Call Agent Builder API and return response."""

    request_body: dict[str, Any] = {
        "connector_id": config.connector_id,
        "input": example.input.get("question"),
    }
    if agent_id is not None:
        request_body["agent_id"] = agent_id

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.kibana_url}/api/agent_builder/converse",
            json=request_body,
            headers=build_agent_builder_headers(config.kibana_api_key),
        )

        response.raise_for_status()
        data = response.json()

    response_payload = data.get("response", {})
    message = response_payload.get("message")
    raw_trace_id = data.get("trace_id") or data.get("traceId")
    if isinstance(raw_trace_id, list):
        trace_id = next((value for value in raw_trace_id if isinstance(value, str)), None)
    else:
        trace_id = raw_trace_id if isinstance(raw_trace_id, str) else None

    task_output = {
        "messages": [{"message": message}] if message is not None else [],
        "steps": data.get("steps", []),
        "traceId": trace_id,
        "conversation_id": data.get("conversation_id"),
    }

    if trace_id:
        task_output["_interaction_trace_id"] = trace_id
    return task_output


async def main() -> None:
    print("Loading environment variables from .env...")
    load_dotenv(ENV_PATH)

    # (1) Prepare data:
    source = "GCS" if USE_GCP else "Hugging Face"
    print(f"\nLoading WixQA data from {source}...")
    all_qa_wix, kb_wix = load_wix_data(use_gcp=USE_GCP)
    qa_wix = select_qa_examples(
        all_qa_wix,
        use_entire_dataset=USE_ENTIRE_DATASET,
        sample_size=DATASET_SAMPLE_SIZE,
    )
    print(f"Selected {len(qa_wix)} of {len(all_qa_wix)} QA examples; knowledge base contains {len(kb_wix)} documents")

    # (2) Indexing the knowledge base into elasticsearch (if not indexed already):
    print("Indexing knowledge base into Elasticsearch...")
    es_client = get_elasticsearch_client()
    es_client.create_index(INDEX_NAME, mappings=None, recreate=True)
    indexed, errors = es_client.index_records(
        INDEX_NAME,
        kb_wix.to_dict(orient="records"),
        id_field="id",
    )
    print(f"Indexed {indexed} docs into '{INDEX_NAME}' ({len(errors)} errors)")

    # (3) Create suite: - not implemented for now and probably not applicable / not needed

    # (4) Setup elastic-evals-client:
    print("\nBuilding configurations from env vars and storing them into an ElasticEvalsConfig instance")
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)
    print("\nInitializing ElasticEvalsClient with the config...")
    elastic_evals_client = ElasticEvalsClient(config)

    # (5) Create dataset:
    print("\nLoading dataset into elastic-evals...")
    examples = [
        UpsertDatasetExamplePayload(
            id=row["meta_query_id"],
            input={"question": row["input_question"]},
            output={"expected": row["output_expected"]},
            metadata={
                GROUND_TRUTH_COLUMN: row[GROUND_TRUTH_COLUMN],
                "relevant_doc_ids": row["relevant_doc_ids"],
                "meta_query_id": row["meta_query_id"],
            },
        )
        for _, row in qa_wix.iterrows()
    ]

    examples_dataset = EvaluationDataset(
        name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
        examples=[
            Example(
                input={"question": row["input_question"]},
                output={"expected": row["output_expected"]},
                metadata={
                    GROUND_TRUTH_COLUMN: row[GROUND_TRUTH_COLUMN],
                    "relevant_doc_ids": row["relevant_doc_ids"],
                    "meta_query_id": row["meta_query_id"],
                },
            )
            for _, row in qa_wix.iterrows()
        ],
    )

    # [IMPORTANT] NOTE: if the dataset doesn't exist, then it creates a new one with the given name. If the dataset already exists,
    # then it's going to cross-check examples and fully rewrite/update accordingly. Example of a response after sending
    # the request for the same dataset, but with only the first 5 examples:
    # UpsertDatasetResponse(dataset_id='0b5ee7b6-9f4a-5c66-b196-6b8cc5154eec', added=0, removed=47, unchanged=5)

    upsert_dataset_response = await elastic_evals_client._datasets_client.upsert(
        name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
        examples=examples,
    )
    print(
        f"Dataset upserted:\n\tid: {upsert_dataset_response.dataset_id}\n\tadded: {upsert_dataset_response.added}\n\tunchanged: {upsert_dataset_response.unchanged}\n\tremoved: {upsert_dataset_response.removed}"
    )

    print("Sanity check through direct retrieval of the dataset just created...")
    get_dataset_response = await elastic_evals_client._datasets_client.get(
        dataset_id=upsert_dataset_response.dataset_id
    )
    print(f"Retrieved dataset: {get_dataset_response.name} with {len(get_dataset_response.examples)} examples")

    # NOTE: as an alternative, I can call .run_experiment() directly and pass the dataset, which is going to be uploaded.
    print("\nGetting inference client from ElasticEvalsClient...")
    inference_client = elastic_evals_client.get_inference_client()  # to be used later for llm as a judge...
    trace_client = elastic_evals_client.get_trace_client()
    log = config.logger

    # (6) Setup agent builder:
    print("\nSetting up Agent Builder client...")
    ab_client = AgentBuilderClient(
        kibana_url=os.environ.get("KIBANA_URL", "http://localhost:5601"),
        api_key=os.environ.get("KIBANA_API_KEY"),
    )
    print("Creating tool for searching the knowledge base...")
    tool = await ab_client.create_tool(
        CreateToolRequest(
            id=SEARCH_TOOL_ID,
            type="index_search",
            description=SEARCH_TOOL_DESCRIPTION,
            configuration=IndexSearchToolConfig(pattern=INDEX_NAME),
        ),
        update_if_exists=True,
    )
    print("Creating agent")
    agent = await ab_client.create_agent(
        CreateAgentRequest(
            id=AGENT_ID,
            name=AGENT_NAME,
            description="Agent for wix_qa retrieval documents.",
            configuration=AgentConfiguration(
                instructions=AGENT_INSTRUCTIONS,
                tools=[ToolSelection(tool_ids=[tool.id])],
            ),
        ),
        update_if_exists=True,
    )

    print(f"Agent ready: {agent.id} (tool: {tool.id} -> index: {INDEX_NAME})")
    print(f"Agent created — verify in UI: {os.environ['KIBANA_URL']}/app/agent_builder/manage/agents/{agent.id}")

    # (7) Create built-in evaluators:
    correctness_analysis = create_correctness_analysis_evaluator(inference_client=inference_client, log=log)
    groundedness_analysis = create_groundedness_analysis_evaluator(inference_client=inference_client, log=log)

    criteria_evaluator = create_criteria_evaluator(
        inference_client=inference_client,
        criteria=WIX_RESPONSE_CRITERIA,
        log=log,
    )

    # (8) Create custom evaluator:
    document_recall_evaluator = create_document_recall_evaluator(tool_id=SEARCH_TOOL_ID)

    evaluators = [
        *create_quantitative_correctness_evaluators(),
        create_quantitative_groundedness_evaluator(),
        criteria_evaluator,
        create_latency_evaluator(trace_client=trace_client, log=log),
        create_input_tokens_evaluator(trace_client=trace_client, log=log),
        create_output_tokens_evaluator(trace_client=trace_client, log=log),
        create_tool_calls_evaluator(trace_client=trace_client, log=log),
        document_recall_evaluator,
    ]

    # (9) Run experiment + evals:
    async def task(example: Example):
        response = await agent_builder_task(example, config, agent_id=agent.id)
        params = EvaluatorParams(
            input=example.input,
            output=response,
            expected=example.output,
            metadata=example.metadata,
        )
        correctness_result = await correctness_analysis.evaluate(params)
        groundedness_result = await groundedness_analysis.evaluate(params)
        response["correctnessAnalysis"] = correctness_result.metadata
        response["groundednessAnalysis"] = groundedness_result.metadata
        return response

    await elastic_evals_client.run_experiment(
        dataset=examples_dataset,
        task=task,
        evaluators=evaluators,
        experiment_name="Wix QA - managed run_experiment workflow",
    )

    print("\n\n#### Finished kbn/evals loop. ####\n\n")


if __name__ == "__main__":
    asyncio.run(main())
