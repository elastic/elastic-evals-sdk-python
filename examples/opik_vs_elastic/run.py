from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

import pandas as pd
from dotenv import load_dotenv

from elastic_evals.agent_builder import (
    AgentBuilderClient,
    AgentConfiguration,
    CreateAgentRequest,
    CreateToolRequest,
    IndexSearchToolConfig,
    ToolSelection,
    build_agent_builder_headers
)
from elastic_evals.api.datasets_models import UpsertDatasetExamplePayload
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.evaluators.correctness import (
    create_correctness_analysis_evaluator,
    create_quantitative_correctness_evaluators,
)
from elastic_evals.evaluators.groundedness import (
    create_groundedness_analysis_evaluator,
    create_quantitative_groundedness_evaluator,
)
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.indexing import get_elasticsearch_client
from elastic_evals.types import EvaluationDataset, EvaluatorParams, Example 

sys.path.insert(0, str(Path(__file__).parents[4] / "orca-framework" / "orca" / "src"))
from orca.clients.agent_builder_client import get_agent_builder_client


INDEX_NAME = "wix_knowledge_base"
MAPPINGS_PATH = Path(__file__).parent / "configs" / "wix_knowledge_base_mappings.json"
ENV_PATH = Path(__file__).parent / ".env"
GROUND_TRUTH_COLUMN = "gt_customer_support_wix_knowledge_base"


async def agent_builder_task(
    example: Example, config: ElasticEvalsConfig, agent_id: str | None = None
) -> dict[str, Any]:
    """Call Agent Builder API and return response."""

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.kibana_url}/api/agent_builder/converse",
            json={
                "connector_id": config.connector_id,
                "agent_id": config.agent_id if agent_id is None else agent_id,
                "input": example.input.get("input_question"),
            },
            headers=build_agent_builder_headers(config.kibana_api_key)
            #{
            #    "kbn-xsrf": "true",
            #    "elastic-api-version": "2023-10-31",
            #},
        )
        response.raise_for_status()
        data = response.json()

    response_payload = data.get("response", {})
    message = response_payload.get("message")
    return {
        "messages": [{"message": message}] if message is not None else [],
        "steps": data.get("steps", []),
        "traceId": data.get("trace_id") or data.get("traceId"),
        "conversation_id": data.get("conversation_id"),
    }


async def main() -> None:
    print("Loading environment variables from .env...")
    load_dotenv(ENV_PATH)  

    # (1) Prepare data:
    print("\nLoading QA pairs from GCS (wix_qa dataset)...")
    qa_wix = pd.read_csv("gs://agent-builder-data-science-datasets/queries/wix_qa.csv")
    print(f"Dataset shape: {qa_wix.shape}")  # output: (52, 4)
    # print(qa_wix.head())

    print("\nLoading knowledge base from GCS (wix_knowledge_base dataset)...")
    kb_wix = pd.read_csv(
        "gs://agent-builder-data-science-datasets/knowledge_bases/cleaned/customer_support/wix_knowledge_base/wix_knowledge_base.csv"
    )
    print(f"Dataset shape: {kb_wix.shape}")  # output: (6222, 4)
    #print(kb_wix.head())

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
    print("\nInitializing ElasticEvalsClient with the config...")
    elastic_evals_client = ElasticEvalsClient(config)

    # (5) Create dataset:
    print("\nLoading dataset into elastic-evals...")
    examples = [
        UpsertDatasetExamplePayload(
            id=row["meta_query_id"],
            input={"input_question": row["input_question"]},
            output={"output_expected": row["output_expected"]},
            metadata={
                "gt_customer_support_wix_knowledge_base": row["gt_customer_support_wix_knowledge_base"],
                "meta_query_id": row["meta_query_id"],
            }
        )
        for _, row in qa_wix.iterrows()
    ]

    examples_dataset = EvaluationDataset(
        name="wix_qa_smoke",
        description="Small sample of WixQA golden Q&A pairs for smoke-testing the pipeline.",
        examples=[Example(
            input={"input_question": row["input_question"]},
            output={"output_expected": row["output_expected"]},
            metadata={
                "gt_customer_support_wix_knowledge_base": row["gt_customer_support_wix_knowledge_base"],
                "meta_query_id": row["meta_query_id"],
            }
        ) for _, row in qa_wix.iterrows() if _ < 10]  # take only the first 10 examples for a quick iteration
    )

    # [IMPORTANT] NOTE: if the dataset doesn't exist, then it creates a new one with the given name. If the dataset already exists, 
    # then it's going to cross-check examples and fully rewrite/update accordingly. Example of a response after sending 
    # the request for the same dataset, but with only the first 5 examples: 
    # UpsertDatasetResponse(dataset_id='0b5ee7b6-9f4a-5c66-b196-6b8cc5154eec', added=0, removed=47, unchanged=5)
    upsert_dataset_response = await elastic_evals_client._datasets_client.upsert(
        name="wix_qa_smoke",
        description="Small sample of WixQA golden Q&A pairs for smoke-testing the pipeline.",
        examples=examples
    )
    print(f"Dataset upserted:\n\tid: {upsert_dataset_response.dataset_id}\n\tadded: {upsert_dataset_response.added}\n\tunchanged: {upsert_dataset_response.unchanged}\n\tremoved: {upsert_dataset_response.removed}")

    print("Sanity check through direct retrieval of the dataset just created...")
    get_dataset_response = await elastic_evals_client._datasets_client.get(
        dataset_id=upsert_dataset_response.dataset_id
    )
    print(f"Retrieved dataset: {get_dataset_response.name} with {len(get_dataset_response.examples)} examples")
    
    # NOTE: as an alternative, I can call .run_experiment() directly and pass the dataset, which is going to be uploaded.

    print("\nGetting inference client from ElasticEvalsClient...")
    inference_client = elastic_evals_client.get_inference_client()  # to be used later for llm as a judge...
    log = config.logger

    # (6) Setup agent builder:
    print("\nSetting up Agent Builder client...")
    ab_client = AgentBuilderClient(
        kibana_url=os.environ.get("KIBANA_URL", "http://localhost:5601"),
        api_key=os.environ.get("KIBANA_API_KEY"),
    )
    print("Creating tool for searching the knowledge base...")
    tool = await ab_client.get_or_create_tool(
        CreateToolRequest(
            id="wix-knowledge-search",
            type="index_search",
            description="Search the Wix knowledge base articles.",
            configuration=IndexSearchToolConfig(pattern=INDEX_NAME),
        )
    )
    print("Creating agent")
    agent = await ab_client.get_or_create_agent(
        CreateAgentRequest(
            id="wix-eval-agent",
            name="Wix Agent",
            description="Agent for wix_qa retrieval documents.",
            configuration=AgentConfiguration(
                instructions="Answer questions using the Wix knowledge base.",
                tools=[ToolSelection(tool_ids=[tool.id])],
            ),
        )
    )

    print(f"Agent ready: {agent.id} (tool: {tool.id} -> index: {INDEX_NAME})")
    print(
        f"Agent created — verify in UI: {os.environ['KIBANA_URL']}/app/agent_builder/manage/agents/{agent.id}"
    )

    # NOTE: if an agent is created during the script, the configs can be updated on the running to avoid using the default AgentBuilder:
    # config.agent_id = agent.id  # set the agent_id in the config for later use in the task function

    # Check if original orca AgentBuilderClient implementation works: 
    #another_agent = get_agent_builder_client(
    #    agent_id=agent.id,
    #    connector_id=config.connector_id,
    #)

    # (6) Create evaluators:
    correctness_analysis = create_correctness_analysis_evaluator(
        inference_client=inference_client, log=log
    )
    groundedness_analysis = create_groundedness_analysis_evaluator(
        inference_client=inference_client, log=log
    )
    quantitative_evaluators = [
        *create_quantitative_correctness_evaluators(),
        create_quantitative_groundedness_evaluator(),
    ]

    # (7) Create custom evaluators: 

    # (8) Run experiment + evals: 
    async def task(example: Example):
        response = await agent_builder_task(example, config)
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
        evaluators=quantitative_evaluators,
    )

    print("\n\n#### Finished kbn/evals loop. ####\n\n")


if __name__ == "__main__":
    asyncio.run(main())