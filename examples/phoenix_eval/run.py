"""Example: Load dataset from Phoenix and run comprehensive evaluation.

This example demonstrates:
1. Loading a dataset from Arize Phoenix
2. Running RAG metrics evaluators (Precision@K, Recall@K, F1@K)
3. Running groundedness evaluator (LLM-based)
4. Running trace-based evaluators (latency, tokens, tool calls)

Prerequisites:
- pip install elastic-evals[phoenix] python-dotenv
- Phoenix server running (or Phoenix Cloud credentials)
- Elasticsearch for trace data (for trace-based evaluators)
- Kibana connector configured (for LLM-based evaluators)

Configuration:
    Copy .env.example to .env and fill in your values:

        cp examples/phoenix_eval/.env.example examples/phoenix_eval/.env
        # Edit .env with your configuration

    Or set environment variables directly in your shell.

Example usage:
    # Option 1: Using .env file (recommended)
    cp examples/phoenix_eval/.env.example examples/phoenix_eval/.env
    # Edit .env with your values
    python examples/phoenix_eval/run.py

    # Option 2: Using environment variables
    export CONNECTOR_ID="your-connector-id"
    export KIBANA_AUTH="base64-encoded-auth"
    python examples/phoenix_eval/run.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from elasticsearch import AsyncElasticsearch

from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.datasets import load_dataset_from_phoenix, PhoenixDatasetConfig
from elastic_evals.evaluators.groundedness import (
    create_groundedness_analysis_evaluator,
    create_quantitative_groundedness_evaluator,
)
from elastic_evals.evaluators.rag import (
    create_rag_evaluators,
    RagEvaluatorConfig,
    RetrievedDoc,
)
from elastic_evals.evaluators.trace_based import (
    create_latency_evaluator,
    create_input_tokens_evaluator,
    create_output_tokens_evaluator,
    create_tool_calls_evaluator,
)
from elastic_evals.executor import ElasticEvalsClient
from elastic_evals.export import EvaluationScoreRepository
from elastic_evals.export.documents import ModelInfo
from elastic_evals.export.repository import build_flattened_score_documents
from elastic_evals.inference import KibanaInferenceClient
from elastic_evals.reporting import DefaultReporter
from elastic_evals.tracing import init_tracing
from elastic_evals.types import (
    Evaluator,
    EvaluatorParams,
    Example,
)


def _load_dotenv() -> None:
    """Load environment variables from .env file if present."""
    example_dir = Path(__file__).parent
    env_file = example_dir / ".env"

    if env_file.exists():
        env_path = env_file
    elif Path(".env").exists():
        env_path = Path(".env")
    else:
        print("No .env file found. Using environment variables.")
        print(f"  Hint: cp {example_dir}/.env.example {example_dir}/.env")
        return

    try:
        from dotenv import load_dotenv

        # override=True ensures .env values take precedence
        loaded = load_dotenv(env_path, override=True)
        if loaded:
            print(f"Loaded configuration from: {env_path}")
        else:
            print(f"Warning: Failed to load {env_path}")
    except ImportError:
        print(f"Warning: python-dotenv not installed. Cannot load {env_path}")
        print("  Install with: uv add python-dotenv")
        print("  Or: pip install python-dotenv")


# =============================================================================
# Configuration
# =============================================================================


def get_phoenix_dataset_name() -> str:
    """Get the Phoenix dataset name from environment or use default."""
    return os.environ.get("PHOENIX_DATASET_NAME", "rag-evaluation-dataset")


def get_trace_es_url() -> str | None:
    """Get the Elasticsearch URL for trace data."""
    return os.environ.get("TRACE_ES_URL")


# =============================================================================
# RAG Evaluation Helpers
# =============================================================================


def extract_retrieved_docs(output: dict[str, Any]) -> list[RetrievedDoc]:
    """Extract retrieved documents from task output.

    Customize this function based on your task output format.
    Expected output structure:
    {
        "retrieved_docs": [
            {"index": "my-index", "id": "doc-1", ...},
            {"index": "my-index", "id": "doc-2", ...},
        ],
        ...
    }
    """
    docs = output.get("retrieved_docs", [])
    return [
        RetrievedDoc(
            index=doc.get("index", ""),
            id=doc.get("id", doc.get("_id", "")),
        )
        for doc in docs
        if isinstance(doc, dict)
    ]


def extract_ground_truth(expected: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract ground truth relevance from expected output.

    Supports multiple formats:
    1. Phoenix format: {"groundTruth": {"index": {"doc-id": score}}}
    2. Standard format: {"ground_truth": {"index": {"doc-id": score}}}
    3. Simple format: {"relevant_docs": ["doc-id"], "index": "my-index"}
    """
    # Format 1: Phoenix format (camelCase)
    if "groundTruth" in expected:
        return expected["groundTruth"]

    # Format 2: Standard format (snake_case)
    if "ground_truth" in expected:
        return expected["ground_truth"]

    # Format 3: Simple list of relevant doc IDs
    if "relevant_docs" in expected:
        index = expected.get("index", "default")
        return {index: {doc_id: 1.0 for doc_id in expected["relevant_docs"]}}

    return {}


def create_rag_config(k: int = 5) -> RagEvaluatorConfig:
    """Create RAG evaluator configuration."""
    return RagEvaluatorConfig(
        k=k,
        extract_retrieved_docs=extract_retrieved_docs,
        extract_ground_truth=extract_ground_truth,
        relevance_threshold=0.5,  # Docs with score >= 0.5 are considered relevant
    )


# =============================================================================
# Task Definition
# =============================================================================


async def rag_task(
    example: Example,
    config: ElasticEvalsConfig,
    inference_client: KibanaInferenceClient,
) -> dict[str, Any]:
    """Execute the RAG task by calling the Kibana Agent Builder API.

    This calls the actual agent to get real responses for evaluation.

    Phoenix dataset format expected:
    - input.question: The query to send to the agent
    - output.expected: The expected answer (for comparison)
    - output.groundTruth: Ground truth docs {index: {doc_id: relevance}}
    """
    import httpx

    # Extract the question from Phoenix input
    question = example.input.get("question") or example.input.get("query") or ""

    if not question:
        raise ValueError("No question found in example input")

    # Call the Kibana Agent Builder API
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{config.kibana_url}/api/agent_builder/converse",
            json={
                "connector_id": config.connector_id,
                "input": question,
            },
            headers={
                "kbn-xsrf": "true",
                "Authorization": f"Basic {config.kibana_auth}",
                "elastic-api-version": "2023-10-31",
            },
        )
        response.raise_for_status()
        data = response.json()

    # Extract response from agent
    response_payload = data.get("response", {})
    message = response_payload.get("message", "")
    steps = data.get("steps", [])
    trace_id = data.get("trace_id") or data.get("traceId")
    conversation_id = data.get("conversation_id")

    # Extract retrieved documents from agent response
    retrieved_docs = []
    for step in steps:
        if step.get("type") != "tool_call":
            continue

        for result in step.get("results") or []:
            if not isinstance(result, dict) or result.get("type") == "error":
                continue

            result_data = result.get("data", {})
            if not isinstance(result_data, dict):
                continue

            # Agent Builder returns documents with 'reference' field
            reference = result_data.get("reference")
            if isinstance(reference, dict) and reference.get("id"):
                retrieved_docs.append(
                    {
                        "index": reference.get("index", ""),
                        "id": reference.get("id", ""),
                        "content": result_data.get("content", {}),
                    }
                )
                continue

            # Fallback: documents/hits list format
            docs = result_data.get("documents") or result_data.get("hits") or []
            for doc in docs:
                if isinstance(doc, dict):
                    retrieved_docs.append(
                        {
                            "index": doc.get("_index", doc.get("index", "")),
                            "id": doc.get("_id", doc.get("id", "")),
                            "content": doc.get("_source", doc.get("content", {})),
                        }
                    )

    return {
        "answer": message,
        "messages": [{"message": message}] if message else [],
        "steps": steps,
        "traceId": trace_id,
        "conversation_id": conversation_id,
        "retrieved_docs": retrieved_docs,
    }


# =============================================================================
# Evaluator Setup
# =============================================================================


def create_evaluators(
    config: ElasticEvalsConfig,
    inference_client: KibanaInferenceClient,
    trace_es_client: AsyncElasticsearch | None,
    log: logging.Logger,
) -> list[Evaluator]:
    """Create all evaluators for the evaluation run."""
    evaluators: list[Evaluator] = []

    # 1. RAG Metrics Evaluators (Code-based)
    rag_config = create_rag_config(k=5)
    evaluators.extend(create_rag_evaluators(rag_config))
    print("  - RAG evaluators: Precision@K, Recall@K, F1@K")

    # 2. Groundedness Evaluator (LLM-based)
    # Note: The groundedness analysis evaluator is typically run in the task
    # and the quantitative evaluator extracts scores from the analysis
    evaluators.append(create_quantitative_groundedness_evaluator())
    print("  - Groundedness evaluator")

    # 3. Trace-based Evaluators (require Elasticsearch with trace data)
    if trace_es_client:
        evaluators.append(
            create_latency_evaluator(trace_es_client=trace_es_client, log=log)
        )
        evaluators.append(
            create_input_tokens_evaluator(trace_es_client=trace_es_client, log=log)
        )
        evaluators.append(
            create_output_tokens_evaluator(trace_es_client=trace_es_client, log=log)
        )
        evaluators.append(
            create_tool_calls_evaluator(trace_es_client=trace_es_client, log=log)
        )
        print("  - Trace-based evaluators: Latency, Input/Output Tokens, Tool Calls")
    else:
        print("  - Trace-based evaluators: SKIPPED (no TRACE_ES_URL configured)")

    return evaluators


# =============================================================================
# Reporting Helpers
# =============================================================================


def model_info_from_config(config: ElasticEvalsConfig) -> ModelInfo:
    """Extract model info from configuration."""
    model = config.model or {}
    return ModelInfo(
        id=model.get("id"),
        family=model.get("family", "unknown"),
        provider=model.get("provider", "unknown"),
    )


# =============================================================================
# Main Execution
# =============================================================================


async def main() -> None:
    """Main entry point for the Phoenix evaluation example."""
    _load_dotenv()

    print("=" * 60)
    print("Phoenix Dataset Evaluation Example")
    print("=" * 60)
    print()

    # 1. Load configuration
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)
    log = config.logger

    # 2. Load dataset from Phoenix
    phoenix_dataset_name = get_phoenix_dataset_name()
    print(f"Loading dataset from Phoenix: {phoenix_dataset_name}")

    phoenix_config = PhoenixDatasetConfig.from_env()
    print(f"  Phoenix URL: {phoenix_config.base_url}")

    try:
        dataset = load_dataset_from_phoenix(
            dataset_name=phoenix_dataset_name,
            config=phoenix_config,
        )
        print(f"  Loaded {len(dataset.examples)} examples")
    except Exception as e:
        print(f"  ERROR: Failed to load dataset from Phoenix: {e}")
        print()
        print("Make sure Phoenix is running and the dataset exists.")
        print("You can create a dataset in Phoenix or use a local dataset instead.")
        return

    print()

    # 3. Initialize clients
    client = ElasticEvalsClient(config)
    inference_client = client.get_inference_client()

    # Initialize trace ES client if configured
    trace_es_url = get_trace_es_url()
    trace_es_client = AsyncElasticsearch(trace_es_url) if trace_es_url else None

    # 4. Create evaluators
    print("Creating evaluators:")
    evaluators = create_evaluators(config, inference_client, trace_es_client, log)
    print()

    # 5. Create groundedness analysis evaluator for use in task
    groundedness_analysis = create_groundedness_analysis_evaluator(
        inference_client=inference_client, log=log
    )

    # 6. Define task wrapper that includes groundedness analysis
    async def task_with_groundedness(example: Example) -> dict[str, Any]:
        """Task wrapper that runs groundedness analysis."""
        response = await rag_task(example, config, inference_client)

        # Run groundedness analysis and attach to response
        params = EvaluatorParams(
            input=example.input,
            output=response,
            expected=example.output,
            metadata=example.metadata,
        )

        try:
            groundedness_result = await groundedness_analysis.evaluate(params)
            response["groundednessAnalysis"] = groundedness_result.metadata
        except Exception as e:
            log.warning(f"Groundedness analysis failed: {e}")
            response["groundednessAnalysis"] = None

        return response

    # 7. Run experiment
    print("Running evaluation:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Dataset: {dataset.name}")
    print(f"  Examples: {len(dataset.examples)}")
    print(f"  Repetitions: {config.repetitions}")
    print(f"  Concurrency: {config.concurrency}")
    print()

    result = await client.run_experiment(
        dataset=dataset,
        task=task_with_groundedness,
        evaluators=evaluators,
    )

    print(f"Completed experiment: {result.id}")
    print(f"  Total evaluation runs: {len(result.evaluation_runs)}")
    print()

    # 8. Export to Elasticsearch
    print(f"Exporting scores to Elasticsearch: {config.evaluations_es_url}")
    es = AsyncElasticsearch(config.evaluations_es_url)
    repository = EvaluationScoreRepository(es, log)

    task_model = model_info_from_config(config)
    evaluator_model = model_info_from_config(config)

    documents = build_flattened_score_documents(
        experiments=[result],
        task_model=task_model,
        evaluator_model=evaluator_model,
        run_id=config.run_id,
        total_repetitions=config.repetitions,
    )

    await repository.export_scores(documents)
    print(f"  Exported {len(documents)} score documents")
    print()

    # 9. Export to Phoenix Experiments (if enabled)
    if config.phoenix_experiment_export:
        print("Exporting experiment to Phoenix...")
        try:
            from elastic_evals.export.phoenix_experiments import (
                export_experiment_to_phoenix,
            )

            phoenix_result = await export_experiment_to_phoenix(
                experiment=result,
                dataset=dataset,
                experiment_name=f"elastic-evals: {config.run_id}",
                experiment_metadata={
                    "source": "phoenix_eval_example",
                },
                config=phoenix_config,
            )
            print(f"  Phoenix Experiment ID: {phoenix_result.experiment_id}")
            if phoenix_result.experiment_url:
                print(f"  View in Phoenix: {phoenix_result.experiment_url}")
            print(f"  Task runs exported: {phoenix_result.task_runs_exported}")
            print(
                f"  Evaluation runs exported: {phoenix_result.evaluation_runs_exported}"
            )
        except ImportError as e:
            print(f"  WARNING: Phoenix export not available: {e}")
            print("  Install with: pip install elastic-evals[phoenix]")
        except Exception as e:
            print(f"  ERROR: Failed to export to Phoenix: {e}")
        print()

    # 10. Generate report
    run_stats = await repository.get_stats_by_run_id(config.run_id)
    await es.close()

    if run_stats:
        print("Evaluation Results:")
        print("-" * 60)

        # Configure display options for non-percentage metrics
        from elastic_evals.reporting.types import (
            ReportDisplayOptions,
            EvaluatorDisplayOptions,
        )

        # Trace-based evaluators return absolute values, not percentages
        display_options = ReportDisplayOptions(
            evaluator_display_options={
                "Latency": EvaluatorDisplayOptions(
                    stats_to_include=["mean", "median", "std_dev", "min", "max"],
                    unit_suffix="s",
                    decimal_places=2,
                ),
                "Input Tokens": EvaluatorDisplayOptions(
                    stats_to_include=["mean", "median", "std_dev", "min", "max"],
                    decimal_places=0,
                ),
                "Output Tokens": EvaluatorDisplayOptions(
                    stats_to_include=["mean", "median", "std_dev", "min", "max"],
                    decimal_places=0,
                ),
                "Tool Calls": EvaluatorDisplayOptions(
                    stats_to_include=["mean", "median", "std_dev", "min", "max"],
                    decimal_places=0,
                ),
            }
        )

        reporter = DefaultReporter()
        reporter.report(
            stats=run_stats.stats,
            repetitions=run_stats.total_repetitions,
            task_model=run_stats.task_model,
            evaluator_model=run_stats.evaluator_model,
            display_options=display_options,
        )

    # 11. Cleanup
    if trace_es_client:
        await trace_es_client.close()

    print()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
