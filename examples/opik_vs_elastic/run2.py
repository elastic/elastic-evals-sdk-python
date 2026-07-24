# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Granular Wix evaluation workflow using the public evals API clients."""

from __future__ import annotations

import asyncio
import socket
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from orca.evaluation.evaluators.retrieval import (  # type: ignore[import-untyped]
    F1AtK,
    PrecisionAtK,
    RecallAtK,
)
from orca.evaluation.evidence.types import (  # type: ignore[import-untyped]
    EvaluationEvidence,
    EvidenceArtifact,
)

from elastic_evals.agent_builder import (
    AgentBuilderClient,
    AgentConfiguration,
    CreateAgentRequest,
    CreateToolRequest,
    IndexSearchToolConfig,
    ToolSelection,
)
from elastic_evals.api import (
    DatasetExample,
    Environment,
    EvaluateEvaluatorConfig,
    EvaluateRequest,
    EvaluationInstrumentation,
    EvaluationSubject,
    EvaluationTrace,
    EvaluatorDefinition,
    IngestScoresRequest,
    KibanaDatasetsClient,
    KibanaEvaluatorsClient,
    KibanaEvaluatorsError,
    Model,
    RunMetadata,
    UpsertDatasetExamplePayload,
    ValidateEvaluatorConfig,
    ValidateEvaluatorsRequest,
)
from elastic_evals.api.scores_client import KibanaScoresClient
from elastic_evals.config import ElasticEvalsConfig
from elastic_evals.export import build_ingest_score_item, get_git_metadata
from elastic_evals.indexing import get_elasticsearch_client
from elastic_evals.tracing import init_tracing, with_evaluator_span, with_task_span
from elastic_evals.types import EvaluationResult, EvaluationRun, EvaluatorParams, Example, RunData
from examples.opik_vs_elastic import run as managed_run

ENV_PATH = Path(__file__).parent / ".env"
GROUND_TRUTH_COLUMN = managed_run.GROUND_TRUTH_COLUMN
INDEX_NAME = managed_run.INDEX_NAME
SEARCH_TOOL_ID = managed_run.SEARCH_TOOL_ID
EXPERIMENT_NAME = "Wix QA - granular Evaluators and Scores API workflow"
DATASET_NAME = "wix_qa_granular_smoke"
DATASET_DESCRIPTION = "Small Wix QA sample for the granular API workflow."
EXAMPLE_LIMIT = 3
EVALUATOR_TIMEOUT_SECONDS = 180.0
TRACE_READY_ATTEMPTS = 10
TRACE_READY_WAIT_SECONDS = 2.0
KIBANA_EVALUATOR_NAMES = (
    "correctness",
    "groundedness",
    "latency",
    "input_tokens",
    "output_tokens",
    "tool_calls",
)


@dataclass(frozen=True)
class ExecutedRun:
    key: str
    example_id: str
    data: RunData


def _phase(number: int, title: str) -> None:
    print(f"\n[{number}/8] {title}")


def _task_model(config: ElasticEvalsConfig) -> Model:
    configured = config.model or {}
    model_id = configured.get("id")
    return Model(
        id=str(model_id) if model_id is not None else config.connector_id,
        family=str(configured["family"]) if configured.get("family") is not None else None,
        provider=str(configured["provider"]) if configured.get("provider") is not None else None,
    )


def _reference_data(expected: Any) -> dict[str, Any] | None:
    if not isinstance(expected, dict):
        return None
    value = expected.get("expected")
    return {"expected": value} if isinstance(value, str) and value else None


def _orca_evidence(document_ids: list[str]) -> EvaluationEvidence:
    return EvaluationEvidence(
        artifacts=[
            EvidenceArtifact(
                type="resource_list",
                data={"resources": [{"reference": {"id": document_id}} for document_id in document_ids]},
            )
        ]
    )


def _build_score_request(
    *,
    experiment_id: str,
    experiment_name: str,
    config: ElasticEvalsConfig,
    dataset_id: str,
    dataset_name: str,
    task_model: Model,
    evaluator_model: Model,
    run_metadata: RunMetadata,
    environment: Environment,
    scored_runs: list[tuple[ExecutedRun, EvaluationRun]],
) -> IngestScoresRequest | None:
    payloads = [
        build_ingest_score_item(
            run_id=config.run_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            suite_id=config.suite_id,
            task_model=task_model,
            evaluator_model=evaluator_model,
            run_metadata=run_metadata,
            environment=environment,
            ci=None,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            example_id=executed.example_id,
            example_index=executed.data.example_index,
            example_input=executed.data.input,
            task_run=executed.data,
            evaluation_run=evaluation,
        )
        for executed, evaluation in scored_runs
    ]
    if not payloads:
        return None

    return payloads[0].model_copy(update={"scores": [payload.scores[0] for payload in payloads]})


async def _ingest_scores(
    *,
    label: str,
    client: KibanaScoresClient,
    payload: IngestScoresRequest | None,
) -> None:
    if payload is None:
        print(f"{label}: no scores to ingest")
        return

    result = await client.ingest_scores(payload)
    print(f"{label}: ingested={result.ingested}, conflicted={result.conflicted}, failed={len(result.failed)}")
    if result.failed:
        reasons = "; ".join(failure.reason for failure in result.failed)
        raise RuntimeError(f"{label} had failed score documents: {reasons}")


async def _setup_agent(config: ElasticEvalsConfig) -> str:
    client = AgentBuilderClient(
        kibana_url=config.kibana_url,
        api_key=config.kibana_api_key,
    )
    tool = await client.get_or_create_tool(
        CreateToolRequest(
            id=SEARCH_TOOL_ID,
            type="index_search",
            description="Search the Wix knowledge base articles.",
            configuration=IndexSearchToolConfig(pattern=INDEX_NAME),
        )
    )
    agent = await client.get_or_create_agent(
        CreateAgentRequest(
            id="wix-eval-agent",
            name="Wix Agent",
            description="Agent for Wix QA retrieval.",
            configuration=AgentConfiguration(
                instructions="Answer questions using the Wix knowledge base.",
                tools=[ToolSelection(tool_ids=[tool.id])],
            ),
        )
    )
    print(f"Agent ready: {agent.id} (tool: {tool.id})")
    return agent.id


async def _execute_tasks(
    *,
    examples: list[DatasetExample],
    config: ElasticEvalsConfig,
    agent_id: str,
) -> list[ExecutedRun]:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def execute(example: DatasetExample, example_index: int, repetition: int) -> ExecutedRun:
        async with semaphore:
            sdk_example = Example(
                input=example.input or {},
                output=example.output,
                metadata=example.metadata,
            )

            async def task_runner() -> dict[str, Any]:
                return await managed_run.agent_builder_task(sdk_example, config, agent_id=agent_id)

            output, sdk_trace_id = await with_task_span(
                "wix_agent_builder_task",
                {
                    "example.index": example_index,
                    "task.repetition": repetition,
                },
                task_runner,
            )
            interaction_trace_id = output.pop("_interaction_trace_id", sdk_trace_id)
            run = ExecutedRun(
                key=f"{example_index}-{repetition}-{uuid.uuid4()}",
                example_id=example.id,
                data=RunData(
                    example_index=example_index,
                    repetition=repetition,
                    input=example.input or {},
                    expected=example.output,
                    metadata=example.metadata or {},
                    output=output,
                    trace_id=interaction_trace_id,
                ),
            )
            print(
                f"Task complete: example={example_index}, repetition={repetition}, "
                f"trace={interaction_trace_id or 'unavailable'}"
            )
            return run

    jobs = [
        execute(example, example_index, repetition)
        for repetition in range(config.repetitions)
        for example_index, example in enumerate(examples)
    ]
    return list(await asyncio.gather(*jobs))


async def _resolve_subject(
    client: KibanaEvaluatorsClient,
    executed: ExecutedRun,
) -> EvaluationSubject | None:
    trace_id = executed.data.trace_id
    if not trace_id:
        print(f"Skipping Kibana evaluators for example={executed.data.example_index}: no trace ID")
        return None

    for attempt in range(1, TRACE_READY_ATTEMPTS + 1):
        try:
            resolved = await client.resolve_instrumentation(trace_id)
            break
        except KibanaEvaluatorsError as error:
            if error.status_code != 404 or attempt == TRACE_READY_ATTEMPTS:
                raise
            if attempt == 1:
                print(f"Trace {trace_id} is still indexing; waiting...")
            await asyncio.sleep(TRACE_READY_WAIT_SECONDS)

    recommendation = resolved.recommended_instrumentation
    instrumentation = (
        EvaluationInstrumentation.model_validate({"profile": recommendation.profile})
        if recommendation
        else EvaluationInstrumentation()
    )
    print(f"Trace ready: example={executed.data.example_index}, profile={instrumentation.profile}")
    return EvaluationSubject(
        traces=[
            EvaluationTrace(
                trace_id=trace_id,
                reference_data=_reference_data(executed.data.expected),
            )
        ],
        instrumentation=instrumentation,
    )


async def _evaluate_with_kibana(
    *,
    client: KibanaEvaluatorsClient,
    definitions: list[EvaluatorDefinition],
    executed_runs: list[ExecutedRun],
    connector_id: str,
    concurrency: int,
) -> list[tuple[ExecutedRun, EvaluationRun]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_run(executed: ExecutedRun) -> list[tuple[ExecutedRun, EvaluationRun]]:
        async with semaphore:
            subject = await _resolve_subject(client, executed)
            if subject is None:
                return []

            validation = await client.validate(
                ValidateEvaluatorsRequest(
                    subject=subject,
                    evaluators=[
                        ValidateEvaluatorConfig(name=definition.name, version=definition.version)
                        for definition in definitions
                    ],
                )
            )
            ready = sum(result.ready for result in validation.evaluators)
            print(f"Validation: example={executed.data.example_index}, ready={ready}/{len(validation.evaluators)}")
            for validation_result in validation.evaluators:
                if not validation_result.ready:
                    detail = validation_result.remediation or ", ".join(validation_result.unmet)
                    print(f"  {validation_result.name}: not ready ({detail})")

            response = await client.evaluate(
                EvaluateRequest(
                    subject=subject,
                    evaluators=[
                        EvaluateEvaluatorConfig(
                            name=definition.name,
                            version=definition.version,
                            connector_id=connector_id if definition.kind == "llm" else None,
                        )
                        for definition in definitions
                    ],
                )
            )

            scored: list[tuple[ExecutedRun, EvaluationRun]] = []
            for api_result in response.results:
                if api_result.status == "error":
                    message = api_result.error.message if api_result.error else "unknown error"
                    print(f"  {api_result.evaluator.name}: error ({message})")
                    continue

                for score in api_result.scores or []:
                    metadata = dict(score.metadata or {})
                    metadata.update(
                        {
                            "source": "kibana_evaluators_api",
                            "evaluator_name": api_result.evaluator.name,
                            "evaluator_version": api_result.evaluator.version,
                            "evaluator_kind": api_result.evaluator.kind,
                        }
                    )
                    scored.append(
                        (
                            executed,
                            EvaluationRun(
                                name=score.name,
                                result=EvaluationResult(
                                    score=score.score,
                                    label=score.label,
                                    explanation=score.explanation,
                                    metadata=metadata,
                                ),
                                example_index=executed.data.example_index,
                                repetition_index=executed.data.repetition,
                                experiment_run_id=executed.key,
                                example_id=executed.example_id,
                            ),
                        )
                    )
            print(f"Kibana evaluation complete: example={executed.data.example_index}, scores={len(scored)}")
            return scored

    nested = await asyncio.gather(*(evaluate_run(executed) for executed in executed_runs))
    return [item for group in nested for item in group]


async def _evaluate_document_recall(
    executed_runs: list[ExecutedRun],
) -> list[tuple[ExecutedRun, EvaluationRun]]:
    evaluator = managed_run.create_document_recall_evaluator(tool_id=SEARCH_TOOL_ID)
    scored: list[tuple[ExecutedRun, EvaluationRun]] = []

    for executed in executed_runs:
        params = EvaluatorParams(
            input=executed.data.input,
            output=executed.data.output,
            expected=executed.data.expected,
            metadata=executed.data.metadata,
            trace_id=executed.data.trace_id,
        )

        async def evaluator_runner() -> EvaluationResult:
            return await evaluator.evaluate(params)

        result, evaluator_trace_id = await with_evaluator_span(
            evaluator.name,
            {"example.index": executed.data.example_index},
            evaluator_runner,
        )
        scored.append(
            (
                executed,
                EvaluationRun(
                    name=evaluator.name,
                    result=result,
                    example_index=executed.data.example_index,
                    repetition_index=executed.data.repetition,
                    experiment_run_id=executed.key,
                    trace_id=evaluator_trace_id,
                    example_id=executed.example_id,
                ),
            )
        )
    print(f"Document Recall complete: scores={len(scored)}")
    return scored


def _evaluate_with_orca(
    executed_runs: list[ExecutedRun],
) -> list[tuple[ExecutedRun, EvaluationRun]]:
    metrics = (PrecisionAtK(k=3), RecallAtK(k=3), F1AtK(k=3))
    scored: list[tuple[ExecutedRun, EvaluationRun]] = []

    for executed in executed_runs:
        retrieved = managed_run._extract_retrieved_doc_ids(executed.data.output, tool_id=SEARCH_TOOL_ID)
        relevant = managed_run._to_string_list((executed.data.metadata or {}).get("relevant_doc_ids"))
        evidence = _orca_evidence(retrieved)
        harness = {
            "name": "agent_builder_converse",
            "adapter": "run2_task_output",
            "locator": None,
            "metadata": {"trace_id": executed.data.trace_id},
        }

        for metric in metrics:
            result = metric.score(
                evidence=evidence.model_dump(mode="json"),
                harness=harness,
                relevant_doc_ids=relevant,
            )
            if isinstance(result, list):
                raise TypeError(f"Orca evaluator {metric.name} returned multiple scores")

            scored.append(
                (
                    executed,
                    EvaluationRun(
                        name=f"orca.{result.name}",
                        result=EvaluationResult(
                            score=None if result.scoring_failed else float(result.value),
                            label="unavailable" if result.scoring_failed else None,
                            explanation=result.reason,
                            metadata={
                                "source": "orca",
                                "original_name": result.name,
                                "scoring_failed": result.scoring_failed,
                                "k": 3,
                            },
                        ),
                        example_index=executed.data.example_index,
                        repetition_index=executed.data.repetition,
                        experiment_run_id=executed.key,
                        example_id=executed.example_id,
                    ),
                )
            )
    print(f"Orca evaluation complete: scores={len(scored)}")
    return scored


async def main() -> None:
    _phase(1, "Load configuration and prepare the knowledge base")
    load_dotenv(ENV_PATH)
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)

    qa_wix = pd.read_csv("gs://agent-builder-data-science-datasets/queries/wix_qa.csv")
    qa_wix["relevant_doc_ids"] = qa_wix[GROUND_TRUTH_COLUMN].apply(managed_run._parse_relevant_doc_ids)
    kb_wix = pd.read_csv(
        "gs://agent-builder-data-science-datasets/knowledge_bases/cleaned/"
        "customer_support/wix_knowledge_base/wix_knowledge_base.csv"
    )
    es_client = get_elasticsearch_client()
    es_client.create_index(INDEX_NAME, mappings=None, recreate=True)
    records = cast(
        Iterable[Mapping[str, Any]],
        kb_wix.to_dict(orient="records"),
    )
    indexed, errors = es_client.index_records(
        INDEX_NAME,
        records,
        id_field="id",
    )
    print(f"Indexed {indexed} knowledge-base documents ({len(errors)} errors)")

    datasets_client = KibanaDatasetsClient(
        config.kibana_url,
        api_key=config.kibana_api_key,
    )
    evaluators_client = KibanaEvaluatorsClient(
        config.kibana_url,
        api_key=config.kibana_api_key,
        timeout=EVALUATOR_TIMEOUT_SECONDS,
    )
    scores_client = KibanaScoresClient(
        config.kibana_url,
        api_key=config.kibana_api_key,
    )

    experiment_id = str(uuid.uuid4())
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Experiment ID: {experiment_id}")

    _phase(2, "Upsert the dataset and set up Agent Builder")
    sample = qa_wix.head(EXAMPLE_LIMIT)
    dataset_examples = [
        UpsertDatasetExamplePayload(
            input={"question": str(row["input_question"])},
            output={"expected": str(row["output_expected"])},
            metadata={
                GROUND_TRUTH_COLUMN: row[GROUND_TRUTH_COLUMN],
                "relevant_doc_ids": row["relevant_doc_ids"],
                "meta_query_id": str(row["meta_query_id"]),
            },
        )
        for _, row in sample.iterrows()
    ]
    upserted = await datasets_client.upsert(
        DATASET_NAME,
        DATASET_DESCRIPTION,
        dataset_examples,
    )
    dataset = await datasets_client.get(upserted.dataset_id)
    print(
        f"Dataset ready: {dataset.name} ({len(dataset.examples)} examples, "
        f"added={upserted.added}, unchanged={upserted.unchanged})"
    )
    agent_id = await _setup_agent(config)

    _phase(3, "Discover Kibana evaluators")
    catalog = await evaluators_client.list_evaluators()
    definitions_by_name = {definition.name: definition for definition in catalog.evaluators}
    missing = [name for name in KIBANA_EVALUATOR_NAMES if name not in definitions_by_name]
    if missing:
        raise RuntimeError(f"Kibana evaluator catalog is missing: {', '.join(missing)}")
    definitions = [definitions_by_name[name] for name in KIBANA_EVALUATOR_NAMES]
    for definition in definitions:
        print(f"  {definition.name} v{definition.version} ({definition.kind})")

    _phase(4, "Execute Agent Builder tasks")
    executed_runs = await _execute_tasks(
        examples=dataset.examples,
        config=config,
        agent_id=agent_id,
    )
    print(f"Task phase complete: {len(executed_runs)} runs recorded")

    _phase(5, "Evaluate stored runs with Kibana and Document Recall")
    connector_id = config.evaluator_connector_id or config.connector_id
    kibana_scores, document_recall_scores = await asyncio.gather(
        _evaluate_with_kibana(
            client=evaluators_client,
            definitions=definitions,
            executed_runs=executed_runs,
            connector_id=connector_id,
            concurrency=config.concurrency,
        ),
        _evaluate_document_recall(executed_runs),
    )

    git = get_git_metadata()
    run_metadata = RunMetadata(
        total_repetitions=config.repetitions,
        git_branch=git.branch,
        git_commit_sha=git.commit_sha,
    )
    environment = Environment(hostname=socket.gethostname())
    task_model = _task_model(config)

    _phase(6, "Create the experiment with the initial scores")
    await _ingest_scores(
        label="Kibana evaluator scores",
        client=scores_client,
        payload=_build_score_request(
            experiment_id=experiment_id,
            experiment_name=EXPERIMENT_NAME,
            config=config,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            task_model=task_model,
            evaluator_model=Model(id=connector_id),
            run_metadata=run_metadata,
            environment=environment,
            scored_runs=kibana_scores,
        ),
    )
    await _ingest_scores(
        label="Document Recall scores",
        client=scores_client,
        payload=_build_score_request(
            experiment_id=experiment_id,
            experiment_name=EXPERIMENT_NAME,
            config=config,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            task_model=task_model,
            evaluator_model=Model(id="elastic-evals-sdk-python"),
            run_metadata=run_metadata,
            environment=environment,
            scored_runs=document_recall_scores,
        ),
    )

    _phase(7, "Run external Orca evaluators and attach their scores")
    orca_scores = _evaluate_with_orca(executed_runs)
    await _ingest_scores(
        label="External Orca scores",
        client=scores_client,
        payload=_build_score_request(
            experiment_id=experiment_id,
            experiment_name=EXPERIMENT_NAME,
            config=config,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            task_model=task_model,
            evaluator_model=Model(id="orca"),
            run_metadata=run_metadata,
            environment=environment,
            scored_runs=orca_scores,
        ),
    )

    _phase(8, "Finished")
    print(f"Experiment '{EXPERIMENT_NAME}' is ready in Kibana.")
    print(f"Experiment ID: {experiment_id}")


if __name__ == "__main__":
    asyncio.run(main())
