# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Granular Wix evaluation workflow using the public evals API clients."""

from __future__ import annotations

import asyncio
import socket
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, cast

import httpx
import pandas as pd
from dotenv import load_dotenv
from orca.evaluation.evaluators.retrieval import F1AtK, PrecisionAtK, RecallAtK  # type: ignore[import-untyped]
from orca.evaluation.evidence.types import EvaluationEvidence, EvidenceArtifact  # type: ignore[import-untyped]

from elastic_evals.agent_builder import (
    AgentBuilderClient,
    AgentConfiguration,
    CreateAgentRequest,
    CreateToolRequest,
    IndexSearchToolConfig,
    ToolSelection,
    build_agent_builder_headers,
)
from elastic_evals.api import (
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
from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.export import build_ingest_score_item, get_git_metadata
from elastic_evals.indexing import get_elasticsearch_client
from elastic_evals.tracing import init_tracing, with_evaluator_span, with_task_span
from elastic_evals.types import EvaluationResult, EvaluationRun, Evaluator, EvaluatorParams, Example, RunData
from examples.opik_vs_elastic.helpers.helpers import (
    AGENT_ID,
    AGENT_INSTRUCTIONS,
    AGENT_NAME,
    ENV_PATH,
    GROUND_TRUTH_COLUMN,
    INDEX_NAME,
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_ID,
    WIX_KNOWLEDGE_BASE_PATH,
    WIX_QA_DATASET_PATH,
    _extract_retrieved_doc_ids,  # noqa: PLC2701
    _parse_relevant_doc_ids,  # noqa: PLC2701
    _to_string_list,  # noqa: PLC2701
)

EXPERIMENT_NAME = "Wix QA - granular Evaluators and Scores API workflow"
DATASET_NAME = "wix_qa_granular_smoke"
DATASET_DESCRIPTION = "Small Wix QA sample for the granular API workflow."
EXAMPLE_LIMIT = 3
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

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.kibana_url}/api/agent_builder/converse",
            json={
                "connector_id": config.connector_id,
                "agent_id": config.agent_id if agent_id is None else agent_id,
                "input": example.input.get("question"),
            },
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


def _task_model(config: ElasticEvalsConfig) -> Model:
    configured = config.model or {}
    model_id = configured.get("id")
    return Model(
        id=str(model_id) if model_id is not None else config.connector_id,
        family=str(configured["family"]) if configured.get("family") is not None else None,
        provider=str(configured["provider"]) if configured.get("provider") is not None else None,
    )


def _reference_data(expected: Any) -> dict[str, Any] | None:
    value = expected.get("expected") if isinstance(expected, dict) else None
    return {"expected": value} if isinstance(value, str) and value else None


def _evaluator_configs(
    definitions: list[EvaluatorDefinition],
    connector_id: str,
) -> tuple[list[ValidateEvaluatorConfig], list[EvaluateEvaluatorConfig]]:
    return (
        [ValidateEvaluatorConfig(name=definition.name, version=definition.version) for definition in definitions],
        [
            EvaluateEvaluatorConfig(
                name=definition.name,
                version=definition.version,
                connector_id=connector_id if definition.kind == "llm" else None,
            )
            for definition in definitions
        ],
    )


def _scored_run(
    executed: dict[str, Any],
    *,
    name: str,
    result: EvaluationResult,
    trace_id: str | None = None,
) -> tuple[dict[str, Any], EvaluationRun]:
    run_data = executed["data"]
    return (
        executed,
        EvaluationRun(
            name=name,
            result=result,
            example_index=run_data.example_index,
            repetition_index=run_data.repetition,
            experiment_run_id=executed["key"],
            trace_id=trace_id,
            example_id=executed["example_id"],
        ),
    )


def _score_request(
    context: dict[str, Any],
    *,
    evaluator_model: Model,
    scored_runs: list[tuple[dict[str, Any], EvaluationRun]],
) -> IngestScoresRequest | None:
    payloads = [
        build_ingest_score_item(
            run_id=context["config"].run_id,
            experiment_id=context["experiment_id"],
            experiment_name=context["experiment_name"],
            suite_id=context["config"].suite_id,
            task_model=context["task_model"],
            evaluator_model=evaluator_model,
            run_metadata=context["run_metadata"],
            environment=context["environment"],
            ci=None,
            dataset_id=context["dataset_id"],
            dataset_name=context["dataset_name"],
            example_id=executed["example_id"],
            example_index=executed["data"].example_index,
            example_input=executed["data"].input,
            task_run=executed["data"],
            evaluation_run=evaluation,
        )
        for executed, evaluation in scored_runs
    ]
    if not payloads:
        return None
    return payloads[0].model_copy(update={"scores": [payload.scores[0] for payload in payloads]})


async def _ingest_scores(
    client: KibanaScoresClient,
    *,
    label: str,
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


async def _export_scores(
    client: KibanaScoresClient,
    context: dict[str, Any],
    *,
    label: str,
    evaluator_model_id: str,
    scored_runs: list[tuple[dict[str, Any], EvaluationRun]],
) -> None:
    await _ingest_scores(
        client,
        label=label,
        payload=_score_request(
            context,
            evaluator_model=Model(id=evaluator_model_id),
            scored_runs=scored_runs,
        ),
    )


async def _resolve_subject(
    client: KibanaEvaluatorsClient,
    executed: dict[str, Any],
) -> EvaluationSubject | None:
    run_data = executed["data"]
    trace_id = run_data.trace_id
    if not trace_id:
        print(f"Skipping Kibana evaluators for example={run_data.example_index}: no trace ID")
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
    print(f"Trace ready: example={run_data.example_index}, profile={instrumentation.profile}")
    return EvaluationSubject(
        traces=[
            EvaluationTrace(
                trace_id=trace_id,
                reference_data=_reference_data(run_data.expected),
            )
        ],
        instrumentation=instrumentation,
    )


def _orca_evidence(document_ids: list[str]) -> EvaluationEvidence:
    return EvaluationEvidence(
        artifacts=[
            EvidenceArtifact(
                type="resource_list",
                data={"resources": [{"reference": {"id": document_id}} for document_id in document_ids]},
            )
        ]
    )


def _evaluate_with_orca(
    executed_runs: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], EvaluationRun]]:
    scored: list[tuple[dict[str, Any], EvaluationRun]] = []
    for executed in executed_runs:
        run_data = executed["data"]
        retrieved = _extract_retrieved_doc_ids(run_data.output, tool_id=SEARCH_TOOL_ID)
        relevant = _to_string_list((run_data.metadata or {}).get("relevant_doc_ids"))
        evidence = _orca_evidence(retrieved)
        harness = {
            "name": "agent_builder_converse",
            "adapter": "run2_task_output",
            "locator": None,
            "metadata": {"trace_id": run_data.trace_id},
        }
        for metric in (PrecisionAtK(k=3), RecallAtK(k=3), F1AtK(k=3)):
            score = metric.score(
                evidence=evidence.model_dump(mode="json"),
                harness=harness,
                relevant_doc_ids=relevant,
            )
            if isinstance(score, list):
                raise TypeError(f"Orca evaluator {metric.name} returned multiple scores")
            scored.append(
                _scored_run(
                    executed,
                    name=f"orca.{score.name}",
                    result=EvaluationResult(
                        score=None if score.scoring_failed else float(score.value),
                        label="unavailable" if score.scoring_failed else None,
                        explanation=score.reason,
                        metadata={
                            "source": "orca",
                            "scoring_failed": score.scoring_failed,
                            "k": 3,
                        },
                    ),
                )
            )
    print(f"Orca evaluation complete: scores={len(scored)}")
    return scored


async def main() -> None:
    print("[1] Load configuration and prepare the knowledge base")
    load_dotenv(ENV_PATH)
    config = ElasticEvalsConfig.from_env()
    init_tracing(config.tracing)

    qa_wix = pd.read_csv(WIX_QA_DATASET_PATH)
    qa_wix["relevant_doc_ids"] = qa_wix[GROUND_TRUTH_COLUMN].apply(_parse_relevant_doc_ids)
    kb_wix = pd.read_csv(WIX_KNOWLEDGE_BASE_PATH)
    es_client = get_elasticsearch_client()
    es_client.create_index(INDEX_NAME, mappings=None, recreate=True)
    records = cast(Iterable[Mapping[str, Any]], kb_wix.to_dict(orient="records"))
    indexed, errors = es_client.index_records(INDEX_NAME, records, id_field="id")
    print(f"Indexed {indexed} knowledge-base documents ({len(errors)} errors)")

    # Each API client is constructed once. ElasticEvalsClient is not used.
    datasets_client = KibanaDatasetsClient(config.kibana_url, api_key=config.kibana_api_key)
    evaluators_client = KibanaEvaluatorsClient(
        config.kibana_url,
        api_key=config.kibana_api_key,
        timeout=180.0,
    )
    scores_client = KibanaScoresClient(config.kibana_url, api_key=config.kibana_api_key)
    agent_builder_client = AgentBuilderClient(config.kibana_url, api_key=config.kibana_api_key)

    experiment_id = str(uuid.uuid4())
    print(f"Experiment: {EXPERIMENT_NAME}\nExperiment ID: {experiment_id}")

    print("[2] Upsert the dataset and set up Agent Builder")
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
        for _, row in qa_wix.head(EXAMPLE_LIMIT).iterrows()
    ]
    upserted = await datasets_client.upsert(
        DATASET_NAME,
        DATASET_DESCRIPTION,
        dataset_examples,
    )
    dataset = await datasets_client.get(upserted.dataset_id)
    print(f"Dataset ready: {dataset.name} ({len(dataset.examples)} examples)")

    tool = await agent_builder_client.create_tool(
        CreateToolRequest(
            id=SEARCH_TOOL_ID,
            type="index_search",
            description=SEARCH_TOOL_DESCRIPTION,
            configuration=IndexSearchToolConfig(pattern=INDEX_NAME),
        ),
        update_if_exists=True,
    )
    agent = await agent_builder_client.create_agent(
        CreateAgentRequest(
            id=AGENT_ID,
            name=AGENT_NAME,
            description="Agent for Wix QA retrieval.",
            configuration=AgentConfiguration(
                instructions=AGENT_INSTRUCTIONS,
                tools=[ToolSelection(tool_ids=[tool.id])],
            ),
        ),
        update_if_exists=True,
    )
    print(f"Agent ready: {agent.id} (tool: {tool.id})")

    print("[3] Discover Kibana evaluators")
    catalog = await evaluators_client.list_evaluators()
    by_name = {definition.name: definition for definition in catalog.evaluators}
    missing = [name for name in KIBANA_EVALUATOR_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(f"Kibana evaluator catalog is missing: {', '.join(missing)}")
    definitions = [by_name[name] for name in KIBANA_EVALUATOR_NAMES]
    validate_configs, evaluate_configs = _evaluator_configs(
        definitions,
        config.evaluator_connector_id or config.connector_id,
    )
    for definition in definitions:
        print(f"  {definition.name} v{definition.version} ({definition.kind})")

    print("[4] Execute Agent Builder tasks")
    # Local PoC bookkeeping; these dictionaries are not SDK models.
    executed_runs: list[dict[str, Any]] = []
    for repetition in range(config.repetitions):
        for example_index, example in enumerate(dataset.examples):
            sdk_example = Example(
                input=example.input or {},
                output=example.output,
                metadata=example.metadata,
            )

            async def task_runner() -> dict[str, Any]:
                return await agent_builder_task(
                    sdk_example,
                    config,
                    agent_id=agent.id,
                )

            output, sdk_trace_id = await with_task_span(
                "wix_agent_builder_task",
                {"example.index": example_index, "task.repetition": repetition},
                task_runner,
            )
            interaction_trace_id = output.pop("_interaction_trace_id", sdk_trace_id)
            executed_runs.append(
                {
                    "key": f"{example_index}-{repetition}-{uuid.uuid4()}",
                    "example_id": example.id,
                    "data": RunData(
                        example_index=example_index,
                        repetition=repetition,
                        input=example.input or {},
                        expected=example.output,
                        metadata=example.metadata or {},
                        output=output,
                        trace_id=interaction_trace_id,
                    ),
                }
            )
            print(
                f"Task complete: example={example_index}, repetition={repetition}, "
                f"trace={interaction_trace_id or 'unavailable'}"
            )

    print("[5] Evaluate stored runs with Kibana and Document Recall")
    document_recall = create_document_recall_evaluator(tool_id=SEARCH_TOOL_ID)
    kibana_scores: list[tuple[dict[str, Any], EvaluationRun]] = []
    document_recall_scores: list[tuple[dict[str, Any], EvaluationRun]] = []

    for executed in executed_runs:
        run_data = executed["data"]
        params = EvaluatorParams(
            input=run_data.input,
            output=run_data.output,
            expected=run_data.expected,
            metadata=run_data.metadata,
            trace_id=run_data.trace_id,
        )

        async def document_recall_runner() -> EvaluationResult:
            return await document_recall.evaluate(params)

        recall_result, recall_trace_id = await with_evaluator_span(
            document_recall.name,
            {"example.index": run_data.example_index},
            document_recall_runner,
        )
        document_recall_scores.append(
            _scored_run(
                executed,
                name=document_recall.name,
                result=recall_result,
                trace_id=recall_trace_id,
            )
        )

        subject = await _resolve_subject(evaluators_client, executed)
        if subject is None:
            continue
        validation = await evaluators_client.validate(
            ValidateEvaluatorsRequest(subject=subject, evaluators=validate_configs)
        )
        ready = sum(result.ready for result in validation.evaluators)
        print(f"Validation: example={run_data.example_index}, ready={ready}/{len(validation.evaluators)}")
        for result in validation.evaluators:
            if not result.ready:
                detail = result.remediation or ", ".join(result.unmet)
                print(f"  {result.name}: not ready ({detail})")

        response = await evaluators_client.evaluate(EvaluateRequest(subject=subject, evaluators=evaluate_configs))
        before = len(kibana_scores)
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
                    }
                )
                kibana_scores.append(
                    _scored_run(
                        executed,
                        name=score.name,
                        result=EvaluationResult(
                            score=score.score,
                            label=score.label,
                            explanation=score.explanation,
                            metadata=metadata,
                        ),
                    )
                )
        print(
            f"Evaluation complete: example={run_data.example_index}, "
            f"Kibana scores={len(kibana_scores) - before}, Document Recall=1"
        )

    git = get_git_metadata()
    export_context = {
        "config": config,
        "experiment_id": experiment_id,
        "experiment_name": EXPERIMENT_NAME,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "task_model": _task_model(config),
        "run_metadata": RunMetadata(
            total_repetitions=config.repetitions,
            git_branch=git.branch,
            git_commit_sha=git.commit_sha,
        ),
        "environment": Environment(hostname=socket.gethostname()),
    }
    connector_id = config.evaluator_connector_id or config.connector_id

    print("[6] Create the experiment with the initial scores")
    await _export_scores(
        scores_client,
        export_context,
        label="Kibana evaluator scores",
        evaluator_model_id=connector_id,
        scored_runs=kibana_scores,
    )
    await _export_scores(
        scores_client,
        export_context,
        label="Document Recall scores",
        evaluator_model_id="elastic-evals-sdk-python",
        scored_runs=document_recall_scores,
    )

    print("[7] Run external Orca evaluators and attach their scores")
    await _export_scores(
        scores_client,
        export_context,
        label="External Orca scores",
        evaluator_model_id="orca",
        scored_runs=_evaluate_with_orca(executed_runs),
    )

    print(f"Finished.\nExperiment '{EXPERIMENT_NAME}' is ready in Kibana.\nExperiment ID: {experiment_id}")


if __name__ == "__main__":
    asyncio.run(main())
