"""Elasticsearch score export repository."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from socket import gethostname
from typing import Any, Iterable

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan

from elastic_evals.export.documents import (
    DatasetInfo,
    EnvironmentInfo,
    EvaluationScoreDocument,
    EvaluatorInfo,
    ExampleInfo,
    ModelInfo,
    RunMetadata,
    TaskInfo,
)
from elastic_evals.export.git_metadata import get_git_metadata
from elastic_evals.tracing import get_current_trace_id
from elastic_evals.utils.logging import (
    log_bulk_error,
    log_export_header,
    log_export_query_hint,
    log_export_success,
    log_index_template_created,
    log_no_scores_warning,
    log_scores_indexed,
)
from elastic_evals.types import EvaluationResult, RanExperiment

EVALUATIONS_DATA_STREAM_ALIAS = ".kibana-evaluations"
EVALUATIONS_DATA_STREAM_WILDCARD = ".kibana-evaluations*"
EVALUATIONS_DATA_STREAM_TEMPLATE = "kibana-evaluations-template"


def compute_input_hash(input_data: Any) -> str:
    try:
        json_data = json.dumps(input_data)
        return hashlib.sha256(json_data.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _build_index_template() -> dict[str, Any]:
    return {
        "index_patterns": [EVALUATIONS_DATA_STREAM_WILDCARD],
        "data_stream": {"hidden": True},
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "5s",
                "index.hidden": True,
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "run_id": {"type": "keyword"},
                    "experiment_id": {"type": "keyword"},
                    "example": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "keyword"},
                            "index": {"type": "integer"},
                            "input_hash": {"type": "keyword"},
                            "dataset": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "name": {"type": "keyword"},
                                },
                            },
                        },
                    },
                    "task": {
                        "type": "object",
                        "properties": {
                            "trace_id": {"type": "keyword"},
                            "repetition_index": {"type": "integer"},
                            "model": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "family": {"type": "keyword"},
                                    "provider": {"type": "keyword"},
                                },
                            },
                        },
                    },
                    "evaluator": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "keyword"},
                            "score": {"type": "float"},
                            "label": {"type": "keyword"},
                            "explanation": {"type": "text", "index": False},
                            "metadata": {"type": "flattened"},
                            "trace_id": {"type": "keyword"},
                            "model": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "family": {"type": "keyword"},
                                    "provider": {"type": "keyword"},
                                },
                            },
                        },
                    },
                    "run_metadata": {
                        "type": "object",
                        "properties": {
                            "git_branch": {"type": "keyword"},
                            "git_commit_sha": {"type": "keyword"},
                            "total_repetitions": {"type": "integer"},
                        },
                    },
                    "environment": {
                        "type": "object",
                        "properties": {
                            "hostname": {"type": "keyword"},
                        },
                    },
                },
            },
        },
    }


class EvaluationScoreRepository:
    def __init__(self, es_client: AsyncElasticsearch, log) -> None:
        self._es = es_client
        self._log = log

    async def ensure_index_template(self) -> None:
        template_body = _build_index_template()
        try:
            exists = await self._es.indices.exists_index_template(
                name=EVALUATIONS_DATA_STREAM_TEMPLATE
            )
        except Exception:
            exists = False

        if not exists:
            await self._es.indices.put_index_template(
                name=EVALUATIONS_DATA_STREAM_TEMPLATE,
                index_patterns=template_body["index_patterns"],
                data_stream=template_body["data_stream"],
                template=template_body["template"],
            )
            log_index_template_created()

    async def ensure_datastream(self) -> None:
        try:
            await self._es.indices.get_data_stream(name=EVALUATIONS_DATA_STREAM_ALIAS)
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "status", None)
            if status == 404:
                await self._es.indices.create_data_stream(name=EVALUATIONS_DATA_STREAM_ALIAS)
                self._log.debug(f"Created datastream: {EVALUATIONS_DATA_STREAM_ALIAS}")
            else:
                raise

    async def export_scores(self, documents: list[EvaluationScoreDocument]) -> None:
        await self.ensure_index_template()
        await self.ensure_datastream()

        if not documents:
            log_no_scores_warning()
            return

        log_export_header()

        operations: list[dict[str, Any]] = []
        for doc in documents:
            doc_id = "-".join(
                [
                    doc.run_id,
                    doc.example.dataset.id,
                    doc.example.id,
                    doc.evaluator.name,
                    str(doc.task.repetition_index),
                ]
            )
            operations.append({"create": {"_index": EVALUATIONS_DATA_STREAM_ALIAS, "_id": doc_id}})
            operations.append(doc.model_dump(by_alias=True, mode="json"))

        response = await self._es.bulk(operations=operations, refresh="wait_for")
        if response.get("errors"):
            failed = 0
            for item in response.get("items", []):
                create_result = item.get("create") if isinstance(item, dict) else None
                if create_result and create_result.get("error"):
                    failed += 1
            log_bulk_error(failed, len(documents))
            raise RuntimeError("Bulk indexing failed for evaluation scores")

        log_scores_indexed(len(documents))
        log_export_success()
        first_doc = documents[0]
        log_export_query_hint(
            first_doc.environment.hostname,
            first_doc.task.model.id,
            first_doc.run_id,
        )

    async def get_scores_by_run_id(self, run_id: str) -> list[EvaluationScoreDocument]:
        results: list[EvaluationScoreDocument] = []
        query = {"term": {"run_id": run_id}}

        async for hit in async_scan(
            self._es,
            index=EVALUATIONS_DATA_STREAM_WILDCARD,
            query={"query": query},
        ):
            source = hit.get("_source", {})
            try:
                results.append(EvaluationScoreDocument(**source))
            except Exception as exc:
                self._log.debug("Skipping invalid score document: %s", exc)

        return results


def build_flattened_score_documents(
    *,
    experiments: Iterable[RanExperiment],
    task_model: ModelInfo,
    evaluator_model: ModelInfo,
    run_id: str,
    total_repetitions: int,
) -> list[EvaluationScoreDocument]:
    documents: list[EvaluationScoreDocument] = []
    timestamp = datetime.utcnow()
    git_metadata = get_git_metadata()
    host_name = gethostname()

    for experiment in experiments:
        dataset_id = experiment.dataset_id
        dataset_name = experiment.dataset_name or dataset_id
        runs_by_id = experiment.runs or {}
        runs_list = list(runs_by_id.values())

        for eval_run in experiment.evaluation_runs or []:
            run_entry = None
            if eval_run.experiment_run_id and eval_run.experiment_run_id in runs_by_id:
                run_entry = runs_by_id[eval_run.experiment_run_id]
            elif eval_run.example_index is not None and eval_run.repetition_index is not None:
                run_entry = next(
                    (
                        run
                        for run in runs_list
                        if run.example_index == eval_run.example_index
                        and run.repetition == eval_run.repetition_index
                    ),
                    None,
                )

            example_index = run_entry.example_index if run_entry else (eval_run.example_index or 0)
            if eval_run.repetition_index is not None:
                repetition_index = eval_run.repetition_index
            elif run_entry:
                repetition_index = run_entry.repetition
            else:
                repetition_index = 0
            example_id = (
                eval_run.example_id
                or getattr(run_entry, "dataset_example_id", None)
                or str(example_index)
            )
            input_hash = compute_input_hash(run_entry.input) if run_entry else ""
            trace_id = run_entry.trace_id if run_entry else get_current_trace_id()

            evaluator_result = eval_run.result or EvaluationResult()
            documents.append(
                EvaluationScoreDocument(
                    **{
                        "@timestamp": timestamp,
                        "run_id": run_id,
                        "experiment_id": experiment.id or "",
                        "example": ExampleInfo(
                            id=example_id,
                            index=example_index,
                            input_hash=input_hash,
                            dataset=DatasetInfo(id=dataset_id, name=dataset_name),
                        ),
                        "task": TaskInfo(
                            trace_id=trace_id,
                            repetition_index=repetition_index,
                            model=task_model,
                        ),
                        "evaluator": EvaluatorInfo(
                            name=eval_run.name,
                            score=evaluator_result.score,
                            label=evaluator_result.label,
                            explanation=evaluator_result.explanation,
                            metadata=evaluator_result.metadata,
                            trace_id=eval_run.trace_id or get_current_trace_id(),
                            model=evaluator_model,
                        ),
                        "run_metadata": RunMetadata(
                            git_branch=git_metadata.branch,
                            git_commit_sha=git_metadata.commit_sha,
                            total_repetitions=total_repetitions,
                        ),
                        "environment": EnvironmentInfo(hostname=host_name),
                    }
                )
            )

    return documents
