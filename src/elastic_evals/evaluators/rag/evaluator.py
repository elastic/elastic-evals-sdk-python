"""RAG evaluators."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeVar

from elastic_evals.evaluators.base import SimpleEvaluator
from elastic_evals.evaluators.rag.metrics import (
    DEFAULT_RELEVANCE_THRESHOLD,
    calculate_f1,
    calculate_precision,
    calculate_recall,
    count_relevant_in_ground_truth,
    filter_docs_by_ground_truth_indices,
    get_relevant_docs,
)
from elastic_evals.evaluators.rag.types import (
    GroundTruth,
    RagEvaluatorConfig,
    RetrievedDoc,
)
from elastic_evals.types import EvaluationResult, Evaluator, EvaluatorParams

TOutput = TypeVar("TOutput")
TReferenceOutput = TypeVar("TReferenceOutput")

PRECISION_EVALUATOR_NAME = "Precision@K"
RECALL_EVALUATOR_NAME = "Recall@K"
F1_EVALUATOR_NAME = "F1@K"


def _should_filter_by_ground_truth_indices(config: RagEvaluatorConfig) -> bool:
    if config.filter_by_ground_truth_indices is not None:
        return config.filter_by_ground_truth_indices
    return os.environ.get("INDEX_FOCUSED_RAG_EVAL") == "true"


def _get_effective_k(config_k: int) -> int:
    env_k = os.environ.get("RAG_EVAL_K")
    if env_k is not None:
        try:
            parsed = int(env_k)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return config_k


@dataclass(frozen=True)
class RagMetrics:
    precision: float
    recall: float
    f1: float
    hits: int
    k: int
    total_relevant: int


def _compute_rag_metrics(
    config: RagEvaluatorConfig[TOutput, TReferenceOutput],
    output: TOutput,
    reference_output: TReferenceOutput | None,
) -> RagMetrics | None:
    if reference_output is None:
        return None

    k = _get_effective_k(config.k)
    threshold = (
        DEFAULT_RELEVANCE_THRESHOLD
        if config.relevance_threshold is None
        else config.relevance_threshold
    )
    ground_truth: GroundTruth = config.extract_ground_truth(reference_output)

    if not ground_truth:
        return None

    all_retrieved_docs: list[RetrievedDoc] = config.extract_retrieved_docs(output)
    if _should_filter_by_ground_truth_indices(config):
        all_retrieved_docs = filter_docs_by_ground_truth_indices(
            all_retrieved_docs, ground_truth
        )

    top_k_docs = all_retrieved_docs[:k]
    relevant_in_top_k = get_relevant_docs(top_k_docs, ground_truth, threshold)
    hits = len(relevant_in_top_k)
    total_relevant = count_relevant_in_ground_truth(ground_truth, threshold)

    precision = calculate_precision(hits, k)
    recall = calculate_recall(hits, total_relevant)
    f1 = calculate_f1(precision, recall)

    return RagMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hits=hits,
        k=k,
        total_relevant=total_relevant,
    )


def create_precision_at_k_evaluator(
    config: RagEvaluatorConfig[TOutput, TReferenceOutput],
) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        try:
            metrics = _compute_rag_metrics(config, params.output, params.expected)
        except Exception as error:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation=f"Precision@K evaluation failed: {error}",
            )

        if not metrics:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation="No ground truth available for Precision@K evaluation",
            )

        return EvaluationResult(
            score=metrics.precision,
            explanation=(
                f"{metrics.hits} relevant docs in top {metrics.k} "
                f"(Precision: {metrics.precision * 100:.1f}%)"
            ),
            metadata={"hits": metrics.hits, "k": metrics.k},
        )

    return SimpleEvaluator(
        name=PRECISION_EVALUATOR_NAME, kind="CODE", evaluate=evaluate
    )


def create_recall_at_k_evaluator(
    config: RagEvaluatorConfig[TOutput, TReferenceOutput],
) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        try:
            metrics = _compute_rag_metrics(config, params.output, params.expected)
        except Exception as error:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation=f"Recall@K evaluation failed: {error}",
            )

        if not metrics:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation="No ground truth available for Recall@K evaluation",
            )

        return EvaluationResult(
            score=metrics.recall,
            explanation=(
                f"{metrics.hits} of {metrics.total_relevant} relevant docs retrieved "
                f"(Recall: {metrics.recall * 100:.1f}%)"
            ),
            metadata={"hits": metrics.hits, "totalRelevant": metrics.total_relevant},
        )

    return SimpleEvaluator(name=RECALL_EVALUATOR_NAME, kind="CODE", evaluate=evaluate)


def create_f1_at_k_evaluator(
    config: RagEvaluatorConfig[TOutput, TReferenceOutput],
) -> Evaluator:
    async def evaluate(params: EvaluatorParams) -> EvaluationResult:
        try:
            metrics = _compute_rag_metrics(config, params.output, params.expected)
        except Exception as error:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation=f"F1@K evaluation failed: {error}",
            )

        if not metrics:
            return EvaluationResult(
                score=None,
                label="unavailable",
                explanation="No ground truth available for F1@K evaluation",
            )

        return EvaluationResult(
            score=metrics.f1,
            explanation=(
                f"F1@{metrics.k}: {metrics.f1 * 100:.1f}% "
                f"(P: {metrics.precision * 100:.1f}%, "
                f"R: {metrics.recall * 100:.1f}%)"
            ),
            metadata={
                "precision": metrics.precision,
                "recall": metrics.recall,
                "hits": metrics.hits,
                "k": metrics.k,
                "totalRelevant": metrics.total_relevant,
            },
        )

    return SimpleEvaluator(name=F1_EVALUATOR_NAME, kind="CODE", evaluate=evaluate)


def create_rag_evaluators(
    config: RagEvaluatorConfig[TOutput, TReferenceOutput],
) -> list[Evaluator]:
    return [
        create_precision_at_k_evaluator(config),
        create_recall_at_k_evaluator(config),
        create_f1_at_k_evaluator(config),
    ]
