"""RAG evaluation metric helpers."""

from __future__ import annotations

from elastic_evals.evaluators.rag.types import GroundTruth, RetrievedDoc

DEFAULT_RELEVANCE_THRESHOLD = 1


def is_relevant(doc: RetrievedDoc, ground_truth: GroundTruth, threshold: float) -> bool:
    index_ground_truth = ground_truth.get(doc.index)
    if not index_ground_truth:
        return False
    score = index_ground_truth.get(doc.id)
    return score is not None and score >= threshold


def get_relevant_docs(
    retrieved_docs: list[RetrievedDoc], ground_truth: GroundTruth, threshold: float
) -> list[RetrievedDoc]:
    return [doc for doc in retrieved_docs if is_relevant(doc, ground_truth, threshold)]


def count_relevant_in_ground_truth(ground_truth: GroundTruth, threshold: float) -> int:
    count = 0
    for index_docs in ground_truth.values():
        count += sum(1 for score in index_docs.values() if score >= threshold)
    return count


def filter_docs_by_ground_truth_indices(
    docs: list[RetrievedDoc], ground_truth: GroundTruth
) -> list[RetrievedDoc]:
    indices = set(ground_truth.keys())
    return [doc for doc in docs if doc.index in indices]


def calculate_precision(hits: int, k: int) -> float:
    """Precision@K = relevant in top K / K."""
    return hits / k if k > 0 else 0.0


def calculate_recall(hits: int, total_relevant: int) -> float:
    """Recall@K = relevant in top K / total relevant."""
    return hits / total_relevant if total_relevant > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """F1@K = harmonic mean of precision and recall."""
    if precision + recall <= 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)
