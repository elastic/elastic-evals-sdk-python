"""Export utilities for elastic-evals."""

from .documents import EvaluationScoreDocument
from .git_metadata import GitMetadata, get_git_metadata
from .repository import (
    EvaluationScoreRepository,
    build_flattened_score_documents,
    compute_input_hash,
)

# Phoenix exports are optional - only available if phoenix is installed
try:
    from .phoenix_experiments import (
        PhoenixExperimentExporter,
        PhoenixExperimentResult,
        export_experiment_to_phoenix,
    )

    _PHOENIX_EXPORTS = [
        "PhoenixExperimentExporter",
        "PhoenixExperimentResult",
        "export_experiment_to_phoenix",
    ]
except ImportError:
    _PHOENIX_EXPORTS = []

__all__ = [
    "EvaluationScoreDocument",
    "EvaluationScoreRepository",
    "GitMetadata",
    "build_flattened_score_documents",
    "compute_input_hash",
    "get_git_metadata",
    *_PHOENIX_EXPORTS,
]
