"""Phoenix Experiments exporter for elastic-evals.

This module provides functionality to export completed experiments to Arize Phoenix
for persistence and visualization in the Phoenix Experiments UI.

Requires the phoenix optional dependency:
    pip install elastic-evals[phoenix]
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from elastic_evals.datasets.phoenix import PhoenixDatasetConfig, _get_async_phoenix_client
from elastic_evals.types import EvaluationDataset, EvaluationRun, RanExperiment, RunData

logger = logging.getLogger(__name__)


class PhoenixExperimentResult(BaseModel):
    """Result of exporting an experiment to Phoenix."""

    experiment_id: str
    experiment_url: str | None = None
    dataset_id: str
    task_runs_exported: int
    evaluation_runs_exported: int


class PhoenixExperimentExporter:
    """Export completed experiments to Phoenix for persistence and visualization.

    This exporter syncs experiment results to Phoenix after they have been
    executed by elastic-evals. It uses passthrough wrappers to submit
    pre-computed task outputs and evaluation scores to Phoenix.

    Example:
        >>> exporter = PhoenixExperimentExporter()
        >>> result = await exporter.export_experiment(
        ...     experiment=ran_experiment,
        ...     dataset=dataset,
        ...     experiment_name="My Evaluation Run",
        ... )
        >>> print(f"View in Phoenix: {result.experiment_url}")
    """

    def __init__(self, config: PhoenixDatasetConfig | None = None) -> None:
        """Initialize the exporter.

        Args:
            config: Phoenix connection configuration. Uses environment variables
                if not provided (PHOENIX_BASE_URL, PHOENIX_API_KEY).
        """
        self._config = config

    async def export_experiment(
        self,
        experiment: RanExperiment,
        dataset: EvaluationDataset,
        experiment_name: str | None = None,
        experiment_description: str | None = None,
        experiment_metadata: dict[str, Any] | None = None,
    ) -> PhoenixExperimentResult:
        """Export a completed experiment to Phoenix.

        This method creates a Phoenix experiment linked to the original dataset
        and submits all task outputs and evaluation scores.

        Args:
            experiment: The completed experiment from elastic-evals
            dataset: The evaluation dataset (should have phoenix_dataset_id in metadata)
            experiment_name: Name for the Phoenix experiment (defaults to run_id)
            experiment_description: Description for the Phoenix experiment
            experiment_metadata: Additional metadata to attach to the experiment

        Returns:
            PhoenixExperimentResult with experiment ID, URL, and export counts

        Raises:
            ImportError: If phoenix client is not installed
            ValueError: If the Phoenix dataset cannot be found
        """
        client = _get_async_phoenix_client(self._config)

        # Get Phoenix dataset
        phoenix_dataset = await self._get_phoenix_dataset(client, dataset)
        # Phoenix Dataset can be a dict or an object depending on the client version
        if isinstance(phoenix_dataset, dict):
            phoenix_dataset_id = phoenix_dataset.get("id", "")
        else:
            phoenix_dataset_id = getattr(phoenix_dataset, "id", "")

        # Build experiment metadata
        exp_metadata = {
            "elastic_evals_experiment_id": experiment.id,
            "elastic_evals_dataset_id": experiment.dataset_id,
            "source": "elastic-evals",
        }
        if experiment.experiment_metadata:
            run_id = experiment.experiment_metadata.get("run_id")
            if run_id:
                exp_metadata["elastic_evals_run_id"] = run_id
            model_info = experiment.experiment_metadata.get("model")
            if model_info:
                exp_metadata["model"] = model_info
        if experiment_metadata:
            exp_metadata.update(experiment_metadata)

        # Build name
        exp_name = experiment_name
        if not exp_name:
            run_id = (experiment.experiment_metadata or {}).get("run_id", experiment.id)
            exp_name = f"elastic-evals: {run_id}"

        # Create mapping of (example_index, repetition) -> RunData
        runs_by_key = self._build_runs_index(experiment)

        # Create mapping of (example_index, repetition, evaluator_name) -> EvaluationRun
        evals_by_key = self._build_evaluations_index(experiment)

        # Get unique evaluator names
        evaluator_names = list({er.name for er in experiment.evaluation_runs})

        # Build passthrough task that returns pre-computed outputs
        def create_passthrough_task(runs_index: dict[tuple[int, int], RunData]):
            """Create a task that returns pre-computed outputs."""
            # Track which example we're on (Phoenix calls task per example)
            call_counter = {"count": 0}

            def passthrough_task(example: Any) -> dict[str, Any]:
                """Return pre-computed output for this example."""
                idx = call_counter["count"]
                call_counter["count"] += 1

                # For repetition 0 (Phoenix doesn't support repetitions the same way)
                key = (idx, 0)
                if key in runs_index:
                    run_data = runs_index[key]
                    output = run_data.output
                    # Ensure output is JSON serializable
                    if isinstance(output, dict):
                        return output
                    return {"output": output}
                return {"output": None, "error": "No pre-computed output found"}

            return passthrough_task

        # Build passthrough evaluators that return pre-computed scores
        def create_passthrough_evaluator(
            evaluator_name: str,
            evals_index: dict[tuple[int, int, str], EvaluationRun],
        ):
            """Create an evaluator that returns pre-computed scores."""
            call_counter = {"count": 0}

            def passthrough_evaluator(output: Any) -> dict[str, Any]:
                """Return pre-computed evaluation result."""
                idx = call_counter["count"]
                call_counter["count"] += 1

                key = (idx, 0, evaluator_name)
                if key in evals_index:
                    eval_run = evals_index[key]
                    result = eval_run.result
                    if result:
                        return {
                            "score": result.score,
                            "label": result.label,
                            "explanation": result.explanation,
                            "metadata": result.metadata,
                        }
                return {"score": None, "label": "missing", "explanation": "No pre-computed result"}

            return passthrough_evaluator

        # Create evaluators dict for Phoenix
        evaluators_dict = {
            name: create_passthrough_evaluator(name, evals_by_key)
            for name in evaluator_names
        }

        # Run experiment through Phoenix API with passthrough wrappers
        logger.info(f"Exporting experiment to Phoenix: {exp_name}")

        phoenix_experiment = await client.experiments.run_experiment(
            dataset=phoenix_dataset,
            task=create_passthrough_task(runs_by_key),
            evaluators=evaluators_dict if evaluators_dict else None,
            experiment_name=exp_name,
            experiment_description=experiment_description or experiment.dataset_description,
            experiment_metadata=exp_metadata,
            print_summary=False,
        )

        # Extract experiment ID (phoenix_experiment can be dict or object)
        if isinstance(phoenix_experiment, dict):
            phoenix_experiment_id = phoenix_experiment.get("id", "")
        else:
            phoenix_experiment_id = getattr(phoenix_experiment, "id", "")

        # Get experiment URL
        experiment_url = None
        try:
            experiment_url = client.experiments.get_experiment_url(
                dataset_id=phoenix_dataset_id,
                experiment_id=phoenix_experiment_id,
            )
        except Exception:
            # URL generation might fail, that's okay
            pass

        return PhoenixExperimentResult(
            experiment_id=phoenix_experiment_id,
            experiment_url=experiment_url,
            dataset_id=phoenix_dataset_id,
            task_runs_exported=len(runs_by_key),
            evaluation_runs_exported=len(evals_by_key),
        )

    async def _get_phoenix_dataset(
        self,
        client: Any,
        dataset: EvaluationDataset,
    ) -> Any:
        """Get the Phoenix dataset object.

        Tries to get by ID first (if available in metadata), then by name.
        """
        metadata = dataset.metadata or {}

        # Try to get by Phoenix dataset ID
        phoenix_dataset_id = metadata.get("phoenix_dataset_id")
        if phoenix_dataset_id:
            try:
                return await client.datasets.get_dataset(dataset=phoenix_dataset_id)
            except Exception as e:
                logger.warning(f"Could not get Phoenix dataset by ID {phoenix_dataset_id}: {e}")

        # Try to get by name
        phoenix_dataset_name = metadata.get("phoenix_dataset_name") or dataset.name
        try:
            return await client.datasets.get_dataset(dataset=phoenix_dataset_name)
        except Exception as e:
            raise ValueError(
                f"Could not find Phoenix dataset '{phoenix_dataset_name}'. "
                f"Make sure the dataset exists in Phoenix. Error: {e}"
            ) from e

    def _build_runs_index(
        self,
        experiment: RanExperiment,
    ) -> dict[tuple[int, int], RunData]:
        """Build index of runs by (example_index, repetition)."""
        index: dict[tuple[int, int], RunData] = {}
        for run_data in experiment.runs.values():
            key = (run_data.example_index, run_data.repetition)
            index[key] = run_data
        return index

    def _build_evaluations_index(
        self,
        experiment: RanExperiment,
    ) -> dict[tuple[int, int, str], EvaluationRun]:
        """Build index of evaluations by (example_index, repetition, evaluator_name)."""
        index: dict[tuple[int, int, str], EvaluationRun] = {}
        for eval_run in experiment.evaluation_runs:
            if eval_run.example_index is not None and eval_run.repetition_index is not None:
                key = (eval_run.example_index, eval_run.repetition_index, eval_run.name)
                index[key] = eval_run
        return index


async def export_experiment_to_phoenix(
    experiment: RanExperiment,
    dataset: EvaluationDataset,
    experiment_name: str | None = None,
    experiment_description: str | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    config: PhoenixDatasetConfig | None = None,
) -> PhoenixExperimentResult:
    """Export a completed experiment to Phoenix.

    This is a convenience function that creates a PhoenixExperimentExporter
    and exports the experiment.

    Args:
        experiment: The completed experiment from elastic-evals
        dataset: The evaluation dataset (should have phoenix_dataset_id in metadata)
        experiment_name: Name for the Phoenix experiment (defaults to run_id)
        experiment_description: Description for the Phoenix experiment
        experiment_metadata: Additional metadata to attach to the experiment
        config: Phoenix connection configuration (uses env vars if not provided)

    Returns:
        PhoenixExperimentResult with experiment ID, URL, and export counts

    Example:
        >>> from elastic_evals.export.phoenix_experiments import export_experiment_to_phoenix
        >>>
        >>> # After running your experiment...
        >>> result = await export_experiment_to_phoenix(
        ...     experiment=ran_experiment,
        ...     dataset=dataset,
        ...     experiment_name="My Evaluation Run",
        ... )
        >>> print(f"Exported {result.task_runs_exported} runs to Phoenix")
        >>> print(f"View at: {result.experiment_url}")
    """
    exporter = PhoenixExperimentExporter(config=config)
    return await exporter.export_experiment(
        experiment=experiment,
        dataset=dataset,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
        experiment_metadata=experiment_metadata,
    )
