"""Compare evaluation runs from Elasticsearch."""

from __future__ import annotations

import asyncio

import click
from elasticsearch import AsyncElasticsearch

from elastic_evals.export import EvaluationScoreRepository
from elastic_evals.reporting.compare import (
    compare_runs,
    pair_scores,
    print_comparison_report,
)
from elastic_evals.utils.logging import setup_logging


@click.command("compare")
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option(
    "--es-url", envvar="EVALUATIONS_ES_URL", required=True, help="Elasticsearch URL."
)
@click.option(
    "--significance", default=0.05, show_default=True, help="P-value threshold."
)
def compare_cmd(run_id_1: str, run_id_2: str, es_url: str, significance: float) -> None:
    """Compare two evaluation runs using paired t-test."""
    log = setup_logging()

    async def _compare() -> None:
        es = AsyncElasticsearch(es_url)
        repo = EvaluationScoreRepository(es, log)

        scores_a, scores_b = await asyncio.gather(
            repo.get_scores_by_run_id(run_id_1),
            repo.get_scores_by_run_id(run_id_2),
        )

        if not scores_a:
            raise click.ClickException(f"No scores found for run ID: {run_id_1}")
        if not scores_b:
            raise click.ClickException(f"No scores found for run ID: {run_id_2}")

        datasets_a = {
            score.example.dataset.id: score.example.dataset.name for score in scores_a
        }
        datasets_b = {
            score.example.dataset.id: score.example.dataset.name for score in scores_b
        }
        overlap = set(datasets_a).intersection(datasets_b)
        if not overlap:
            raise click.ClickException(
                "No overlapping datasets found between the two runs."
            )

        filtered_a = [
            score for score in scores_a if score.example.dataset.id in overlap
        ]
        filtered_b = [
            score for score in scores_b if score.example.dataset.id in overlap
        ]

        paired = pair_scores(filtered_a, filtered_b)
        if not paired.pairs:
            raise click.ClickException("No paired scores found between the two runs.")

        click.echo(
            f"Paired {len(paired.pairs)} scores "
            f"(skipped {paired.skipped_missing_pairs} missing pairs, "
            f"{paired.skipped_null_scores} null scores)."
        )

        results = compare_runs(filtered_a, filtered_b)
        if not results:
            click.echo("No t-test results returned.")
            return

        print_comparison_report(
            results=results,
            run_id_a=run_id_1,
            run_id_b=run_id_2,
            significance_threshold=significance,
        )

        await es.close()

    asyncio.run(_compare())
