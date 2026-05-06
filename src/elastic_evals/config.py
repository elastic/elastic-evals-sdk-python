"""Configuration helpers for elastic-evals."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

from elastic_evals.tracing import TracingConfig
from elastic_evals.utils.logging import setup_logging


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise KeyError(f'Missing required env var "{name}"')
    return value


def _parse_int(value: str, *, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_bool(value: str | None, *, name: str, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


class ElasticEvalsConfig(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    repetitions: int = 1
    concurrency: int = 5

    kibana_url: str = "http://localhost:5601"
    kibana_api_key: str | None = None
    connector_id: str
    evaluator_connector_id: str | None = None

    trace_es_url: str | None = None

    tracing: TracingConfig = Field(default_factory=TracingConfig)

    suite_id: str | None = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model: dict[str, Any] | None = None

    @property
    def logger(self) -> logging.Logger:
        return setup_logging(self.log_level)

    @classmethod
    def from_env(cls) -> "ElasticEvalsConfig":
        run_id = os.environ.get("ELASTIC_EVALS_RUN_ID") or str(uuid.uuid4())
        repetitions = _parse_int(
            os.environ.get("ELASTIC_EVALS_REPETITIONS", "1"), name="repetitions"
        )
        concurrency = _parse_int(
            os.environ.get("ELASTIC_EVALS_CONCURRENCY", "5"), name="concurrency"
        )
        kibana_url = os.environ.get("KIBANA_URL", "http://localhost:5601")
        kibana_api_key = os.environ.get("KIBANA_API_KEY")
        connector_id = _get_required_env("CONNECTOR_ID")
        evaluator_connector_id = os.environ.get("EVALUATION_CONNECTOR_ID")
        trace_es_url = os.environ.get("TRACE_ES_URL")

        log_level = os.environ.get("ELASTIC_EVALS_LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError(
                "ELASTIC_EVALS_LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR"
            )

        suite_id = os.environ.get("ELASTIC_EVALS_SUITE_ID")

        model_payload = os.environ.get("ELASTIC_EVALS_MODEL")
        model = None
        if model_payload:
            try:
                model = json.loads(model_payload)
            except json.JSONDecodeError as exc:
                raise ValueError("ELASTIC_EVALS_MODEL must be valid JSON") from exc

        tracing_enabled = _parse_bool(
            os.environ.get("ELASTIC_EVALS_TRACING_ENABLED"),
            name="ELASTIC_EVALS_TRACING_ENABLED",
            default=True,
        )
        tracing = TracingConfig(
            enabled=tracing_enabled,
            endpoint=os.environ.get(
                "ELASTIC_EVALS_TRACING_ENDPOINT", "http://localhost:4318/v1/traces"
            ),
            service_name=os.environ.get(
                "ELASTIC_EVALS_TRACING_SERVICE_NAME", "elastic-evals"
            ),
            run_id=run_id,
        )

        return cls(
            run_id=run_id,
            repetitions=repetitions,
            concurrency=concurrency,
            kibana_url=kibana_url,
            kibana_api_key=kibana_api_key,
            connector_id=connector_id,
            evaluator_connector_id=evaluator_connector_id,
            trace_es_url=trace_es_url,
            suite_id=suite_id,
            tracing=tracing,
            log_level=cast(Literal["DEBUG", "INFO", "WARNING", "ERROR"], log_level),
            model=model,
        )

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Concurrency must be at least 1")
        return value

    @field_validator("repetitions")
    @classmethod
    def validate_repetitions(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Repetitions must be at least 1")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        return value.upper()
