"""Configuration helpers for elastic-evals."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError, field_validator

from elastic_evals.tracing import ExporterConfig, TracingConfig
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


def _parse_tracing_exporters() -> list[ExporterConfig]:
    """Parse tracing exporter configuration from environment variables.

    Supports multiple configuration formats:
    1. JSON array: ELASTIC_EVALS_TRACING_EXPORTERS='[{"type":"otlp",...},{"type":"phoenix",...}]'
    2. Comma-separated targets: ELASTIC_EVALS_TRACING_TARGETS=otlp,phoenix
    3. Legacy single exporter: ELASTIC_EVALS_TRACING_EXPORTER=otlp

    Phoenix-specific env vars (used when "phoenix" is in targets):
    - PHOENIX_COLLECTOR_ENDPOINT: Phoenix server URL (default: http://localhost:6006)
    - PHOENIX_PROJECT_NAME: Project name in Phoenix
    - PHOENIX_API_KEY: API key for Phoenix Cloud
    - ELASTIC_EVALS_PHOENIX_USE_GRPC: Use gRPC transport (default: false)
    """
    # Option 1: Full JSON configuration
    exporters_json = os.environ.get("ELASTIC_EVALS_TRACING_EXPORTERS")
    if exporters_json:
        try:
            exporters_data = json.loads(exporters_json)
        except json.JSONDecodeError as exc:
            raise ValueError("ELASTIC_EVALS_TRACING_EXPORTERS must be valid JSON") from exc

        if not isinstance(exporters_data, list):
            raise ValueError("ELASTIC_EVALS_TRACING_EXPORTERS must be a JSON array")

        try:
            return [ExporterConfig(**exp) for exp in exporters_data]
        except TypeError as exc:
            raise ValueError(
                "ELASTIC_EVALS_TRACING_EXPORTERS items must be objects, not primitives"
            ) from exc
        except ValidationError as exc:
            raise ValueError(
                f"Invalid exporter config in ELASTIC_EVALS_TRACING_EXPORTERS: {exc.errors()[0]['msg']}"
            ) from exc

    # Option 2: Comma-separated targets (simplified)
    targets = os.environ.get("ELASTIC_EVALS_TRACING_TARGETS")
    if targets:
        exporters = []
        for target in targets.split(","):
            target = target.strip().lower()
            if target == "phoenix":
                exporters.append(_create_phoenix_exporter_config())
            elif target == "otlp":
                exporters.append(_create_otlp_exporter_config())
            elif target == "console":
                exporters.append(ExporterConfig(type="console"))
            elif target:
                raise ValueError(
                    f"Unknown tracing target '{target}'. "
                    "Valid targets: otlp, phoenix, console"
                )
        return exporters

    # Option 3: Legacy single exporter
    exporter = os.environ.get("ELASTIC_EVALS_TRACING_EXPORTER", "otlp")
    if exporter == "none":
        return []
    if exporter == "phoenix":
        return [_create_phoenix_exporter_config()]
    if exporter == "otlp":
        return [_create_otlp_exporter_config()]
    if exporter == "console":
        return [ExporterConfig(type="console")]

    raise ValueError(
        "ELASTIC_EVALS_TRACING_EXPORTER must be one of otlp, phoenix, console, none"
    )


def _create_phoenix_exporter_config() -> ExporterConfig:
    """Create Phoenix exporter config from environment variables."""
    return ExporterConfig(
        type="phoenix",
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
        project_name=os.environ.get("PHOENIX_PROJECT_NAME"),
        api_key=os.environ.get("PHOENIX_API_KEY"),
        use_grpc=_parse_bool(
            os.environ.get("ELASTIC_EVALS_PHOENIX_USE_GRPC"),
            name="ELASTIC_EVALS_PHOENIX_USE_GRPC",
            default=False,
        ),
    )


def _create_otlp_exporter_config() -> ExporterConfig:
    """Create OTLP exporter config from environment variables."""
    return ExporterConfig(
        type="otlp",
        endpoint=os.environ.get(
            "ELASTIC_EVALS_TRACING_ENDPOINT", "http://localhost:4318/v1/traces"
        ),
    )


class ElasticEvalsConfig(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    repetitions: int = 3
    concurrency: int = 5

    kibana_url: str = "http://localhost:5601"
    connector_id: str
    evaluator_connector_id: str | None = None
    kibana_auth: str

    evaluations_es_url: str | None = None
    trace_es_url: str | None = None

    tracing: TracingConfig = Field(default_factory=TracingConfig)

    # Phoenix experiment export
    phoenix_experiment_export: bool = False

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    model: dict[str, Any] | None = None

    @property
    def logger(self) -> logging.Logger:
        return setup_logging(self.log_level)

    @classmethod
    def from_env(cls) -> "ElasticEvalsConfig":
        run_id = os.environ.get("ELASTIC_EVALS_RUN_ID") or str(uuid.uuid4())
        repetitions = _parse_int(os.environ.get("ELASTIC_EVALS_REPETITIONS", "3"), name="repetitions")
        concurrency = _parse_int(os.environ.get("ELASTIC_EVALS_CONCURRENCY", "5"), name="concurrency")
        kibana_url = os.environ.get("KIBANA_URL", "http://localhost:5601")
        connector_id = _get_required_env("CONNECTOR_ID")
        evaluator_connector_id = os.environ.get("EVALUATION_CONNECTOR_ID")
        kibana_auth = _get_required_env("KIBANA_AUTH")
        evaluations_es_url = os.environ.get("EVALUATIONS_ES_URL")
        trace_es_url = os.environ.get("TRACE_ES_URL")

        log_level = os.environ.get("ELASTIC_EVALS_LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("ELASTIC_EVALS_LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR")

        model_payload = os.environ.get("ELASTIC_EVALS_MODEL")
        model = None
        if model_payload:
            try:
                model = json.loads(model_payload)
            except json.JSONDecodeError as exc:
                raise ValueError("ELASTIC_EVALS_MODEL must be valid JSON") from exc

        # Parse multi-exporter tracing configuration
        tracing_enabled = _parse_bool(
            os.environ.get("ELASTIC_EVALS_TRACING_ENABLED"),
            name="ELASTIC_EVALS_TRACING_ENABLED",
            default=True,
        )
        exporters = _parse_tracing_exporters() if tracing_enabled else []

        tracing = TracingConfig(
            enabled=tracing_enabled,
            exporters=exporters,
            service_name=os.environ.get("ELASTIC_EVALS_TRACING_SERVICE_NAME", "elastic-evals"),
            run_id=run_id,
        )

        # Phoenix experiment export
        phoenix_experiment_export = _parse_bool(
            os.environ.get("ELASTIC_EVALS_PHOENIX_EXPERIMENT_EXPORT"),
            name="ELASTIC_EVALS_PHOENIX_EXPERIMENT_EXPORT",
            default=False,
        )

        return cls(
            run_id=run_id,
            repetitions=repetitions,
            concurrency=concurrency,
            kibana_url=kibana_url,
            connector_id=connector_id,
            evaluator_connector_id=evaluator_connector_id,
            kibana_auth=kibana_auth,
            evaluations_es_url=evaluations_es_url,
            trace_es_url=trace_es_url,
            tracing=tracing,
            phoenix_experiment_export=phoenix_experiment_export,
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
