#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ ! -f .env ]]; then
  echo "Missing .env in ${SCRIPT_DIR}"
  echo "Create it first: cp .env.example .env"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found on PATH"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose is required but unavailable"
  exit 1
fi

if ! curl -sf "${ELASTICSEARCH_URL:-http://localhost:9200}" >/dev/null 2>&1; then
  echo "Warning: Elasticsearch reachability check failed at ${ELASTICSEARCH_URL:-http://localhost:9200}"
  echo "Continuing anyway (auth/network may intentionally block this probe)."
fi

docker compose --env-file .env up -d --wait

echo "App: http://localhost:4000   |   Eval: uv run elastic-evals run --suite chatbot-rag-eval-example ..."
