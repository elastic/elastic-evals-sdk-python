# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Constants for the Kibana Agent Builder integration."""

from __future__ import annotations

AGENT_BUILDER_API_VERSION = "2023-10-31"

AGENTS_URL = "/api/agent_builder/agents"
AGENT_URL = "/api/agent_builder/agents/{agent_id}"
CONVERSE_URL = "/api/agent_builder/converse"
TOOLS_URL = "/api/agent_builder/tools"
TOOL_URL = "/api/agent_builder/tools/{tool_id}"
