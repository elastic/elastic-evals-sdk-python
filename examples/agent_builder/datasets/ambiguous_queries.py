# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Ambiguous query dataset for Agent Builder."""

from __future__ import annotations

from elastic_evals.types import EvaluationDataset, Example

ambiguous_queries_dataset = EvaluationDataset(
    name="agent builder: default-agent-ambiguous-queries",
    description="Dataset containing ambiguous queries that should trigger clarification requests.",
    examples=[
        Example(
            input={"question": "List projects which are unhealthy?"},
            output={"expected": "Can you clarify how to determine if a project is unhealthy"},
            metadata={"query_intent": "Factual"},
        ),
        Example(
            input={"question": "Can I get a list of our most active users from last week?"},
            output={"expected": "What is the definition of an active user?"},
            metadata={"query_intent": "Investigative"},
        ),
        Example(
            input={"question": "Who are our top-performing support agents?"},
            output={"expected": "How do you define performance for an agent"},
            metadata={"query_intent": "Investigative"},
        ),
        Example(
            input={"question": "I would like to view my invoices."},
            output={"expected": "Can you clarify which invoices would you like to see?"},
            metadata={"query_intent": "Procedural"},
        ),
    ],
)
