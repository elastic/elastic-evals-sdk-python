"""Workplace questions dataset for chatbot RAG app evaluation."""

from __future__ import annotations

from typing import Any

from elastic_evals.types import Example, EvaluationDataset

WorkplaceQuestionExample = Example[dict[str, str], dict[str, list[str]], dict[str, Any]]

workplace_questions_dataset: EvaluationDataset[WorkplaceQuestionExample] = (
    EvaluationDataset(
        name="chatbot-rag-app: workplace-questions",
        description="Two-question dataset covering positive and negative retrieval cases.",
        examples=[
            Example(
                input={"question": "What is our working from home policy?"},
                output={"expected_sources": ["Work From Home Policy"]},
                metadata={
                    "query_intent": "Factual",
                    "criteria": [
                        "The answer references the Work From Home Policy document",
                        "The answer mentions eligibility, equipment, or workspace expectations",
                        "The answer ends with a SOURCES: line citing at least one document",
                    ],
                },
            ),
            Example(
                input={"question": "What's the NASA sales team?"},
                output={"expected_sources": list[str]()},
                metadata={
                    "query_intent": "Factual",
                    "criteria": [
                        "The answer admits it doesn't know or finds no relevant information",
                        "The answer does not fabricate details about a NASA sales team",
                        "The answer does not invent source citations",
                    ],
                },
            ),
        ],
    )
)
