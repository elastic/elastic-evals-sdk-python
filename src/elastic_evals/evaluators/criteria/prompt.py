"""Prompt templates for criteria evaluation."""

from __future__ import annotations

from typing import Any

PROMPT_NAME = "llm_criteria_evaluation"
PROMPT_DESCRIPTION = "Prompt for evaluating LLM outputs against criteria"

SYSTEM_PROMPT = """You are an automated quality-assurance LLM.

## Task
For each criterion listed below, decide whether the provided **output** satisfies the requirement, fails to satisfy it, or is not applicable for the given example.

Return your decision by calling the **score** tool exactly once.

Each decision must be one of:
- **PASS** – the output clearly meets the criterion.
- **FAIL** – the output clearly violates or misses the criterion.
- **N/A** – the criterion does not apply to this example.

## Examples

### Example A
*Criterion:* `syntax-valid`  
*Input / Output:* A well-formed SQL string.  
*Decision:* **PASS**  

```json
{
  "criteria": [
    { "id": "syntax-valid", "result": "PASS" }
  ]
}
````

### Example B

*Criterion:* `contains-latency-agg`
*Input / Output:* Query lacks any latency aggregation.
*Decision:* **FAIL**

```json
{
  "criteria": [
    { "id": "contains-latency-agg", "result": "FAIL" }
  ]
}
```

### Example C

*Criterion:* `requires-image`
*Input / Output:* Text-only request — no images involved.
*Decision:* **N/A**

```json
{
  "criteria": [
    { "id": "requires-image", "result": "N/A" }
  ]
}
```

When you score the real example, follow exactly the same JSON format—no additional keys, no commentary outside the tool call.

## Criteria to score
{{#criteria}}
- {{{.}}}
{{/criteria}}
"""

USER_PROMPT = """Here is the example to evaluate:

**Input**  
{{{input}}}

**Output**  
{{{output}}}

**Metadata**  
{{{metadata}}}

Please review the output against each of the criteria provided in the system instructions and respond by invoking the `score` tool.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score",
            "description": "Return PASS, FAIL, or N/A for every evaluation criterion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "The unique identifier of the criterion.",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Briefly explain the reasoning behind your judgement",
                                },
                                "result": {
                                    "type": "string",
                                    "description": "Outcome of evaluating the criterion.",
                                    "enum": ["PASS", "FAIL", "N/A"],
                                },
                            },
                            "required": ["id", "result"],
                        },
                        "description": "A verdict for every criterion.",
                    }
                },
                "required": ["criteria"],
            },
        },
    }
]

INFERENCE_TOOLS = {
    tool["function"]["name"]: {
        "description": tool["function"]["description"],
        "schema": tool["function"]["parameters"],
    }
    for tool in TOOLS
}


def render_system_prompt(criteria: list[str]) -> str:
    lines = "\n".join(f"- {criterion}" for criterion in criteria)
    return SYSTEM_PROMPT.replace("{{#criteria}}\n- {{{.}}}\n{{/criteria}}", lines)


def render_user_prompt(input_text: str, output_text: str, metadata_text: str) -> str:
    return (
        USER_PROMPT.replace("{{{input}}}", input_text)
        .replace("{{{output}}}", output_text)
        .replace("{{{metadata}}}", metadata_text)
    )


def tool_choice() -> dict[str, Any]:
    return {"function": "score"}


def build_prompt(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {
        "name": PROMPT_NAME,
        "description": PROMPT_DESCRIPTION,
        "versions": [
            {
                "system": system_prompt,
                "template": {"static": {"content": user_prompt}},
                "tools": INFERENCE_TOOLS,
            }
        ],
    }
