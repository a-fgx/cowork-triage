"""
Intake Node

The first node in the workflow. It processes the user's raw bug report
and extracts structured information.

Input State:
    - messages: Contains the user's bug report as the last message

Output State Updates:
    - bug_report: Structured BugReport with extracted fields
    - current_phase: "classification" (next phase)
    - missing_info: [] (will be populated by classifier)
    - info_gathering_attempts: 0

How LangGraph Nodes Work:
    1. Node receives the full state as input
    2. Node does its work (LLM calls, processing, etc.)
    3. Node returns a dict with ONLY the fields to update
    4. LangGraph merges the updates into the state

Example:
    def my_node(state: DiagnosticState) -> dict:
        # Read from state
        messages = state.get("messages", [])

        # Do work...
        result = process(messages)

        # Return updates (NOT the full state)
        return {
            "my_field": result,
            "current_phase": "next_phase"
        }
"""

import json
import re
from typing import Any

from src.state import DiagnosticState, BugReport
from src.llm import get_llm, invoke_with_system
from src.prompts.templates import INTAKE_PROMPT


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response that may contain markdown or extra text.
    """
    if not response or not response.strip():
        return {}

    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in the text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def intake_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Process the initial bug report and extract structured information.

    This is the entry point of the diagnostic workflow. It takes the raw
    user message and extracts structured fields like error messages,
    steps to reproduce, environment details, etc.

    Args:
        state: The current graph state (contains messages)

    Returns:
        State updates with structured bug_report
    """
    # Get the conversation messages
    messages = state.get("messages", [])

    # Find the last user message (this is the bug report)
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message:
        # No user message found - return empty bug report
        return {
            "bug_report": {"raw_description": ""},
            "current_phase": "classification",
            "missing_info": [],
            "info_gathering_attempts": 0,
            "related_issues": [],
            "rag_results": [],
            "hypotheses": [],
            "resolution_plan": [],
            "needs_user_input": False,
        }

    # Use the LLM to extract structured information
    llm = get_llm()

    response = invoke_with_system(
        llm,
        system_prompt=INTAKE_PROMPT,
        user_message=user_message,
    )

    # Parse the JSON response (handles markdown code blocks, etc.)
    extracted = extract_json_from_response(response)

    # Build the structured bug report
    bug_report: BugReport = {
        "raw_description": user_message,
        # Add extracted fields if present (use .get() for safety)
        **({"title": extracted["title"]} if extracted.get("title") else {}),
        **(
            {"steps_to_reproduce": extracted["steps_to_reproduce"]}
            if extracted.get("steps_to_reproduce")
            else {}
        ),
        **(
            {"expected_behavior": extracted["expected_behavior"]}
            if extracted.get("expected_behavior")
            else {}
        ),
        **(
            {"actual_behavior": extracted["actual_behavior"]}
            if extracted.get("actual_behavior")
            else {}
        ),
        **(
            {"environment": extracted["environment"]}
            if extracted.get("environment")
            else {}
        ),
        **(
            {"error_message": extracted["error_message"]}
            if extracted.get("error_message")
            else {}
        ),
        **(
            {"stack_trace": extracted["stack_trace"]}
            if extracted.get("stack_trace")
            else {}
        ),
    }

    # Return state updates
    return {
        "bug_report": bug_report,
        "current_phase": "classification",
        "missing_info": [],
        "info_gathering_attempts": 0,
        "related_issues": [],
        "rag_results": [],
        "hypotheses": [],
        "resolution_plan": [],
        "needs_user_input": False,
    }
