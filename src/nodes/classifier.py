"""
Classifier Node

Classifies the bug report into failure types and identifies missing information.

Input State:
    - bug_report: Structured bug report from intake

Output State Updates:
    - classification: Classification with failure_type, confidence, reasoning
    - missing_info: List of critical missing information
    - current_phase: "gathering" if info missing, else "searching"

Key Decision:
    This node determines whether we need to ask the user for more information
    or can proceed directly to diagnosis.
"""

import json
import re
from typing import Any

from src.state import DiagnosticState, Classification
from src.llm import get_llm, invoke_with_system
from src.prompts.templates import CLASSIFICATION_PROMPT


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response that may contain markdown or extra text.
    """
    if not response or not response.strip():
        raise ValueError("Empty response from LLM")

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

    raise ValueError(f"Could not extract JSON from response: {response[:200]}...")


def classifier_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Classify the bug report and identify missing information.

    Analyzes the structured bug report to:
    1. Determine the failure type (api, version, dependency, etc.)
    2. Assess confidence in the classification
    3. Identify critical missing information

    Args:
        state: The current graph state

    Returns:
        State updates with classification and missing_info
    """
    bug_report = state.get("bug_report", {})

    # Prepare the bug report for the LLM
    bug_report_text = json.dumps(bug_report, indent=2)

    # Use the LLM to classify
    llm = get_llm()

    try:
        response = invoke_with_system(
            llm,
            system_prompt=CLASSIFICATION_PROMPT,
            user_message=f"Bug Report:\n{bug_report_text}",
        )

        # Parse the JSON response
        result = extract_json_from_response(response)

        classification: Classification = {
            "failure_type": result.get("failure_type", "unknown"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }

        missing_info = result.get("missing_info", [])

    except (json.JSONDecodeError, KeyError, ValueError):
        # If parsing fails, default to unknown
        classification: Classification = {
            "failure_type": "unknown",
            "confidence": 0.3,
            "reasoning": "Failed to parse classification response",
        }
        missing_info = ["error message", "steps to reproduce"]

    # Decide next phase based on missing info and confidence
    # If we're missing critical info and confidence is low, ask for more
    if missing_info and classification["confidence"] < 0.7:
        next_phase = "gathering"
    else:
        next_phase = "searching"

    return {
        "classification": classification,
        "missing_info": missing_info,
        "current_phase": next_phase,
    }
