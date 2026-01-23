"""
Info Gatherer Node

Handles human-in-the-loop interaction to collect missing information.

This module contains two nodes:
1. info_gatherer_node: Generates a question based on missing info
2. process_user_response_node: Processes the user's answer

Key Concept: LangGraph Interrupt
    LangGraph's interrupt() function pauses the workflow and returns control
    to the caller. The workflow can be resumed later with the user's response.

    This is how we implement "ask and wait" behavior in a stateless workflow.
"""

import json
from typing import Any

from langgraph.types import interrupt

from src.state import DiagnosticState
from src.llm import get_llm, invoke_with_system
from src.prompts.templates import INFO_GATHERING_PROMPT


def info_gatherer_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Generate a question to collect missing information.

    Uses the LLM to craft a targeted question based on what's missing
    from the bug report.

    Args:
        state: Current graph state

    Returns:
        State updates with the question and needs_user_input flag
    """
    missing_info = state.get("missing_info", [])
    bug_report = state.get("bug_report", {})
    attempts = state.get("info_gathering_attempts", 0)

    # Check if we've asked too many times
    if attempts >= 3:
        # Give up and proceed with what we have
        return {
            "current_phase": "searching",
            "needs_user_input": False,
            "missing_info": [],  # Clear so we don't loop
        }

    # Build context for the LLM
    context = f"""
Current bug report:
{json.dumps(bug_report, indent=2)}

Missing information:
{json.dumps(missing_info, indent=2)}

Number of questions already asked: {attempts}
"""

    # Generate a question
    llm = get_llm()
    question = invoke_with_system(
        llm,
        system_prompt=INFO_GATHERING_PROMPT,
        user_message=context,
    )

    # Add the question to messages so it appears in the conversation
    return {
        "messages": [{"role": "assistant", "content": question}],
        "needs_user_input": True,
        "user_question": question,
        "info_gathering_attempts": attempts + 1,
    }


def user_input_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Pause execution to wait for user input.

    This node uses LangGraph's interrupt() to pause the workflow.
    The caller must resume the workflow with the user's response.

    How interrupt works:
    1. interrupt() is called with context (the question)
    2. The graph execution pauses and returns to the caller
    3. The caller gets the user's response somehow (CLI input, web form, etc.)
    4. The caller resumes the graph with graph.invoke(Command(resume=response))
    5. This node receives the response and updates the state

    Args:
        state: Current graph state

    Returns:
        State updates with the user's response incorporated
    """
    question = state.get("user_question", "Could you provide more details?")

    # INTERRUPT: Pause execution and wait for user input
    # The returned value will be whatever is passed to Command(resume=...)
    user_response = interrupt(
        {
            "question": question,
            "type": "user_input_needed",
        }
    )

    # When we resume, user_response contains the user's answer
    bug_report = state.get("bug_report", {})

    # Add the user's response to the bug report
    # Store it in a dedicated field for additional context
    additional_context = bug_report.get("additional_context", "")
    if additional_context:
        additional_context += "\n\n"
    additional_context += f"User clarification: {user_response}"

    updated_report = {
        **bug_report,
        "additional_context": additional_context,
    }

    # Re-classify with the new information
    return {
        "bug_report": updated_report,
        "messages": [{"role": "user", "content": user_response}],
        "needs_user_input": False,
        "current_phase": "classification",  # Re-classify with new info
    }
