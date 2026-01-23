"""
Resolution Node

Creates a step-by-step resolution plan based on the diagnosis.

This is the final node in the diagnostic workflow. It takes the
selected hypothesis and generates actionable steps to fix the issue.
"""

import json
from typing import Any

from src.state import DiagnosticState, ResolutionStep
from src.llm import get_llm, invoke_with_system
from src.prompts.templates import RESOLUTION_PROMPT


def resolution_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Generate a step-by-step resolution plan.

    Uses the selected hypothesis to create actionable steps
    that the user can follow to fix their issue.

    Args:
        state: Current graph state with diagnosis

    Returns:
        State updates with resolution_plan and final message
    """
    hypothesis = state.get("selected_hypothesis")
    bug_report = state.get("bug_report", {})
    related_issues = state.get("related_issues", [])
    rag_results = state.get("rag_results", [])

    if not hypothesis:
        # No hypothesis to resolve - shouldn't happen, but handle gracefully
        return {
            "resolution_plan": [],
            "current_phase": "complete",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Unable to generate a resolution plan. Please review the hypotheses manually.",
                }
            ],
        }

    # Build context for resolution
    context_parts = []

    context_parts.append("## Selected Diagnosis")
    context_parts.append(f"Root Cause: {hypothesis['description']}")
    context_parts.append(f"Confidence: {hypothesis['likelihood']}")
    context_parts.append(f"Evidence: {json.dumps(hypothesis['evidence'])}")

    context_parts.append("\n## Original Bug Report")
    context_parts.append(json.dumps(bug_report, indent=2))

    # Include relevant solutions from RAG
    if rag_results:
        context_parts.append("\n## Known Solutions for Similar Errors")
        for r in rag_results[:2]:
            context_parts.append(f"- {r['solution']}")

    # Include any solutions from GitHub issues
    relevant_issues = [
        issue for issue in related_issues if issue.get("state") == "closed"
    ]
    if relevant_issues:
        context_parts.append("\n## Solutions from Related GitHub Issues")
        for issue in relevant_issues[:2]:
            context_parts.append(
                f"- Issue #{issue['number']}: See {issue['url']}"
            )

    full_context = "\n".join(context_parts)

    # Generate resolution plan
    llm = get_llm()

    try:
        response = invoke_with_system(
            llm,
            system_prompt=RESOLUTION_PROMPT,
            user_message=full_context,
        )

        result = json.loads(response)
        raw_steps = result.get("steps", [])

        # Convert to our ResolutionStep type
        resolution_plan: list[ResolutionStep] = []
        for step in raw_steps:
            resolution_plan.append(
                {
                    "order": step.get("order", len(resolution_plan) + 1),
                    "action": step.get("action", ""),
                    "rationale": step.get("rationale", ""),
                    "expected_outcome": step.get("expected_outcome", ""),
                }
            )

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Resolution parsing error: {e}")
        # Fallback to basic steps
        resolution_plan = [
            {
                "order": 1,
                "action": "Review the error message and stack trace carefully",
                "rationale": "Understanding the exact error is the first step",
                "expected_outcome": "Identify the specific line or function causing the issue",
            },
            {
                "order": 2,
                "action": "Search for the error message online",
                "rationale": "Others may have encountered and solved this issue",
                "expected_outcome": "Find relevant Stack Overflow posts or documentation",
            },
        ]

    # Format the final response
    library_detection = state.get("library_detection")
    confidence_breakdown = state.get("confidence_breakdown")
    summary = format_resolution_summary(
        hypothesis, resolution_plan, related_issues, library_detection, confidence_breakdown
    )

    return {
        "resolution_plan": resolution_plan,
        "current_phase": "complete",
        "messages": [{"role": "assistant", "content": summary}],
    }


def format_resolution_summary(
    hypothesis: dict,
    steps: list[ResolutionStep],
    related_issues: list[dict],
    library_detection: dict | None = None,
    confidence_breakdown: dict | None = None,
) -> str:
    """
    Format the diagnosis and resolution plan as a readable summary.

    Creates a markdown-formatted message for the user.
    """
    lines = []

    # Header
    lines.append("# Diagnostic Report")
    lines.append("")

    # Library detection info
    if library_detection and library_detection.get("primary") != "unknown":
        lines.append("## Detected Libraries")
        primary = library_detection.get("primary", "unknown")
        lines.append(f"**Primary:** {primary}")
        all_libs = library_detection.get("all_libraries", [])
        if len(all_libs) > 1:
            lines.append(f"**Also involved:** {', '.join([l for l in all_libs if l != primary])}")
        components = library_detection.get("components", [])
        if components:
            lines.append(f"**Components:** {', '.join(components[:5])}")
        lines.append("")

    # Similar GitHub Issues (show first for context)
    if related_issues:
        lines.append("## Similar GitHub Issues Found")
        for issue in related_issues[:5]:
            status = "✅ closed" if issue.get("state") == "closed" else "🔴 open"
            repo = issue.get("repo", "")
            repo_label = ""
            if repo:
                # Extract just the repo name (e.g., "langgraph" from "langchain-ai/langgraph")
                repo_name = repo.split("/")[-1] if "/" in repo else repo
                repo_label = f" [{repo_name}]"
            lines.append(f"- **[#{issue['number']}: {issue['title']}]({issue['url']})**{repo_label} ({status})")
            if issue.get("summary"):
                summary = issue["summary"][:150].replace("\n", " ")
                lines.append(f"  > {summary}...")
        lines.append("")

    # Diagnosis
    lines.append("## Diagnosis")
    lines.append(f"**Root Cause:** {hypothesis.get('description', 'Unknown')}")
    lines.append(f"**Confidence:** {hypothesis.get('likelihood', 'Unknown').capitalize()}")
    lines.append("")

    # Confidence breakdown
    if confidence_breakdown:
        lines.append("### Confidence Sources")
        lines.append(f"| Source | Score |")
        lines.append(f"|--------|-------|")
        lines.append(f"| LLM Classification | {confidence_breakdown.get('classification', 0):.0%} |")
        lines.append(f"| GitHub Issues | {confidence_breakdown.get('github', 0):.0%} |")
        lines.append(f"| RAG Knowledge Base | {confidence_breakdown.get('rag', 0):.0%} |")
        lines.append(f"| Library Detection | {confidence_breakdown.get('library_detection', 0):.0%} |")
        lines.append(f"| **Overall** | **{confidence_breakdown.get('overall', 0):.0%}** |")
        lines.append("")
        explanation = confidence_breakdown.get("explanation", "")
        if explanation:
            lines.append(f"*{explanation}*")
            lines.append("")

    # Evidence
    evidence = hypothesis.get("evidence", [])
    if evidence:
        lines.append("**Supporting Evidence:**")
        for e in evidence:
            lines.append(f"- {e}")
        lines.append("")

    # Resolution Plan
    lines.append("## Resolution Plan")
    lines.append("")

    for step in steps:
        lines.append(f"### Step {step['order']}: {step['action']}")
        lines.append(f"*Why:* {step['rationale']}")
        lines.append(f"*Expected result:* {step['expected_outcome']}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("*If the issue persists after following these steps, please provide additional details.*")

    return "\n".join(lines)
