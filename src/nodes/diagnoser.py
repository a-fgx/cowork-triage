"""
Diagnoser Node

Generates diagnostic hypotheses based on all gathered context:
- The bug report
- Classification
- Related GitHub issues
- Similar errors from the knowledge base (RAG)

This is where the "diagnosis" magic happens - combining multiple sources
of information to identify the most likely root cause.
"""

import json
import re
from typing import Any

from src.state import DiagnosticState, Hypothesis, ConfidenceBreakdown
from src.llm import get_llm, invoke_with_system
from src.prompts.templates import DIAGNOSIS_PROMPT
from src.rag.retriever import format_rag_context


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response that may contain markdown or extra text.

    Handles cases like:
    - Pure JSON
    - JSON wrapped in ```json ... ```
    - JSON with text before/after
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


def diagnoser_node(state: DiagnosticState) -> dict[str, Any]:
    """
    Generate diagnostic hypotheses from all gathered context.

    This node combines:
    1. The structured bug report
    2. The failure classification
    3. Related GitHub issues
    4. Similar errors from the RAG knowledge base

    It then uses the LLM to generate ranked hypotheses about the root cause.

    Args:
        state: Current graph state with all gathered information

    Returns:
        State updates with hypotheses and selected_hypothesis
    """
    # Gather all context
    bug_report = state.get("bug_report", {})
    classification = state.get("classification", {})
    related_issues = state.get("related_issues", [])
    rag_results = state.get("rag_results", [])
    library_detection = state.get("library_detection", {})
    github_confidence = state.get("github_confidence", 0.0)

    # Build comprehensive context for the LLM
    context_parts = []

    # Bug report summary
    context_parts.append("## Bug Report")
    context_parts.append(json.dumps(bug_report, indent=2))

    # Classification
    context_parts.append("\n## Classification")
    context_parts.append(
        f"Type: {classification.get('failure_type', 'unknown')}"
    )
    context_parts.append(
        f"Confidence: {classification.get('confidence', 0):.0%}"
    )
    context_parts.append(
        f"Reasoning: {classification.get('reasoning', 'N/A')}"
    )

    # GitHub issues
    if related_issues:
        context_parts.append("\n## Related GitHub Issues")
        for issue in related_issues[:3]:  # Top 3
            context_parts.append(
                f"- #{issue['number']}: {issue['title']} ({issue['state']})"
            )
            if issue.get("summary"):
                context_parts.append(f"  Summary: {issue['summary'][:150]}...")

    # RAG results
    if rag_results:
        context_parts.append("\n## Similar Known Errors")
        context_parts.append(format_rag_context(rag_results))

    full_context = "\n".join(context_parts)

    # Generate hypotheses
    llm = get_llm()

    try:
        response = invoke_with_system(
            llm,
            system_prompt=DIAGNOSIS_PROMPT,
            user_message=full_context,
        )

        result = extract_json_from_response(response)
        raw_hypotheses = result.get("hypotheses", [])

        # Convert to our Hypothesis type
        hypotheses: list[Hypothesis] = []
        for h in raw_hypotheses:
            hypotheses.append(
                {
                    "description": h.get("description", ""),
                    "likelihood": h.get("likelihood", "medium"),
                    "evidence": h.get("evidence", []),
                    "required_validations": h.get("required_validations", []),
                }
            )

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback if parsing fails
        print(f"Diagnosis parsing error: {e}")
        hypotheses = [
            {
                "description": "Unable to generate specific hypothesis - please review the error details manually",
                "likelihood": "low",
                "evidence": ["Parsing of diagnostic response failed"],
                "required_validations": ["Manual review of error message and stack trace"],
            }
        ]

    # Sort by likelihood
    likelihood_order = {"high": 0, "medium": 1, "low": 2}
    hypotheses.sort(key=lambda h: likelihood_order.get(h["likelihood"], 3))

    # Select the top hypothesis
    selected = hypotheses[0] if hypotheses else None

    # Compute confidence breakdown
    confidence_breakdown = compute_confidence_breakdown(
        classification=classification,
        github_confidence=github_confidence,
        rag_results=rag_results,
        library_detection=library_detection,
        hypotheses=hypotheses,
    )

    # Determine next phase
    # If we have a high-confidence hypothesis, proceed to resolution
    # Otherwise, end with hypotheses for manual review
    if selected and selected["likelihood"] == "high":
        next_phase = "resolution"
    else:
        next_phase = "complete"

    return {
        "hypotheses": hypotheses,
        "selected_hypothesis": selected,
        "confidence_breakdown": confidence_breakdown,
        "current_phase": next_phase,
    }


def compute_confidence_breakdown(
    classification: dict,
    github_confidence: float,
    rag_results: list,
    library_detection: dict,
    hypotheses: list,
) -> ConfidenceBreakdown:
    """
    Compute a breakdown of confidence scores from different sources.

    Args:
        classification: The bug classification with confidence
        github_confidence: Confidence from GitHub search (0.0-1.0)
        rag_results: RAG search results with similarity scores
        library_detection: Library detection with confidence
        hypotheses: Generated hypotheses

    Returns:
        ConfidenceBreakdown with scores from each source
    """
    # Classification confidence (from LLM)
    classification_conf = float(classification.get("confidence", 0.5))

    # GitHub confidence (already computed)
    github_conf = float(github_confidence)

    # RAG confidence (average similarity of top results)
    if rag_results:
        rag_conf = sum(
            r.get("similarity_score", 0) for r in rag_results[:3]
        ) / min(3, len(rag_results))
    else:
        rag_conf = 0.0

    # Library detection confidence
    lib_conf = float(library_detection.get("confidence", 0.0))

    # Compute weighted overall confidence
    # Weights: Classification (30%), GitHub (35%), RAG (20%), Library (15%)
    weights = {
        "classification": 0.30,
        "github": 0.35,
        "rag": 0.20,
        "library": 0.15,
    }

    overall = (
        classification_conf * weights["classification"]
        + github_conf * weights["github"]
        + rag_conf * weights["rag"]
        + lib_conf * weights["library"]
    )

    # Generate explanation of what contributed most
    contributions = [
        ("LLM classification", classification_conf, weights["classification"]),
        ("GitHub issue matches", github_conf, weights["github"]),
        ("RAG knowledge base", rag_conf, weights["rag"]),
        ("Library detection", lib_conf, weights["library"]),
    ]

    # Sort by contribution (score * weight)
    contributions.sort(key=lambda x: x[1] * x[2], reverse=True)

    # Build explanation
    explanation_parts = []
    for name, score, weight in contributions:
        if score > 0.5:
            explanation_parts.append(f"{name}: {score:.0%}")

    if explanation_parts:
        explanation = "Main contributors: " + ", ".join(explanation_parts[:2])
    else:
        explanation = "Low confidence across all sources"

    return {
        "classification": classification_conf,
        "github": github_conf,
        "rag": rag_conf,
        "library_detection": lib_conf,
        "overall": overall,
        "explanation": explanation,
    }
