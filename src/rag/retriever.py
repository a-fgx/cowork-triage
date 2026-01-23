"""
RAG Retriever

Retrieves similar error messages and solutions from the knowledge base.
This is the main interface for RAG in the diagnostic workflow.

Key Concept: RAG (Retrieval-Augmented Generation)
    Instead of relying solely on the LLM's training data, we:
    1. Retrieve relevant documents from our knowledge base
    2. Include them in the LLM's context
    3. Let the LLM use this fresh, specific information

    This gives us:
    - Up-to-date information (not limited to training cutoff)
    - Domain-specific knowledge (our curated error database)
    - Verifiable sources (we know where the answer came from)
"""

from typing import Any

from src.state import DiagnosticState, RAGResult
from src.rag.vectorstore import similarity_search_with_score, get_default_vectorstore


def search_error_solutions(
    query: str,
    k: int = 5,
    score_threshold: float = 1.0,
) -> list[RAGResult]:
    """
    Search for similar error messages and their solutions.

    Args:
        query: The error message or bug description to search for
        k: Maximum number of results
        score_threshold: Maximum distance score to include (lower = more similar)

    Returns:
        List of RAGResult objects with error patterns and solutions

    Example:
        results = search_error_solutions("TypeError: object is not callable")
        for r in results:
            print(f"Similar error: {r['error_pattern']}")
            print(f"Solution: {r['solution']}")
    """
    # Search the vector store
    results = similarity_search_with_score(query, k=k)

    rag_results: list[RAGResult] = []

    for doc, score in results:
        # Skip results that are too dissimilar
        if score > score_threshold:
            continue

        # Convert distance to similarity (0-1 range, higher = more similar)
        # ChromaDB uses L2 distance, typical range 0-2
        similarity = max(0, 1 - (score / 2))

        rag_results.append(
            {
                "error_pattern": doc.page_content,
                "solution": doc.metadata.get("solution", "No solution provided"),
                "source": doc.metadata.get("source", "unknown"),
                "similarity_score": round(similarity, 3),
            }
        )

    return rag_results


def rag_search_node(state: DiagnosticState) -> dict[str, Any]:
    """
    LangGraph node that searches the knowledge base for similar errors.

    Uses the bug report's error message (if present) or raw description
    to find similar known issues.

    Args:
        state: Current graph state

    Returns:
        State updates with rag_results
    """
    bug_report = state.get("bug_report", {})

    # Build search query
    # Prioritize: error_message > title > raw_description
    query = ""

    if bug_report.get("error_message"):
        query = bug_report["error_message"]
    elif bug_report.get("title"):
        query = bug_report["title"]
    elif bug_report.get("raw_description"):
        query = bug_report["raw_description"][:500]

    if not query:
        # No query to search
        return {
            "rag_results": [],
            "current_phase": "diagnosis",
        }

    # Search the knowledge base
    try:
        results = search_error_solutions(query, k=5)
    except Exception as e:
        # If vector store isn't initialized, return empty results
        print(f"RAG search error: {e}")
        results = []

    return {
        "rag_results": results,
        "current_phase": "diagnosis",
    }


def format_rag_context(results: list[RAGResult]) -> str:
    """
    Format RAG results for inclusion in LLM prompts.

    Creates a readable summary of similar errors and solutions
    that can be added to the diagnosis prompt.

    Args:
        results: List of RAGResult objects

    Returns:
        Formatted string for LLM context
    """
    if not results:
        return "No similar errors found in the knowledge base."

    lines = ["Similar errors from our knowledge base:"]

    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. Error Pattern (similarity: {r['similarity_score']:.0%})")
        lines.append(f"   {r['error_pattern'][:200]}")
        lines.append(f"   Solution: {r['solution'][:200]}")

    return "\n".join(lines)
