"""
State Schema for the Diagnostic Agent

This module defines the data structures that flow through the LangGraph workflow.
The state is like a shared memory that each node can read and update.

Key Concept: TypedDict with Reducers
- TypedDict: Python type hint for dictionaries with specific keys
- Annotated[list, operator.add]: A "reducer" that appends to lists instead of replacing

Example:
    # Without reducer - value is replaced
    state["messages"] = ["new"]  # Old messages lost!

    # With reducer - value is appended
    state["messages"] = ["new"]  # Becomes ["old", "new"]
"""

from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict
import operator


# =============================================================================
# Sub-Structures (building blocks for the main state)
# =============================================================================


class BugReport(TypedDict):
    """
    Structured representation of a user's bug report.

    Extracted from the raw user message by the intake node.
    """

    # The original message from the user (always preserved)
    raw_description: str

    # Extracted fields (may be empty if not provided by user)
    title: NotRequired[str]  # Brief summary, e.g., "API timeout error"
    steps_to_reproduce: NotRequired[list[str]]  # ["1. Open app", "2. Click X"]
    expected_behavior: NotRequired[str]  # What should happen
    actual_behavior: NotRequired[str]  # What actually happens
    environment: NotRequired[dict[str, str]]  # {"python": "3.11", "os": "macOS"}
    error_message: NotRequired[str]  # The actual error text
    stack_trace: NotRequired[str]  # Full stack trace if provided


class Classification(TypedDict):
    """
    Classification of the bug type.

    Determined by the classifier node based on the bug report.
    """

    # The category of failure
    failure_type: Literal[
        "api",  # API call issues (auth, rate limits, responses)
        "version",  # Version incompatibility or deprecation
        "dependency",  # Missing or conflicting packages
        "runtime",  # Execution errors, crashes, exceptions
        "configuration",  # Setup or environment issues
        "unknown",  # Cannot determine from available info
    ]

    # How confident we are in this classification (0.0 to 1.0)
    confidence: float

    # Explanation of why this classification was chosen
    reasoning: str


class LibraryDetection(TypedDict):
    """
    Detection of which LangChain ecosystem libraries are involved.

    This helps target GitHub searches and provide more specific diagnoses.
    """

    # Primary library involved (most likely source of the issue)
    primary: Literal[
        "langchain",      # langchain-core, langchain, langchain-community
        "langgraph",      # LangGraph state machines and agents
        "langsmith",      # LangSmith tracing and observability
        "langserve",      # LangServe deployment
        "unknown",        # Cannot determine
    ]

    # All libraries mentioned or involved in the issue
    all_libraries: list[str]

    # Specific components/modules mentioned (e.g., "tool_node", "ChatOpenAI", "MemorySaver")
    components: list[str]

    # Confidence in the detection (0.0 to 1.0)
    confidence: float


class GitHubIssue(TypedDict):
    """
    A related GitHub issue found during diagnosis.
    """

    number: int  # Issue #123
    title: str
    url: str  # Full URL to the issue
    state: str  # "open" or "closed"
    relevance_score: float  # How relevant to our bug (0.0 to 1.0)
    summary: str  # Brief summary of the issue
    repo: NotRequired[str]  # Repository (e.g., "langchain-ai/langgraph")


class RAGResult(TypedDict):
    """
    A result from the error messages knowledge base (RAG).
    """

    error_pattern: str  # The error message pattern
    solution: str  # Known solution for this error
    source: str  # Where this came from (e.g., "Kaggle dataset")
    similarity_score: float  # How similar to our error (0.0 to 1.0)


class ConfidenceBreakdown(TypedDict):
    """
    Breakdown of confidence scores from different sources.

    This helps users understand where the diagnosis confidence comes from.
    """

    # Classification confidence from LLM analysis (0.0 to 1.0)
    classification: float

    # GitHub search confidence - based on match quality (0.0 to 1.0)
    # High if exact title match found, lower for partial matches
    github: float

    # RAG confidence - based on similarity scores (0.0 to 1.0)
    rag: float

    # Library detection confidence (0.0 to 1.0)
    library_detection: float

    # Combined/overall confidence (weighted average)
    overall: float

    # Human-readable explanation of what contributed most
    explanation: str


class Hypothesis(TypedDict):
    """
    A diagnostic hypothesis about the root cause.
    """

    # Description of the suspected root cause
    description: str

    # How likely this hypothesis is
    likelihood: Literal["high", "medium", "low"]

    # Evidence supporting this hypothesis
    evidence: list[str]

    # Steps to validate this hypothesis
    required_validations: list[str]


class ResolutionStep(TypedDict):
    """
    A single step in the resolution plan.
    """

    order: int  # Step number (1, 2, 3...)
    action: str  # What to do
    rationale: str  # Why this helps
    expected_outcome: str  # What should change after this step


# =============================================================================
# Main Agent State
# =============================================================================


class DiagnosticState(TypedDict):
    """
    The main state that flows through the entire LangGraph workflow.

    Each node receives this state and returns updates to it.
    The graph engine merges the updates using reducers.

    Example node:
        def my_node(state: DiagnosticState) -> dict:
            # Read from state
            bug = state.get("bug_report", {})

            # Return updates (not the full state!)
            return {
                "classification": {"failure_type": "api", ...},
                "current_phase": "diagnosis"
            }
    """

    # === Conversation History ===
    # Uses operator.add reducer: new messages are appended, not replaced
    # This preserves the full conversation history
    messages: Annotated[list[dict], operator.add]

    # === Bug Report Data ===
    # Structured info extracted from the user's description
    bug_report: BugReport

    # === Classification ===
    # What type of problem this is
    classification: NotRequired[Classification]

    # Which LangChain ecosystem libraries are involved
    library_detection: NotRequired[LibraryDetection]

    # === Information Gathering ===
    # What info is still needed from the user
    missing_info: list[str]

    # How many times we've asked for more info (to prevent infinite loops)
    info_gathering_attempts: int

    # === External Data ===
    # Related GitHub issues found
    related_issues: list[GitHubIssue]

    # Similar error messages from the knowledge base
    rag_results: list[RAGResult]

    # === Diagnosis ===
    # Possible root causes, ranked by likelihood
    hypotheses: list[Hypothesis]

    # The most likely hypothesis (first in the list)
    selected_hypothesis: NotRequired[Hypothesis]

    # Breakdown of confidence scores from different sources
    confidence_breakdown: NotRequired[ConfidenceBreakdown]

    # === Resolution ===
    # Step-by-step plan to fix the issue
    resolution_plan: list[ResolutionStep]

    # === Workflow Control ===
    # Current phase of the workflow
    current_phase: Literal[
        "intake",  # Parsing the initial bug report
        "classification",  # Determining the type of bug
        "gathering",  # Collecting missing information
        "searching",  # Searching GitHub and knowledge base
        "diagnosis",  # Generating hypotheses
        "resolution",  # Creating the fix plan
        "complete",  # Done!
    ]

    # Whether we're waiting for user input
    needs_user_input: bool

    # The question to ask the user (if needs_user_input is True)
    user_question: NotRequired[str]
