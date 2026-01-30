"""
LangGraph Agent Assembly

This is where all the nodes come together into a complete workflow.
The StateGraph defines:
- Nodes: The processing steps
- Edges: The connections between steps
- Conditional edges: Dynamic routing based on state

Graph Structure:
    START → intake → classifier → [github_search, rag_search] (parallel) → diagnoser
                                                                    ↓
                                                        [check_confidence?]
                                                                    ↓
                                            high → resolution → END
                                            medium/low → [asked < 2 times?]
                                                              ↓
                                            yes → info_gatherer → user_input → diagnoser
                                            no → END (return hypotheses)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import DiagnosticState
from src.nodes.intake import intake_node
from src.nodes.classifier import classifier_node
from src.nodes.info_gatherer import info_gatherer_node, user_input_node
from src.nodes.diagnoser import diagnoser_node
from src.nodes.resolution import resolution_node
from src.tools.github_tools import github_search_node
from src.rag.retriever import rag_search_node


# =============================================================================
# Routing Functions
# =============================================================================


def route_after_diagnosis(state: DiagnosticState) -> str:
    """
    Decide what to do after diagnosis.

    - High confidence → go to resolution
    - Medium/Low confidence + haven't asked user yet → gather info
    - Medium/Low confidence + already asked → end with hypotheses
    """
    hypotheses = state.get("hypotheses", [])
    attempts = state.get("info_gathering_attempts", 0)

    if hypotheses and hypotheses[0].get("likelihood") == "high":
        return "resolution"
    elif attempts < 2:
        # Try to gather more info to improve confidence
        return "gathering"
    else:
        # Give up and return what we have
        return "end"


def route_after_user_input(state: DiagnosticState) -> str:
    """
    After user provides more information, go back to diagnosis.

    We skip re-searching since we already have search results,
    just re-diagnose with the new context.
    """
    return "diagnoser"


# =============================================================================
# Graph Builder
# =============================================================================


def build_diagnostic_graph() -> StateGraph:
    """
    Build the diagnostic agent graph (uncompiled).

    Returns the StateGraph before compilation. Useful for visualization.

    Returns:
        Uncompiled StateGraph
    """
    # Create the graph with our state schema
    graph = StateGraph(DiagnosticState)

    # === Add all nodes ===
    graph.add_node("intake", intake_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("github_search", github_search_node)
    graph.add_node("rag_search", rag_search_node)
    graph.add_node("diagnoser", diagnoser_node)
    graph.add_node("info_gatherer", info_gatherer_node)
    graph.add_node("user_input", user_input_node)
    graph.add_node("resolution", resolution_node)

    # === Define edges ===

    # Start with intake
    graph.add_edge(START, "intake")

    # Intake → Classifier
    graph.add_edge("intake", "classifier")

    # Fan out: classifier → github_search AND rag_search (parallel)
    graph.add_edge("classifier", "github_search")
    graph.add_edge("classifier", "rag_search")

    # Fan in: both searches → diagnoser (waits for both to complete)
    graph.add_edge("github_search", "diagnoser")
    graph.add_edge("rag_search", "diagnoser")

    # After diagnosis: resolution, gather info, or end
    graph.add_conditional_edges(
        "diagnoser",
        route_after_diagnosis,
        {
            "resolution": "resolution",
            "gathering": "info_gatherer",
            "end": END,
        },
    )

    # Info gathering leads to user input
    graph.add_edge("info_gatherer", "user_input")

    # After user input: back to diagnoser (with new context)
    graph.add_conditional_edges(
        "user_input",
        route_after_user_input,
        {
            "diagnoser": "diagnoser",
        },
    )

    # Resolution always ends
    graph.add_edge("resolution", END)

    return graph


def build_diagnostic_agent(checkpointer=None):
    """
    Build and compile the diagnostic agent.

    The compiled graph can be invoked to process bug reports.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     If None, uses MemorySaver for in-memory persistence.

    Returns:
        Compiled graph ready for invocation

    Example:
        agent = build_diagnostic_agent()

        # Start a new conversation
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "I get an error..."}]},
            config={"configurable": {"thread_id": "thread-1"}}
        )

        # Stream the results
        for event in agent.stream(...):
            print(event)
    """
    graph = build_diagnostic_graph()

    # Use memory checkpointer if none provided
    # This is needed for interrupt/resume functionality
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile with checkpointer
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Visualization Helper
# =============================================================================


def get_graph_image(output_path: str = "graph.png"):
    """
    Generate a visual representation of the graph.

    Requires graphviz to be installed: brew install graphviz

    Args:
        output_path: Where to save the image

    Example:
        get_graph_image("diagnostic_agent.png")
    """
    try:
        graph = build_diagnostic_graph()
        img = graph.compile().get_graph().draw_mermaid_png()

        with open(output_path, "wb") as f:
            f.write(img)

        print(f"Graph saved to: {output_path}")

    except Exception as e:
        print(f"Could not generate graph image: {e}")
        print("Make sure graphviz is installed: brew install graphviz")


# =============================================================================
# Pre-built agent instance
# =============================================================================

# Create a default agent instance
# Import and use: from src.agent import agent
agent = build_diagnostic_agent()

# For LangGraph API/Studio - compiled without checkpointer (platform handles persistence)
graph = build_diagnostic_graph().compile()
