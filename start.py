"""
Cowork-Triage: LangGraph Diagnostic Agent

Main entry point for running the diagnostic agent.

Usage:
    python start.py

    # Or with a specific bug report:
    python start.py "RuntimeError: Event loop is already running"

Prerequisites:
    1. Set up .env file with GOOGLE_API_KEY
    2. (Optional) Load dataset: python scripts/load_dataset.py
    3. (Optional) Set LANGSMITH_API_KEY for tracing
"""

import sys
from uuid import uuid4

# IMPORTANT: Set up LangSmith BEFORE importing LangChain/LangGraph
from src.config import config
config.setup_langsmith()

from langgraph.types import Command
from src.agent import build_diagnostic_agent


def print_banner():
    """Print the welcome banner."""
    print("=" * 60)
    print("  COWORK-TRIAGE: Bug Diagnostic Agent")
    print("  Powered by LangGraph + Gemini + RAG")
    print("=" * 60)
    print()


def print_event(event: dict, phase: str = ""):
    """Print a graph event in a readable format."""
    # Get the current phase
    current_phase = event.get("current_phase", phase)

    # Print phase changes
    if current_phase and current_phase != phase:
        print(f"\n[Phase: {current_phase.upper()}]")

    # Print any new messages
    messages = event.get("messages", [])
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "assistant":
            print(f"\n{content}")
        elif role == "user":
            print(f"\n[User]: {content[:100]}...")

    return current_phase


def run_agent(initial_message: str):
    """
    Run the diagnostic agent with an initial bug report.

    This demonstrates the full workflow:
    1. Initialize the agent
    2. Send the bug report
    3. Handle interrupts (if the agent asks questions)
    4. Display the results

    Args:
        initial_message: The user's bug report
    """
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease create a .env file with your API keys.")
        print("See .env.example for the required variables.")
        return

    # Build the agent
    print("Initializing agent...")
    agent = build_diagnostic_agent()

    # Create a unique thread ID for this conversation
    thread_id = str(uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Initial state with the user's message
    initial_state = {
        "messages": [{"role": "user", "content": initial_message}],
    }

    print(f"\n[Thread: {thread_id[:8]}...]")
    print(f"\n[User]: {initial_message[:200]}...")

    # Run the agent with streaming
    current_phase = ""

    try:
        for event in agent.stream(initial_state, thread_config, stream_mode="values"):
            current_phase = print_event(event, current_phase)

            # Check if we need user input (interrupt)
            if event.get("needs_user_input"):
                # In a real application, you'd get input from the user
                # For this demo, we'll simulate a response
                print("\n[Interrupt: Agent is asking for more information]")

                # Get user input
                user_response = input("\nYour response (or 'skip' to proceed): ").strip()

                if user_response.lower() == "skip":
                    user_response = "I don't have that information right now."

                # Resume the graph with the user's response
                resume_command = Command(resume=user_response)

                for event in agent.stream(resume_command, thread_config, stream_mode="values"):
                    current_phase = print_event(event, current_phase)

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    print("\n" + "=" * 60)
    print("Diagnosis complete!")


def interactive_mode():
    """
    Run the agent in interactive mode.

    Allows the user to enter multiple bug reports in a session.
    """
    print_banner()
    print("Enter your bug report below, or type 'quit' to exit.")
    print("Tip: For best results, include the error message and context.")
    print()

    while True:
        try:
            # Get multi-line input
            print("-" * 40)
            print("Describe your bug (press Enter twice to submit):")

            lines = []
            while True:
                line = input()
                if line == "":
                    if lines:
                        break
                    continue
                if line.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    return
                lines.append(line)

            bug_report = "\n".join(lines)

            if bug_report.strip():
                run_agent(bug_report)
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            break


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Bug report passed as command line argument
        bug_report = " ".join(sys.argv[1:])
        print_banner()
        run_agent(bug_report)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
