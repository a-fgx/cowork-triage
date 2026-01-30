"""
Ticket Processing Script

Processes tickets from the incoming_tickets folder using the diagnostic agent.
All exchanges happen within the ticket file itself.

Usage:
    python scripts/process_ticket.py <ticket_number>
    python scripts/process_ticket.py <ticket_number> --clean  # Reset exchange log first

Example:
    python scripts/process_ticket.py 001
    python scripts/process_ticket.py 001 --clean
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TICKETS_DIR = project_root / "incoming_tickets"

# Phrases that indicate user wants to skip/doesn't have info
SKIP_PHRASES = [
    "skip", "i don't have", "don't have", "n/a", "not available",
    "i don't know", "don't know", "no idea", "can't provide",
    "cannot provide", "unavailable", "none", "no info"
]


def get_ticket_path(ticket_number: str) -> Path:
    """Get the path to a ticket file."""
    return TICKETS_DIR / f"ticket_{ticket_number}.md"


def read_ticket(ticket_number: str) -> str | None:
    """Read a ticket file content."""
    ticket_path = get_ticket_path(ticket_number)
    if not ticket_path.exists():
        print(f"Ticket not found: {ticket_path}")
        return None
    return ticket_path.read_text()


def append_to_exchange_log(ticket_number: str, role: str, message: str):
    """Append a message to the ticket's Exchange Log."""
    ticket_path = get_ticket_path(ticket_number)
    content = ticket_path.read_text()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### [{role}] - {timestamp}\n{message}\n"

    updated_content = content.rstrip() + entry
    ticket_path.write_text(updated_content)


def list_tickets() -> list[str]:
    """List all available tickets."""
    if not TICKETS_DIR.exists():
        return []
    return sorted([f.stem.replace("ticket_", "") for f in TICKETS_DIR.glob("ticket_*.md")])


def clean_exchange_log(ticket_number: str) -> bool:
    """
    Remove all content below '## Exchange Log' in a ticket file.

    Returns True if cleaning was performed, False if no Exchange Log section found.
    """
    ticket_path = get_ticket_path(ticket_number)
    if not ticket_path.exists():
        print(f"Ticket not found: {ticket_path}")
        return False

    content = ticket_path.read_text()

    if "## Exchange Log" not in content:
        print("No '## Exchange Log' section found in ticket.")
        return False

    # Keep everything up to and including "## Exchange Log"
    parts = content.split("## Exchange Log")
    cleaned_content = parts[0] + "## Exchange Log\n"

    ticket_path.write_text(cleaned_content)
    print(f"Cleaned exchange log for ticket {ticket_number}")
    return True


def count_skip_responses(content: str) -> int:
    """Count how many times user indicated they want to skip."""
    exchange_section = ""
    if "## Exchange Log" in content:
        exchange_section = content.split("## Exchange Log")[1].lower()

    skip_count = 0
    for phrase in SKIP_PHRASES:
        if phrase in exchange_section:
            skip_count += 1
    return skip_count


def count_agent_questions(content: str) -> int:
    """Count how many times agent asked for more info."""
    return content.count("**Waiting for more information.")


def preprocess_ticket_for_agent(content: str) -> str:
    """
    Preprocess ticket content before sending to agent.

    If user has indicated they don't have info (skip phrases) OR
    agent has already asked 2+ times, add instruction to proceed.
    """
    skip_count = count_skip_responses(content)
    question_count = count_agent_questions(content)

    if skip_count > 0 or question_count >= 2:
        # Add instruction to proceed without asking more questions
        skip_instruction = """

---
IMPORTANT INSTRUCTION: The user has indicated they don't have additional information
to provide, or the agent has already asked multiple times without getting new info.
Please proceed with the diagnosis using only the information available in this ticket.
Do NOT ask for more information. Make your best assessment with what you have.
---

"""
        # Insert before Exchange Log
        if "## Exchange Log" in content:
            parts = content.split("## Exchange Log")
            return parts[0] + skip_instruction + "## Exchange Log" + parts[1]
        else:
            return content + skip_instruction

    return content


def format_classification(classification: dict) -> str:
    """Format classification info for display."""
    if not classification:
        return ""

    failure_type = classification.get("failure_type", "unknown")
    confidence = classification.get("confidence", 0)
    reasoning = classification.get("reasoning", "")

    return f"""**Classification:**
- Type: `{failure_type}`
- Confidence: {confidence:.0%}
- Reasoning: {reasoning}
"""


def process_ticket(ticket_number: str):
    """Process a ticket using the diagnostic agent."""
    content = read_ticket(ticket_number)
    if not content:
        return

    print(f"Processing ticket {ticket_number}...")
    print("-" * 40)

    # Import dependencies
    from src.config import config
    config.setup_langsmith()

    from src.agent import build_diagnostic_agent

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Preprocess ticket - handle skip responses and repeated questions
    processed_content = preprocess_ticket_for_agent(content)

    # Build the agent
    agent = build_diagnostic_agent()

    thread_id = str(uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [{"role": "user", "content": processed_content}],
    }

    collected_responses = []
    current_phase = ""
    last_classification = None

    try:
        for event in agent.stream(initial_state, thread_config, stream_mode="values"):
            phase = event.get("current_phase", current_phase)
            if phase and phase != current_phase:
                current_phase = phase
                print(f"\n[Phase: {current_phase.upper()}]")

            # Capture classification when available
            classification = event.get("classification")
            if classification and classification != last_classification:
                last_classification = classification
                class_info = format_classification(classification)
                print(f"\n{class_info}")
                collected_responses.append(class_info)

            messages = event.get("messages", [])
            for msg in messages:
                role = msg.get("role", "unknown")
                msg_content = msg.get("content", "")

                if role == "assistant" and msg_content:
                    print(f"\n{msg_content}")
                    collected_responses.append(msg_content)

            # Handle interrupts - agent asking for more info
            if event.get("needs_user_input"):
                print("\n[Agent needs more information - update the ticket and run again]")
                if collected_responses:
                    append_to_exchange_log(ticket_number, "Agent", "\n\n".join(collected_responses))
                append_to_exchange_log(ticket_number, "Agent", "**Waiting for more information. Please add your response below and run the script again. (Type 'skip' if you don't have this info)**")
                return

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    # Write all collected responses to the ticket
    if collected_responses:
        append_to_exchange_log(ticket_number, "Agent", "\n\n".join(collected_responses))

    print("\n" + "-" * 40)
    print(f"Ticket {ticket_number} updated.")


def main():
    parser = argparse.ArgumentParser(description="Process a ticket from incoming_tickets")
    parser.add_argument("ticket_number", nargs="?", help="Ticket number to process (e.g., 001)")
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean/reset the exchange log before processing (removes previous agent responses)"
    )
    args = parser.parse_args()

    if not args.ticket_number:
        print("Available tickets:")
        tickets = list_tickets()
        if tickets:
            for t in tickets:
                print(f"  - {t}")
        else:
            print("  No tickets found in incoming_tickets/")
        print("\nUsage: python scripts/process_ticket.py <ticket_number> [--clean]")
        sys.exit(0)

    # Clean exchange log if requested
    if args.clean:
        clean_exchange_log(args.ticket_number)

    process_ticket(args.ticket_number)


if __name__ == "__main__":
    main()
