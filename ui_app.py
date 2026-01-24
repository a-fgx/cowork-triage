"""
Streamlit UI for Cowork Triage System

Web-based interface for viewing and processing incoming tickets.
"""

import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import streamlit as st
import re

# Add project root to path
project_root = Path(__file__).parent
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
        return None
    return ticket_path.read_text()


def parse_ticket_metadata(content: str) -> dict:
    """Parse ticket metadata from content."""
    metadata = {
        "title": "Unknown",
        "status": "Unknown",
        "description": "",
    }

    # Extract title
    title_match = re.search(r"## Title\n(.+)", content)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    # Extract status
    status_match = re.search(r"## Status: (.+)", content)
    if status_match:
        metadata["status"] = status_match.group(1).strip()

    # Extract description
    desc_match = re.search(r"## Description\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
    if desc_match:
        metadata["description"] = desc_match.group(1).strip()

    return metadata


def list_tickets() -> list[dict]:
    """List all available tickets with their metadata."""
    if not TICKETS_DIR.exists():
        return []

    tickets = []
    for ticket_file in sorted(TICKETS_DIR.glob("ticket_*.md")):
        ticket_number = ticket_file.stem.replace("ticket_", "")
        content = ticket_file.read_text()
        metadata = parse_ticket_metadata(content)
        tickets.append({
            "number": ticket_number,
            **metadata
        })

    return tickets


def append_to_exchange_log(ticket_number: str, role: str, message: str):
    """Append a message to the ticket's Exchange Log."""
    ticket_path = get_ticket_path(ticket_number)
    content = ticket_path.read_text()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### [{role}] - {timestamp}\n{message}\n"

    updated_content = content.rstrip() + entry
    ticket_path.write_text(updated_content)


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
        skip_instruction = """

---
IMPORTANT INSTRUCTION: The user has indicated they don't have additional information
to provide, or the agent has already asked multiple times without getting new info.
Please proceed with the diagnosis using only the information available in this ticket.
Do NOT ask for more information. Make your best assessment with what you have.
---

"""
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


def process_ticket_stream(ticket_number: str):
    """Process a ticket using the diagnostic agent with streaming updates."""
    content = read_ticket(ticket_number)
    if not content:
        st.error(f"Ticket {ticket_number} not found")
        return

    # Import dependencies
    try:
        from src.config import config
        config.setup_langsmith()

        from src.agent import build_diagnostic_agent

        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.info("Make sure you have set up your .env file with required API keys (GOOGLE_API_KEY)")
            return
    except ImportError as e:
        st.error(f"Import error: {e}")
        st.info("Make sure all dependencies are installed: pip install -e .")
        return

    # Preprocess ticket
    processed_content = preprocess_ticket_for_agent(content)

    # Build the agent
    with st.spinner("Building diagnostic agent..."):
        agent = build_diagnostic_agent()

    thread_id = str(uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [{"role": "user", "content": processed_content}],
    }

    collected_responses = []
    current_phase = ""
    last_classification = None

    # Create containers for streaming output
    phase_container = st.empty()
    output_container = st.empty()

    output_text = ""

    try:
        for event in agent.stream(initial_state, thread_config, stream_mode="values"):
            phase = event.get("current_phase", current_phase)
            if phase and phase != current_phase:
                current_phase = phase
                phase_container.info(f"**Phase:** {current_phase.upper()}")

            # Capture classification when available
            classification = event.get("classification")
            if classification and classification != last_classification:
                last_classification = classification
                class_info = format_classification(classification)
                output_text += f"\n\n{class_info}"
                output_container.markdown(output_text)
                collected_responses.append(class_info)

            messages = event.get("messages", [])
            for msg in messages:
                role = msg.get("role", "unknown")
                msg_content = msg.get("content", "")

                if role == "assistant" and msg_content:
                    output_text += f"\n\n{msg_content}"
                    output_container.markdown(output_text)
                    collected_responses.append(msg_content)

            # Handle interrupts - agent asking for more info
            if event.get("needs_user_input"):
                st.warning("Agent needs more information - update the ticket in the response section below")
                if collected_responses:
                    append_to_exchange_log(ticket_number, "Agent", "\n\n".join(collected_responses))
                append_to_exchange_log(
                    ticket_number,
                    "Agent",
                    "**Waiting for more information. Please add your response below and process again. (Type 'skip' if you don't have this info)**"
                )
                st.rerun()
                return

    except KeyboardInterrupt:
        st.warning("Processing interrupted by user")
        return
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return

    # Write all collected responses to the ticket
    if collected_responses:
        append_to_exchange_log(ticket_number, "Agent", "\n\n".join(collected_responses))
        st.success(f"Ticket {ticket_number} updated successfully!")
        st.rerun()


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Cowork Triage System",
        page_icon="🎫",
        layout="wide"
    )

    st.title("🎫 Cowork Triage System")
    st.markdown("### Automated Bug Diagnosis and Triage")

    # Sidebar for ticket selection
    with st.sidebar:
        st.header("Incoming Tickets")

        tickets = list_tickets()

        if not tickets:
            st.warning("No tickets found in incoming_tickets/")
            st.info("Add ticket files to the incoming_tickets/ directory")
            return

        # Display tickets as selectable items
        st.markdown(f"**Total Tickets:** {len(tickets)}")

        selected_ticket = None
        for ticket in tickets:
            status_color = "🟢" if ticket["status"].lower() == "closed" else "🔴"
            button_label = f"{status_color} Ticket {ticket['number']}"

            if st.button(button_label, key=f"btn_{ticket['number']}", use_container_width=True):
                st.session_state.selected_ticket = ticket['number']

            # Show title as small text
            st.caption(ticket['title'][:50] + "..." if len(ticket['title']) > 50 else ticket['title'])
            st.divider()

        # Refresh button
        if st.button("🔄 Refresh Tickets", use_container_width=True):
            st.rerun()

    # Main content area
    if 'selected_ticket' not in st.session_state:
        st.info("👈 Select a ticket from the sidebar to view details")

        # Show ticket summary table
        if tickets:
            st.subheader("Tickets Overview")
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Number": t["number"],
                    "Status": t["status"],
                    "Title": t["title"],
                    "Description": t["description"][:100] + "..." if len(t["description"]) > 100 else t["description"]
                }
                for t in tickets
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    ticket_number = st.session_state.selected_ticket
    content = read_ticket(ticket_number)

    if not content:
        st.error(f"Ticket {ticket_number} not found")
        return

    metadata = parse_ticket_metadata(content)

    # Ticket header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.header(f"Ticket {ticket_number}")
    with col2:
        status_badge = "🟢 Closed" if metadata["status"].lower() == "closed" else "🔴 Open"
        st.markdown(f"### {status_badge}")
    with col3:
        if st.button("🗑️ Deselect", key="deselect"):
            del st.session_state.selected_ticket
            st.rerun()

    st.subheader(metadata["title"])

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 Ticket Content", "⚙️ Process Ticket", "💬 Add Response"])

    with tab1:
        st.markdown("### Full Ticket Content")
        st.markdown(content)

    with tab2:
        st.markdown("### Process Ticket with Diagnostic Agent")
        st.info("This will run the diagnostic agent on the ticket and append results to the Exchange Log")

        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Processing ticket..."):
                process_ticket_stream(ticket_number)

    with tab3:
        st.markdown("### Add User Response to Exchange Log")
        st.info("Use this to respond to agent questions or provide additional information")

        user_response = st.text_area(
            "Your Response:",
            height=150,
            placeholder="Enter your response here...\n\nTip: Type 'skip' if you don't have the requested information"
        )

        if st.button("📝 Add Response", type="primary", use_container_width=True):
            if user_response.strip():
                append_to_exchange_log(ticket_number, "User", user_response.strip())
                st.success("Response added to ticket!")
                st.rerun()
            else:
                st.warning("Please enter a response before submitting")


if __name__ == "__main__":
    main()
