# Cowork Triage System

Automated bug diagnosis and triage system for LangChain ecosystem issues using LangGraph and AI agents.

## Features

- 🎯 **Automated Bug Classification**: Categorizes issues by failure type (API, version, dependency, runtime, configuration)
- 🔍 **GitHub Issue Search**: Finds similar issues across LangChain repositories
- 📚 **RAG-based Knowledge Base**: Searches vector database for similar error patterns
- 🧠 **Root Cause Analysis**: Generates diagnostic hypotheses with confidence scores
- 💡 **Resolution Planning**: Provides step-by-step fix plans
- 🎫 **Ticket Management UI**: Web-based interface for viewing and processing tickets

## Installation

### Prerequisites

- Python 3.11 or higher
- Google Gemini API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cowork-triage
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

4. (Optional) Load the knowledge base dataset:
```bash
python scripts/load_dataset.py
```

## Usage

### Web UI (Recommended)

The web-based UI provides an easy-to-use interface for managing and processing tickets.

**Start the UI:**
```bash
# Using the shell script
./start_ui.sh

# OR using the Python launcher
python start_ui.py

# OR directly with Streamlit
streamlit run ui_app.py
```

**Access the UI:**
Open your browser to http://localhost:8501

**Features:**
- 📋 View all incoming tickets in the sidebar
- 🔍 Select a ticket to view full details
- ⚙️ Process tickets with the diagnostic agent
- 💬 Add user responses to the exchange log
- 🔄 Real-time processing updates

### Command Line Interface

**Interactive Mode:**
```bash
python start.py
```
Launches an interactive session where you can describe bugs directly.

**Batch Processing:**
```bash
# Process a specific ticket
python scripts/process_ticket.py 001

# List all available tickets
python scripts/process_ticket.py
```

## Ticket Format

Tickets are stored as Markdown files in `incoming_tickets/` with the following structure:

```markdown
# Ticket 001

## Status: Open

## Title
Brief description of the issue

## Description
Detailed description of the bug

## Reproduction Steps / Example Code (Python)
```python
# Code to reproduce the issue
```

## Error Message and Stack Trace (if applicable)
Full error output

## Exchange Log
### [Agent] - 2026-01-23 14:53:04
Agent's diagnostic response

### [User] - 2026-01-23 15:10:22
User's follow-up response
```

## Workflow

1. **Ticket Submission**: Add a new ticket file to `incoming_tickets/`
2. **Processing**: Use the UI or CLI to process the ticket
3. **Classification**: Agent classifies the failure type
4. **Research**: Searches GitHub and knowledge base for similar issues
5. **Diagnosis**: Generates root cause hypotheses with confidence scores
6. **Resolution**: If high confidence, provides a fix plan
7. **Follow-up**: If more info needed, agent asks questions in the Exchange Log
8. **Iteration**: User responds and re-processes until resolved

## Project Structure

```
cowork-triage/
├── ui_app.py                # Streamlit web UI
├── start_ui.py              # UI launcher script
├── start_ui.sh              # Shell script launcher
├── start.py                 # Interactive CLI
├── incoming_tickets/        # Ticket storage
│   ├── ticket_001.md
│   └── ticket_002.md
├── scripts/
│   ├── process_ticket.py    # Batch ticket processor
│   └── load_dataset.py      # Load knowledge base
└── src/
    ├── agent.py             # LangGraph workflow
    ├── config.py            # Configuration
    ├── state.py             # State schema
    ├── nodes/               # Workflow nodes
    ├── tools/               # External tools (GitHub API)
    ├── rag/                 # Vector database
    └── prompts/             # LLM prompts
```

## Configuration

Edit `.env` file to configure:

- `GOOGLE_API_KEY`: Required for Gemini API
- `LANGSMITH_API_KEY`: Optional for tracing
- `LANGSMITH_TRACING`: Set to `true` to enable tracing
- `GITHUB_TOKEN`: Optional for higher GitHub API rate limits

## Technology Stack

- **LangGraph 0.2.0+**: Workflow orchestration
- **LangChain**: LLM integrations and tools
- **Google Gemini 2.0-flash**: LLM provider
- **ChromaDB 0.5.0**: Vector database
- **Streamlit 1.30.0+**: Web UI framework
- **httpx**: HTTP client for GitHub API

## Development

**Run tests:**
```bash
pytest
```

**Code formatting:**
```bash
ruff check .
ruff format .
```

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
