# Cowork-Triage

A LangGraph-based AI diagnostic agent that analyzes bug reports and provides structured diagnoses and resolution plans. Designed specifically for troubleshooting issues in the LangChain ecosystem (LangChain, LangGraph, LangSmith).

## Features

- **LLM Reasoning** - Powered by Google Gemini 2.0 Flash
- **GitHub Issue Search** - Finds related known issues across repositories
- **RAG Knowledge Base** - Searches error patterns using ChromaDB
- **Human-in-the-Loop** - Asks clarifying questions when confidence is low
- **LangSmith Integration** - Full tracing and observability

## Project Structure

```
cowork-triage/
├── src/                          # Main source code
│   ├── agent.py                  # LangGraph assembly and state management
│   ├── state.py                  # TypedDict schemas for workflow state
│   ├── config.py                 # Configuration and environment management
│   ├── llm.py                    # Google Gemini LLM integration
│   ├── nodes/                    # Workflow nodes (processing steps)
│   │   ├── intake.py             # Extract structured info from bug reports
│   │   ├── classifier.py         # Categorize bug type and missing info
│   │   ├── diagnoser.py          # Generate diagnostic hypotheses
│   │   ├── info_gatherer.py      # Ask user for missing information
│   │   └── resolution.py         # Generate step-by-step fix plan
│   ├── rag/                      # Retrieval-Augmented Generation
│   │   ├── embeddings.py         # Google embedding model config
│   │   ├── vectorstore.py        # ChromaDB vector store setup
│   │   └── retriever.py          # RAG search interface
│   ├── tools/                    # External tools
│   │   └── github_tools.py       # GitHub API integration
│   └── prompts/
│       └── templates.py          # LLM prompt templates
├── scripts/
│   ├── load_dataset.py           # Load error messages into ChromaDB
│   └── process_ticket.py         # Batch process tickets from files
├── incoming_tickets/             # Sample ticket files for batch processing
├── data/                         # Data storage (CSV datasets, ChromaDB)
├── start.py                      # Main entry point
├── pyproject.toml                # Project dependencies
├── langgraph.json                # LangGraph API/Studio configuration
└── .env.example                  # Example environment variables
```

## Workflow

The agent processes bug reports through a multi-stage pipeline:

```
START
  ↓
[intake] → Extracts structured info from raw bug report
  ↓
[classifier] → Determines bug type (api/version/dependency/runtime/config)
  ↓
[github_search] ←── Parallel ──→ [rag_search]
  ↓
[diagnoser] → Generates ranked hypotheses with confidence breakdown
  ↓
[route_after_diagnosis] → Conditional routing:
  ├─ High confidence → [resolution] → END
  └─ Low confidence → [info_gatherer] → [user_input] → [diagnoser] (loop)
```

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cowork-triage
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google AI API key for Gemini LLM |
| `GITHUB_TOKEN` | No | GitHub token (increases rate limits) |
| `LANGSMITH_API_KEY` | No | LangSmith API key for tracing |
| `LANGSMITH_TRACING` | No | Set to `true` to enable tracing |
| `KAGGLE_USERNAME` | No | Kaggle username for dataset download |
| `KAGGLE_KEY` | No | Kaggle API key |

## Usage

### Load Knowledge Base (Optional)

Load error patterns into the ChromaDB vector store for RAG:

```bash
python scripts/load_dataset.py
```

This downloads a dataset from Kaggle or uses a built-in demo dataset as fallback.

### Run Modes

**1. Interactive Mode (Default)**

```bash
python start.py
```

Enter your bug report and press Enter twice to submit. You can process multiple bugs in one session.

**2. Command-Line Mode**

```bash
python start.py "RuntimeError: Event loop is already running when using async tools"
```

Processes the bug report immediately and exits.

**3. Batch Ticket Processing**

Process ticket files from `incoming_tickets/`:

```bash
python scripts/process_ticket.py 001          # Process ticket_001.md
python scripts/process_ticket.py 001 --clean  # Reset exchange log first
```

## Output

The agent produces a structured markdown report containing:

- Detected libraries and components
- Related GitHub issues (with links)
- Root cause hypothesis with evidence
- Confidence breakdown table
- Step-by-step resolution plan

### Confidence Scoring

The diagnosis confidence is computed from weighted sources:

| Source | Weight |
|--------|--------|
| LLM classification | 30% |
| GitHub matches | 35% |
| RAG knowledge base | 20% |
| Library detection | 15% |

## Configuration

### LangGraph Studio

The project includes `langgraph.json` for deployment with LangGraph Studio:

```bash
langgraph dev  # Start local development server
```

### Customization

- **Add error patterns**: Place CSV files in `data/` and run `load_dataset.py`
- **Change target repositories**: Update `DEFAULT_REPO` in `.env`
- **Adjust prompts**: Modify templates in `src/prompts/templates.py`

## License

See LICENSE file for details.
