"""
Configuration Management for Cowork-Triage Agent

This module handles loading and validating all configuration settings.
It uses environment variables for sensitive data (API keys) and provides
sensible defaults for optional settings.

Key Pattern: Dataclass + Environment Variables
- Dataclass provides type hints and defaults
- os.getenv() loads from environment (or .env file via python-dotenv)
- validate() method ensures required fields are present
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists (useful for local development)
# This must happen before we read os.getenv()
load_dotenv()


@dataclass
class Config:
    """
    Central configuration for the diagnostic agent.

    Usage:
        config = Config()
        config.validate()  # Raises ValueError if required fields missing

        # Access settings
        print(config.google_api_key)
        print(config.langsmith_tracing)
    """

    # === Google Gemini API ===
    # Required for LLM reasoning
    google_api_key: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", "")
    )

    # Model to use for reasoning (gemini-2.0-flash for speed, gemini-1.5-pro for quality)
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # === GitHub API ===
    # Optional - for searching existing issues
    github_token: str = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN", "")
    )

    # Default repository to search (format: owner/repo)
    default_repo: str = field(
        default_factory=lambda: os.getenv("DEFAULT_REPO", "langchain-ai/langgraph")
    )

    # All repositories to search for LangChain ecosystem issues
    langchain_repos: list = field(
        default_factory=lambda: [
            "langchain-ai/langchain",
            "langchain-ai/langgraph",
            "langchain-ai/langsmith-sdk",
        ]
    )

    # === LangSmith Observability ===
    # Optional but recommended for debugging
    langsmith_api_key: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY", "")
    )

    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "cowork-triage")
    )

    # Set to "true" to enable tracing
    langsmith_tracing: bool = field(
        default_factory=lambda: os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    # === RAG Settings ===
    # Path to store ChromaDB data
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        )
    )

    # Collection name in ChromaDB
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "error_messages")
    )

    # === Agent Settings ===
    # Maximum times to ask user for missing info before proceeding
    max_info_gathering_attempts: int = 3

    # Maximum number of hypotheses to generate
    max_hypotheses: int = 5

    def validate(self) -> None:
        """
        Validate that required configuration is present.

        Raises:
            ValueError: If required fields are missing
        """
        errors = []

        if not self.google_api_key:
            errors.append(
                "GOOGLE_API_KEY is required. "
                "Get one at: https://aistudio.google.com/app/apikey"
            )

        if errors:
            raise ValueError("\n".join(errors))

    def setup_langsmith(self) -> None:
        """
        Configure LangSmith environment variables.

        LangSmith reads from environment variables, so we need to set them
        before importing LangChain modules.
        """
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
            os.environ["LANGSMITH_TRACING"] = "true"
            print(f"LangSmith tracing enabled for project: {self.langsmith_project}")
        else:
            os.environ["LANGSMITH_TRACING"] = "false"


# Global config instance (singleton pattern)
# Import and use: from src.config import config
config = Config()
