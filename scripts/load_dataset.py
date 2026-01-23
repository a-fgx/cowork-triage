"""
Dataset Loading Script

Downloads an error messages dataset from Kaggle and loads it into ChromaDB.

This script demonstrates the ETL (Extract, Transform, Load) process:
1. Extract: Download data from Kaggle
2. Transform: Parse and structure the data
3. Load: Store in the vector database

Usage:
    python scripts/load_dataset.py

Prerequisites:
    1. Kaggle API credentials at ~/.kaggle/kaggle.json
       (Download from https://www.kaggle.com/settings)
    2. Set GOOGLE_API_KEY for embeddings

Note:
    If you don't have a Kaggle account or want to test quickly,
    you can use the demo data at the bottom of this file.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file BEFORE importing kaggle
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Set Kaggle credentials from .env if present
# Kaggle API expects KAGGLE_USERNAME and KAGGLE_KEY
if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

import csv
import json
from typing import Optional

from langchain_core.documents import Document

from src.config import config
from src.rag.vectorstore import add_documents_to_store, get_vectorstore


# =============================================================================
# Kaggle Dataset Download
# =============================================================================


def download_kaggle_dataset(
    dataset_name: str = "shamimhasan8/python-code-bug-and-fix-pairs",
    output_dir: Path = project_root / "src" / "data",
) -> Optional[Path]:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset identifier (user/dataset-name)
        output_dir: Where to save the files

    Returns:
        Path to the downloaded files, or None if failed

    Example:
        path = download_kaggle_dataset("victoriaguo/programming-errors")
        if path:
            print(f"Downloaded to {path}")
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True,
        )

        print(f"Downloaded to: {output_dir}")
        return output_dir

    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Using demo data instead...")
        return None


# =============================================================================
# Data Parsing
# =============================================================================


def parse_csv_dataset(file_path: Path) -> list[Document]:
    """
    Parse a CSV file containing error messages and solutions.

    Supports multiple formats:
    - error / error_message / message: The error text
    - solution / fix / resolution: How to fix it
    - buggy_code / fixed_code: Code pairs (shamimhasan8/python-code-bug-and-fix-pairs)
    - category / type: Error category (optional)

    Args:
        file_path: Path to the CSV file

    Returns:
        List of Document objects for the vector store
    """
    documents = []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Check for buggy_code/fixed_code format (code bug fix pairs dataset)
            buggy_code = row.get("buggy_code", "")
            fixed_code = row.get("fixed_code", "")

            if buggy_code and fixed_code:
                # This is a code bug/fix pairs dataset
                commit_message = row.get("commit_message", "")
                doc = Document(
                    page_content=buggy_code.strip(),
                    metadata={
                        "solution": fixed_code.strip(),
                        "commit_message": commit_message.strip(),
                        "source": "kaggle_dataset",
                        "category": "code_fix",
                    },
                )
                documents.append(doc)
                continue

            # Find error column (original format)
            error = (
                row.get("error")
                or row.get("error_message")
                or row.get("message")
                or row.get("Error")
                or ""
            )

            # Find solution column
            solution = (
                row.get("solution")
                or row.get("fix")
                or row.get("resolution")
                or row.get("Solution")
                or ""
            )

            if not error:
                continue

            doc = Document(
                page_content=error.strip(),
                metadata={
                    "solution": solution.strip(),
                    "source": "kaggle_dataset",
                    "category": row.get("category", row.get("type", "unknown")),
                },
            )
            documents.append(doc)

    return documents


# =============================================================================
# Demo Data (fallback if no Kaggle)
# =============================================================================

DEMO_ERROR_DATA = [
    {
        "error": "RuntimeError: Event loop is already running",
        "solution": "Install nest_asyncio and call nest_asyncio.apply() at the start of your script. This is common in Jupyter notebooks.",
        "category": "runtime",
    },
    {
        "error": "TypeError: 'NoneType' object is not callable",
        "solution": "The variable you're calling as a function is None. Check if it was properly assigned and isn't being shadowed by another variable.",
        "category": "runtime",
    },
    {
        "error": "ImportError: No module named 'xyz'",
        "solution": "Install the missing package with: pip install xyz. If already installed, check you're using the correct Python environment.",
        "category": "dependency",
    },
    {
        "error": "ModuleNotFoundError: No module named 'xyz'",
        "solution": "Install the missing package with: pip install xyz. Ensure your virtual environment is activated.",
        "category": "dependency",
    },
    {
        "error": "KeyError: 'key_name'",
        "solution": "The dictionary doesn't contain the key you're accessing. Use .get('key_name', default) to provide a fallback value.",
        "category": "runtime",
    },
    {
        "error": "ValueError: too many values to unpack",
        "solution": "The sequence has more elements than variables. Check the expected format of the data you're unpacking.",
        "category": "runtime",
    },
    {
        "error": "AttributeError: 'NoneType' object has no attribute 'xyz'",
        "solution": "You're calling a method on None. Add a None check before accessing the attribute, or investigate why the object is None.",
        "category": "runtime",
    },
    {
        "error": "requests.exceptions.ConnectionError: Failed to establish connection",
        "solution": "Check your internet connection and verify the URL is correct. The server might be down or blocking your request.",
        "category": "api",
    },
    {
        "error": "JSONDecodeError: Expecting value",
        "solution": "The response is not valid JSON. Check if the API returned an error message or HTML instead of JSON.",
        "category": "api",
    },
    {
        "error": "PermissionError: [Errno 13] Permission denied",
        "solution": "You don't have permission to access this file. Check file permissions or run with elevated privileges.",
        "category": "configuration",
    },
    {
        "error": "FileNotFoundError: [Errno 2] No such file or directory",
        "solution": "The file doesn't exist at the specified path. Check the path is correct and the file exists.",
        "category": "configuration",
    },
    {
        "error": "DeprecationWarning: function xyz is deprecated",
        "solution": "Update your code to use the new recommended function. Check the documentation for migration guide.",
        "category": "version",
    },
    {
        "error": "SSL: CERTIFICATE_VERIFY_FAILED",
        "solution": "Certificate verification failed. Update your CA certificates or use requests with verify=False (not recommended for production).",
        "category": "api",
    },
    {
        "error": "RecursionError: maximum recursion depth exceeded",
        "solution": "Your function is calling itself too many times. Add a base case to stop recursion or convert to an iterative approach.",
        "category": "runtime",
    },
    {
        "error": "MemoryError",
        "solution": "Your program ran out of memory. Process data in smaller chunks, use generators, or increase available memory.",
        "category": "runtime",
    },
    {
        "error": "TimeoutError: Operation timed out",
        "solution": "The operation took too long. Increase the timeout value or optimize the slow operation.",
        "category": "api",
    },
    {
        "error": "OSError: [Errno 28] No space left on device",
        "solution": "Disk is full. Free up space by removing unnecessary files or logs.",
        "category": "configuration",
    },
    {
        "error": "UnicodeDecodeError: 'utf-8' codec can't decode byte",
        "solution": "The file isn't UTF-8 encoded. Try opening with encoding='latin-1' or detect the encoding first.",
        "category": "runtime",
    },
    {
        "error": "AssertionError",
        "solution": "An assert statement failed. Check the condition being asserted and the values involved.",
        "category": "runtime",
    },
    {
        "error": "IndentationError: unexpected indent",
        "solution": "Python found inconsistent indentation. Use consistent spaces (4) or tabs throughout your code.",
        "category": "runtime",
    },
]


def create_demo_documents() -> list[Document]:
    """Create Document objects from the demo data."""
    return [
        Document(
            page_content=item["error"],
            metadata={
                "solution": item["solution"],
                "source": "demo_data",
                "category": item["category"],
            },
        )
        for item in DEMO_ERROR_DATA
    ]


# =============================================================================
# Main Script
# =============================================================================


def load_dataset():
    """
    Main function to load the dataset into ChromaDB.

    Tries to download from Kaggle first, falls back to demo data.
    """
    print("=" * 60)
    print("Loading Error Messages Dataset into ChromaDB")
    print("=" * 60)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set up your .env file.")
        return

    # Try to download from Kaggle
    data_dir = download_kaggle_dataset()

    if data_dir:
        # Find CSV files
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV file(s)")
            documents = []
            for csv_file in csv_files:
                print(f"Parsing: {csv_file.name}")
                docs = parse_csv_dataset(csv_file)
                documents.extend(docs)
                print(f"  Found {len(docs)} entries")
        else:
            print("No CSV files found, using demo data")
            documents = create_demo_documents()
    else:
        print("\nUsing demo data (20 common error patterns)")
        documents = create_demo_documents()

    # Load into ChromaDB
    print(f"\nLoading {len(documents)} documents into ChromaDB...")
    vectorstore = get_vectorstore()
    add_documents_to_store(documents, vectorstore)

    print(f"\nDone! Documents stored at: {config.chroma_persist_dir}")
    print("=" * 60)


if __name__ == "__main__":
    load_dataset()
