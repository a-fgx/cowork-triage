"""
Vector Store Configuration

ChromaDB is a local vector database that stores embeddings and enables
fast similarity search. It persists data to disk for reuse between runs.

Key Concept: Vector Store
    A vector store is like a database optimized for finding similar items:
    1. You add documents (text + metadata)
    2. Each document is converted to a vector (embedding)
    3. When you search, your query is also converted to a vector
    4. The store finds documents with the most similar vectors

Architecture:
    Document → Embedding Model → Vector → ChromaDB
    Query → Embedding Model → Vector → Similarity Search → Results
"""

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import config
from src.rag.embeddings import embeddings


def get_vectorstore(
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path] = None,
) -> Chroma:
    """
    Get or create a ChromaDB vector store.

    If the persist directory exists, loads the existing store.
    Otherwise, creates a new empty store.

    Args:
        collection_name: Name of the collection (defaults to config)
        persist_directory: Where to store data (defaults to config)

    Returns:
        Configured Chroma vector store

    Example:
        store = get_vectorstore()

        # Add documents
        store.add_documents([
            Document(page_content="Error: X", metadata={"solution": "Do Y"}),
        ])

        # Search for similar documents
        results = store.similarity_search("Error X problem", k=3)
    """
    collection = collection_name or config.chroma_collection
    persist_dir = persist_directory or config.chroma_persist_dir

    # Ensure the directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def add_documents_to_store(
    documents: list[Document],
    vectorstore: Optional[Chroma] = None,
) -> Chroma:
    """
    Add documents to the vector store.

    Each document should have:
    - page_content: The text to embed and search
    - metadata: Additional info (solution, source, etc.)

    Args:
        documents: List of LangChain Document objects
        vectorstore: Existing store (creates new if None)

    Returns:
        The vector store (same as input or newly created)

    Example:
        docs = [
            Document(
                page_content="TypeError: 'NoneType' object is not callable",
                metadata={
                    "solution": "Check that the variable is not None before calling",
                    "source": "kaggle_dataset"
                }
            ),
        ]
        store = add_documents_to_store(docs)
    """
    store = vectorstore or get_vectorstore()
    store.add_documents(documents)
    return store


def similarity_search(
    query: str,
    k: int = 5,
    vectorstore: Optional[Chroma] = None,
) -> list[Document]:
    """
    Search for documents similar to the query.

    Args:
        query: The text to search for
        k: Number of results to return
        vectorstore: Store to search (uses default if None)

    Returns:
        List of similar Documents

    Example:
        results = similarity_search("RuntimeError in async function")
        for doc in results:
            print(f"Error: {doc.page_content}")
            print(f"Solution: {doc.metadata.get('solution')}")
    """
    store = vectorstore or get_vectorstore()
    return store.similarity_search(query, k=k)


def similarity_search_with_score(
    query: str,
    k: int = 5,
    vectorstore: Optional[Chroma] = None,
) -> list[tuple[Document, float]]:
    """
    Search for documents similar to the query, with similarity scores.

    Scores are distances (lower = more similar).
    Typical range: 0.0 (identical) to 2.0 (very different).

    Args:
        query: The text to search for
        k: Number of results to return
        vectorstore: Store to search (uses default if None)

    Returns:
        List of (Document, score) tuples

    Example:
        results = similarity_search_with_score("event loop error")
        for doc, score in results:
            if score < 0.5:  # Very similar
                print(f"Strong match: {doc.page_content}")
    """
    store = vectorstore or get_vectorstore()
    return store.similarity_search_with_score(query, k=k)


# =============================================================================
# Convenience: Global vector store instance
# =============================================================================

# Lazy initialization - created on first access
_vectorstore: Optional[Chroma] = None


def get_default_vectorstore() -> Chroma:
    """Get the default vector store (creates if needed)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore
