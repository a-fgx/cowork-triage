"""
Embedding Model Configuration

Embeddings convert text into vectors (lists of numbers) that capture semantic meaning.
Similar texts produce similar vectors, enabling semantic search.

Key Concept: Vector Similarity
    - "Python TypeError" → [0.2, 0.8, 0.1, ...]
    - "Python type error" → [0.21, 0.79, 0.12, ...]  (very similar vector!)
    - "JavaScript callback" → [0.9, 0.1, 0.3, ...]  (very different vector)

    We use cosine similarity to find the closest matches.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import config


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get the configured embedding model.

    Uses Google's text-embedding model for creating vectors.
    The same model must be used for indexing and querying.

    Returns:
        Configured embedding model

    Example:
        embeddings = get_embeddings()

        # Embed a single text
        vector = embeddings.embed_query("RuntimeError: event loop")

        # Embed multiple texts
        vectors = embeddings.embed_documents([
            "TypeError in function",
            "ImportError: no module"
        ])
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=config.google_api_key,
        # Task type helps optimize embeddings for retrieval
        task_type="retrieval_document",
    )


# Pre-configured embeddings instance
embeddings = get_embeddings()
