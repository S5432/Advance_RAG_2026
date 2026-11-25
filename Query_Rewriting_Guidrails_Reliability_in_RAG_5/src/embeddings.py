

"""
Embedding helpers using Pinecone Inference.

Model:
    llama-text-embed-v2

Dimensions:
    1024

Document input type:
    passage

Query input type:
    query
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
import math

from pinecone import Pinecone

from src.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    """Create and reuse the Pinecone client."""

    settings = get_settings()

    return Pinecone(
        api_key=settings.pinecone_api_key
    )


def _extract_embeddings(response: Any) -> list[list[float]]:
    """Extract embedding vectors from Pinecone response."""

    data = response.data

    vectors = []

    for item in data:

        if hasattr(item, "values"):
            values = item.values
        else:
            values = item["values"]

        vector = [float(value) for value in values]

        vectors.append(vector)

    return vectors


def _validate_vectors(
    vectors: list[list[float]],
    expected_count: int,
) -> None:
    """Validate embedding count and dimensions."""

    settings = get_settings()

    if len(vectors) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count}, "
            f"received {len(vectors)}"
        )

    for index, vector in enumerate(vectors):

        if len(vector) != settings.embedding_dimension:
            raise ValueError(
                f"Invalid embedding dimension at position {index}. "
                f"Expected {settings.embedding_dimension}, "
                f"received {len(vector)}"
            )

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                f"Invalid or non-finite embedding values at position {index}"
            )

def _embed(
    texts: list[str],
    input_type: str,
) -> list[list[float]]:
    """Generate embeddings using Pinecone Inference."""

    if not texts:
        return []

    settings = get_settings()
    client = get_pinecone_client()

    cleaned_texts = [
        text.strip()
        for text in texts
    ]

    if any(not text for text in cleaned_texts):
        raise ValueError("Embedding input cannot contain empty text.")

    response = client.inference.embed(
        model=settings.embedding_model,
        inputs=cleaned_texts,
        parameters={
            "input_type": input_type,
            "truncate": settings.embedding_truncate,
            "dimension": settings.embedding_dimension,
        },
    )

    vectors = _extract_embeddings(response)

    _validate_vectors(
        vectors=vectors,
        expected_count=len(cleaned_texts),
    )

    return vectors


# def embed_chunks(
#     chunks: list[dict],
#     batch_size: int | None = None,
# ) -> list[list[float]]:
#     """
#     Generate passage embeddings for document chunks.

#     Returns vectors in exactly the same order as chunks.
#     """

#     if not chunks:
#         return []

#     settings = get_settings()

#     batch_size = batch_size or settings.embedding_batch_size

#     if batch_size <= 0:
#         raise ValueError("batch_size must be greater than zero.")

#     all_vectors = []

#     for start in range(0, len(chunks), batch_size):

#         batch = chunks[start:start + batch_size]

#         texts = [
#             chunk["text"]
#             for chunk in batch
#         ]

#         vectors = _embed(
#             texts=texts,
#             input_type=settings.embedding_input_type_passage,
#         )

#         all_vectors.extend(vectors)

#         logger.info(
#             "Generated embeddings: %d/%d",
#             min(start + len(batch), len(chunks)),
#             len(chunks),
#         )

#     return all_vectors


# def embed_query(query: str) -> list[float]:
#     """
#     Generate a query embedding.

#     IMPORTANT:
#     Use input_type=query for user questions.
#     """

#     if not query or not query.strip():
#         raise ValueError("Query cannot be empty.")

#     settings = get_settings()

#     vectors = _embed(
#         texts=[query],
#         input_type=settings.embedding_input_type_query,
#     )

#     return vectors[0]
