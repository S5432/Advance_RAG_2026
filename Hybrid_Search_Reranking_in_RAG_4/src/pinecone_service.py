


"""
Pinecone Documents API service for schema-based indexes.

Expected index schema:
    chunk_text:
        String field
        Integrated embedding: llama-text-embed-v2
        Dimension: 1024
        Metric: cosine

    title:
        String field

All other document attributes are stored as metadata.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from pinecone import EmbedQuery, Pinecone

from src.config import get_settings

logger = logging.getLogger(__name__)


REQUIRED_METADATA = (
    "document_id",
    "document_name",
    "source_type",
    "page_number",
    "chunk_index",
    "chunk_id",
    "section",
    "uploaded_by",
    "tenant_id",
    "created_at",
)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or SDK response object."""

    if isinstance(obj, dict):
        return obj.get(key, default)

    getter = getattr(obj, "get", None)

    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(obj, key, default)


def _schema_fields(description: Any) -> dict:
    """Extract schema fields from the Pinecone index description."""

    schema = _get_value(description, "schema")

    if schema is None:
        raise RuntimeError(
            "The configured Pinecone index does not expose a document schema. "
            "This application requires a schema-based document index."
        )

    fields = _get_value(schema, "fields")

    if not isinstance(fields, dict):
        raise RuntimeError("Unable to read Pinecone document schema fields.")

    return fields


def _validate_document_schema(description: Any, settings) -> None:
    """Validate the schema required by this application."""

    fields = _schema_fields(description)

    if "chunk_text" not in fields:
        raise RuntimeError(
            "Pinecone index schema is missing the required 'chunk_text' field."
        )

    if "title" not in fields:
        raise RuntimeError(
            "Pinecone index schema is missing the required 'title' field."
        )

    chunk_field = fields["chunk_text"]
    embed_config = _get_value(chunk_field, "embed")

    if embed_config is None:
        raise RuntimeError(
            "'chunk_text' must have integrated embedding enabled."
        )

    actual_model = _get_value(embed_config, "model")
    actual_dimension = _get_value(embed_config, "dimension")
    actual_metric = _get_value(embed_config, "metric")

    if actual_model != settings.embedding_model:
        raise RuntimeError(
            f"Embedding model mismatch: index uses '{actual_model}', "
            f"but application expects '{settings.embedding_model}'."
        )

    if actual_dimension != settings.embedding_dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: index uses "
            f"{actual_dimension}, application expects "
            f"{settings.embedding_dimension}."
        )

    if actual_metric != settings.embedding_metric:
        raise RuntimeError(
            f"Embedding metric mismatch: index uses '{actual_metric}', "
            f"application expects '{settings.embedding_metric}'."
        )

    logger.info(
        "Validated Pinecone schema: field=chunk_text model=%s "
        "dimension=%s metric=%s",
        actual_model,
        actual_dimension,
        actual_metric,
    )


def get_pinecone_index(settings):
    """
    Connect to an existing schema-based Pinecone index.

    Deliberately does not auto-create an index with a different schema.
    """

    pc = Pinecone(api_key=settings.pinecone_api_key)

    name = settings.pinecone_index_name

    description = pc.indexes.describe(name=name)

    _validate_document_schema(description, settings)

    index = pc.index(name=name)

    logger.info(
        "Connected to schema-based Pinecone index=%s",
        name,
    )

    return index


class PineconeService:
    """Wrapper around the Pinecone Documents API."""

    def __init__(self, index, namespace: str = "documents"):
        self.index = index
        self.namespace = namespace

    @classmethod
    def from_settings(cls, settings) -> "PineconeService":
        return cls(
            index=get_pinecone_index(settings),
            namespace=settings.pinecone_namespace,
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """
        Retry transient HTTP errors only.

        Permanent client errors such as HTTP 400 are not retried.
        """

        status = (
            getattr(exc, "status", None)
            or getattr(exc, "status_code", None)
        )

        if status is None:
            return False

        try:
            status = int(status)
        except (TypeError, ValueError):
            return False

        return status in (408, 425, 429) or status >= 500

    def _retry(self, fn, attempts: int = 3):
        for attempt in range(attempts):
            try:
                return fn()

            except Exception as exc:
                if (
                    attempt == attempts - 1
                    or not self._retryable(exc)
                ):
                    raise

                wait = 2 ** attempt

                logger.warning(
                    "Transient Pinecone failure (%s). "
                    "Retrying in %s seconds.",
                    exc,
                    wait,
                )

                time.sleep(wait)

        raise RuntimeError("Pinecone operation failed unexpectedly.")

    # ---------------------------------------------------------
    # UPSERT DOCUMENTS
    # ---------------------------------------------------------

    def upsert_documents(
        self,
        documents: list[dict],
        namespace: str | None = None,
        batch_size: int = 50,
    ) -> int:
        """
        Insert or replace schema-based Pinecone documents.

        Pinecone generates embeddings from the chunk_text field.
        """

        if not documents:
            return 0

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        ns = namespace or self.namespace
        total_upserted = 0

        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]

            response = self._retry(
                lambda b=batch: self.index.documents.upsert(
                    namespace=ns,
                    documents=b,
                )
            )

            upserted_count = _get_value(
                response,
                "upserted_count",
                len(batch),
            )

            total_upserted += int(upserted_count)

            logger.info(
                "Pinecone document upsert progress: %d/%d",
                total_upserted,
                len(documents),
            )

        return total_upserted

    # ---------------------------------------------------------
    # SEMANTIC SEARCH
    # ---------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        namespace: str | None = None,
    ) -> list[dict]:
        """
        Semantic search using Pinecone integrated embedding.

        No client-side query embedding is generated.
        """

        if not query_text or not query_text.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        ns = namespace or self.namespace

        response = self._retry(
            lambda: self.index.documents.search(
                namespace=ns,
                top_k=top_k,
                score_by=[
                    EmbedQuery(
                        field="chunk_text",
                        query=query_text.strip(),
                    )
                ],
                filter=metadata_filter or None,
                include_fields=["*"],
            )
        )

        results = []

        for match in _get_value(response, "matches", []):

            match_id = (
                _get_value(match, "id")
                or _get_value(match, "_id")
            )

            score = _get_value(match, "score", 0.0)

            chunk_text = _get_value(match, "chunk_text", "")

            title = _get_value(match, "title", "unknown")

            # Preserve all custom metadata fields.
            metadata = {}

            for field in REQUIRED_METADATA:
                value = _get_value(match, field)

                if value is not None:
                    metadata[field] = value

            # Preserve additional optional metadata when present.
            optional_fields = (
                "source",
                "url",
                "page_title",
                "row_number",
                "ocr",
                "ocr_confidence",
                "low_ocr_quality",
                # Day 11
                "source_url",
                "parent_chunk_id",
                "chunk_type",
                "chunking_strategy",
                "department",
                "access_level",
                "language",
                "token_count",
                "created_at_ts",
            )

            for field in optional_fields:
                value = _get_value(match, field)

                if value is not None:
                    metadata[field] = value

            metadata["document_name"] = metadata.get(
                "document_name",
                title,
            )

            # Backward compatibility with existing RAG chain.
            metadata["text"] = chunk_text

            results.append(
                {
                    "id": match_id,
                    "score": float(score),
                    "metadata": metadata,
                }
            )

        return results

    # ---------------------------------------------------------
    # FETCH DOCUMENTS
    # ---------------------------------------------------------

    def fetch_documents(
        self,
        ids: list[str],
        namespace: str | None = None,
    ):
        if not ids:
            return {}

        response = self._retry(
            lambda: self.index.documents.fetch(
                namespace=namespace or self.namespace,
                ids=ids,
            )
        )

        return _get_value(response, "documents", {})

    # ---------------------------------------------------------
    # DELETE DOCUMENTS
    # ---------------------------------------------------------

    def delete_ids(
        self,
        ids: list[str],
        namespace: str | None = None,
    ) -> None:
        """Delete documents using their IDs."""

        if not ids:
            return

        ns = namespace or self.namespace

        for start in range(0, len(ids), 1000):
            batch = ids[start:start + 1000]

            self._retry(
                lambda b=batch: self.index.documents.delete(
                    namespace=ns,
                    ids=b,
                )
            )

        logger.info(
            "Requested deletion of %d Pinecone documents.",
            len(ids),
        )

    def delete_by_document_id(
        self,
        document_id: str,
        namespace: str | None = None,
    ) -> None:
        """Delete all chunks belonging to a document."""

        if not document_id:
            raise ValueError("document_id cannot be empty.")

        self._retry(
            lambda: self.index.documents.delete(
                namespace=namespace or self.namespace,
                filter={
                    "document_id": {
                        "$eq": document_id
                    }
                },
            )
        )

        logger.info(
            "Requested deletion for document_id=%s",
            document_id,
        )

    # ---------------------------------------------------------
    # HEALTH CHECK
    # ---------------------------------------------------------

    def ping(self) -> bool:
        self.index.describe_index_stats()
        return True


# =============================================================
# CHUNKS -> DOCUMENTS -> PINECONE
# =============================================================

def upsert_chunks_to_pinecone(
    chunks: list[dict],
    pinecone_service: PineconeService,
    namespace: str = "documents",
) -> list[str]:
    """
    Convert application chunks into schema-based Pinecone documents.

    Important:
        No embeddings argument is accepted.
        Pinecone generates embeddings from chunk_text.
    """

    if not chunks:
        return []

    now = datetime.now(timezone.utc).isoformat()

    documents = []

    for chunk in chunks:

        text = str(chunk.get("text", "")).strip()

        if not text:
            raise ValueError("Chunk text cannot be empty.")

        metadata = dict(chunk.get("metadata") or {})

        metadata.setdefault(
            "document_name",
            metadata.get("source", "unknown"),
        )

        metadata.setdefault("created_at", now)
        metadata.setdefault("section", "")

        chunk_id = metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError(
                "Every chunk must contain a unique chunk_id."
            )

        missing = [
            field
            for field in REQUIRED_METADATA
            if metadata.get(field) is None
        ]

        if missing:
            raise ValueError(
                f"Chunk {chunk_id} is missing metadata: {missing}"
            )

        # The text belongs in the schema's chunk_text field,
        # not in the old vector metadata["text"] field.
        metadata.pop("text", None)
        metadata.pop("chunk_text", None)
        metadata = {k: v for k, v in metadata.items() if v is not None}  # Pinecone rejects null values

        # Keep title as the document's searchable/display field.
        title = str(metadata["document_name"])

        document = {
            "_id": str(chunk_id),
            "chunk_text": text,
            "title": title,
            **metadata,
        }

        documents.append(document)

    upserted_count = pinecone_service.upsert_documents(
        documents=documents,
        namespace=namespace,
    )

    if upserted_count != len(documents):
        raise RuntimeError(
            f"Pinecone upsert count mismatch: "
            f"expected {len(documents)}, received {upserted_count}"
        )

    logger.info(
        "Successfully upserted %d documents into namespace '%s'.",
        upserted_count,
        namespace,
    )

    return [
        document["_id"]
        for document in documents
    ]


# =============================================================
# DAY 11 - INDEX CREATION / CONNECTION (llama-text-embed-v2, 1024, cosine)
# =============================================================

def validate_embedding_settings(settings) -> None:
    """Fail fast unless the configuration is llama-text-embed-v2 / 1024 / cosine."""
    if settings.embedding_model != "llama-text-embed-v2":
        raise ValueError(f"embedding_model must be llama-text-embed-v2, got {settings.embedding_model}")
    if int(settings.embedding_dimension) != 1024:
        raise ValueError(f"embedding_dimension must be 1024, got {settings.embedding_dimension}")
    if settings.embedding_metric != "cosine":
        raise ValueError(f"embedding_metric must be cosine, got {settings.embedding_metric}")


def build_index_schema(settings) -> dict:
    """Schema-based index (same shape that _validate_document_schema reads back)."""
    return {
        "fields": {
            "chunk_text": {
                "type": "string",
                "full_text_search": {"language": "en"},
                "embed": {
                    "model": settings.embedding_model,
                    "dimension": int(settings.embedding_dimension),
                    "metric": settings.embedding_metric,
                },
            },
            "title": {"type": "string"},
        }
    }


def _not_found(exc: Exception) -> bool:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    return status == 404 or "not found" in str(exc).lower()


def create_index_if_missing(pc, settings) -> bool:
    """Create the index when it does not exist. Returns True if it was created.

    NOTE: the create call mirrors the Day 9 schema-index call; if your SDK version uses a different
    signature, create the index once in the Pinecone console (same schema) - connection still works.
    """
    validate_embedding_settings(settings)
    name = settings.pinecone_index_name
    try:
        pc.indexes.describe(name=name)
        return False
    except Exception as exc:
        if not _not_found(exc):
            raise
    pc.indexes.create(
        name=name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        schema=build_index_schema(settings),
    )
    logger.info("Created Pinecone index %s", name)
    return True


def connect_pinecone(settings, create: bool = False):
    """Validated connection; `create=True` creates the index first when missing."""
    validate_embedding_settings(settings)
    if create:
        create_index_if_missing(Pinecone(api_key=settings.pinecone_api_key), settings)
    return get_pinecone_index(settings)
