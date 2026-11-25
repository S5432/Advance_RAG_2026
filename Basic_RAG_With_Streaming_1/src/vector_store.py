"""Chroma-backed chunk store. Embeddings are computed here (one embedder for add AND search) and passed to Chroma explicitly."""
import logging
from datetime import datetime, timezone

logger = logging.getLogger("vector_store")
BATCH = 500

class EmbeddingMismatchError(RuntimeError):
    pass

class ChromaVectorStore:
    def __init__(self, db_path: str, collection_name: str, embedding_model, client=None):
        """embedding_model: a model-name string (loads sentence-transformers lazily) or any object with .encode(list[str]) -> ndarray
        and .model_name. `client` lets tests inject a Chroma client."""
        if isinstance(embedding_model, str):
            from src.embeddings import SentenceEmbedder
            embedding_model = SentenceEmbedder(embedding_model)
        self.embedder = embedding_model
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=db_path)           # data survives restarts
        self.client = client
        name = getattr(self.embedder, "model_name", "unknown")
        self.collection = client.get_or_create_collection(name=collection_name,
                                                          metadata={"hnsw:space": "cosine", "embedding_model": name})
        stored = (self.collection.metadata or {}).get("embedding_model")
        if stored and stored != name:
            raise EmbeddingMismatchError(f"Collection '{collection_name}' was built with '{stored}' but '{name}' is configured. "
                                         "Use another collection name or delete the collection and re-ingest.")

    # ---------- write ----------
    def add_document(self, document_id: str, chunks: list[str], metadata: list[dict]) -> int:
        """Store chunks of one document. Chunk id = f"{document_id}_chunk_{i}". Returns the number of chunks stored."""
        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length")
        if not chunks:
            return 0
        uploaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ids, metas = [], []
        for i, meta in enumerate(metadata):
            m = {k: v for k, v in meta.items() if v is not None}        # Chroma rejects None values
            m.update(document_id=document_id, chunk_index=i, uploaded_at=uploaded_at)
            ids.append(f"{document_id}_chunk_{i}"); metas.append(m)
        embeddings = self.embedder.encode(chunks)
        for s in range(0, len(ids), BATCH):
            self.collection.add(ids=ids[s:s + BATCH], documents=chunks[s:s + BATCH],
                                embeddings=[list(map(float, v)) for v in embeddings[s:s + BATCH]], metadatas=metas[s:s + BATCH])
        logger.info("chunks_stored document_id=%s chunks=%d", document_id, len(ids))
        return len(ids)

    def delete_document(self, document_id: str) -> int:
        found = self.collection.get(where={"document_id": document_id}, include=[])
        ids = found.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    # ---------- read ----------
    @staticmethod
    def _where(metadata_filter: dict | None):
        if not metadata_filter:
            return None
        conds = [{k: v} for k, v in metadata_filter.items()]
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def search(self, query: str, top_k: int = 4, metadata_filter: dict | None = None) -> list[dict]:
        total = self.collection.count()
        if total == 0 or not query.strip():
            return []
        q = self.embedder.encode([query])[0]
        res = self.collection.query(query_embeddings=[list(map(float, q))], n_results=max(1, min(top_k, total)),
                                    where=self._where(metadata_filter), include=["documents", "metadatas", "distances"])
        out = []
        for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"chunk_id": cid, "document_id": meta["document_id"], "document_name": meta.get("document_name", ""),
                        "text": text, "page": meta.get("page"), "chunk_index": meta.get("chunk_index"),
                        "score": round(max(-1.0, min(1.0, 1.0 - float(dist))), 4)})   # cosine distance -> similarity
        return out

    def get_document(self, document_id: str) -> dict | None:
        found = self.collection.get(where={"document_id": document_id}, include=["metadatas"])
        metas = found.get("metadatas") or []
        if not metas:
            return None
        first = metas[0]
        return {"document_id": document_id, "document_name": first.get("document_name", ""), "page_count": first.get("page_count"),
                "chunk_count": len(metas), "uploaded_at": first.get("uploaded_at")}

    def list_documents(self) -> list[dict]:
        found = self.collection.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in found.get("metadatas") or []:
            d = docs.setdefault(meta["document_id"], {"document_id": meta["document_id"], "document_name": meta.get("document_name", ""),
                                                      "page_count": meta.get("page_count"), "chunk_count": 0, "uploaded_at": meta.get("uploaded_at")})
            d["chunk_count"] += 1
        return sorted(docs.values(), key=lambda d: d["document_id"])

    def count(self) -> int:
        return self.collection.count()
