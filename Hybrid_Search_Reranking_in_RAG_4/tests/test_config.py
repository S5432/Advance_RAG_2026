"""Phase 1: configuration + Pinecone index creation/validation (no network)."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.pinecone_service import (build_index_schema, create_index_if_missing,
                                  validate_embedding_settings)

BASE = dict(_env_file=None, gemini_api_key="g", pinecone_api_key="p")


def test_defaults_match_assignment():
    s = Settings(**BASE)
    assert (s.embedding_model, s.embedding_dimension, s.embedding_metric) == ("llama-text-embed-v2", 1024, "cosine")
    assert s.pinecone_index_name == "advanced-rag-document-qa" and s.pinecone_namespace == "documents"
    assert (s.dense_top_k, s.hybrid_top_k, s.final_top_k, s.rrf_k) == (20, 20, 5, 60)
    assert s.chunking_strategy == "recursive"
    assert s.rerank_model == "bge-reranker-v2-m3" and s.rerank_min_score == 0.05


@pytest.mark.parametrize("bad", [dict(embedding_dimension=768), dict(embedding_dimension=2048),
                                 dict(embedding_metric="euclidean"), dict(embedding_model="all-MiniLM-L6-v2")])
def test_invalid_embedding_config_rejected(bad):
    with pytest.raises(ValidationError):
        Settings(**BASE, **bad)


def test_validate_embedding_settings_runtime_guard():
    ok = SimpleNamespace(embedding_model="llama-text-embed-v2", embedding_dimension=1024, embedding_metric="cosine")
    validate_embedding_settings(ok)
    for field, val in [("embedding_dimension", 768), ("embedding_metric", "dotproduct")]:
        with pytest.raises(ValueError):
            validate_embedding_settings(SimpleNamespace(**{**vars(ok), field: val}))


class _PC:
    def __init__(self, exists):
        self.created, self._exists = [], exists
        self.indexes = SimpleNamespace(describe=self._describe, create=lambda **kw: self.created.append(kw))

    def _describe(self, name):
        if not self._exists:
            raise type("NotFound", (Exception,), {"status": 404})("not found")
        return {}


def _settings():
    return SimpleNamespace(embedding_model="llama-text-embed-v2", embedding_dimension=1024, embedding_metric="cosine",
                           pinecone_index_name="advanced-rag-document-qa", pinecone_cloud="aws", pinecone_region="us-east-1")


def test_index_created_only_when_missing_with_correct_schema():
    pc = _PC(exists=False)
    assert create_index_if_missing(pc, _settings()) is True
    emb = pc.created[0]["schema"]["fields"]["chunk_text"]["embed"]
    assert emb == {"model": "llama-text-embed-v2", "dimension": 1024, "metric": "cosine"}
    assert pc.created[0]["name"] == "advanced-rag-document-qa"
    pc2 = _PC(exists=True)
    assert create_index_if_missing(pc2, _settings()) is False and not pc2.created


def test_schema_has_title_and_chunk_text():
    assert set(build_index_schema(_settings())["fields"]) == {"chunk_text", "title"}
