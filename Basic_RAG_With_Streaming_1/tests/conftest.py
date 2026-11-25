import re
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "data" / "sample_policy.pdf"

class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder (no model download). Same instance is used for add and search."""
    model_name = "fake-embedder"
    dim = 512
    STOP = {"the", "a", "an", "of", "to", "is", "are", "what", "how", "do", "does", "should", "after", "and", "in", "for", "on", "get"}
    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in self.STOP: continue
                w = w[:-1] if w.endswith("s") and len(w) > 3 else w          # crude stemming: leaves -> leave
                out[r, sum((i + 1) * ord(c) for i, c in enumerate(w)) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True); n[n == 0] = 1
        return out / n
    @property
    def dimension(self): return self.dim

class FakeGemini:
    """Scripted stand-in for GeminiClient. `structured` is a dict, an exception, or a list of those (consumed in order)."""
    def __init__(self, structured=None, stream_chunks=("Employees get 12 paid leaves ", "[C1]."), error=None):
        self.structured = structured if isinstance(structured, list) else [structured]
        self.stream_chunks, self.error, self.calls, self.prompts = stream_chunks, error, 0, []
    def generate_structured(self, prompt, schema, system_prompt=None, temperature=0.2, top_p=0.8, max_output_tokens=1024):
        self.prompts.append((prompt, system_prompt)); self.calls += 1
        if self.error: raise self.error
        item = self.structured[min(self.calls - 1, len(self.structured) - 1)]
        if isinstance(item, Exception): raise item
        return item
    def stream(self, prompt, system_prompt=None, temperature=0.7, top_p=0.95, max_output_tokens=1024, model=None):
        self.prompts.append((prompt, system_prompt)); self.calls += 1
        if self.error: raise self.error
        yield from self.stream_chunks

GOOD_ANSWER = {"answer": "Employees are eligible for 12 paid leaves per year [C1].", "source_ids": ["C1"],
               "insufficient_context": False, "confidence": "high"}
NO_INFO = {"answer": "I don't have enough information in the provided documents.", "source_ids": [],
           "insufficient_context": True, "confidence": "low"}

@pytest.fixture()
def embedder(): return FakeEmbedder()

@pytest.fixture()
def store(tmp_path, embedder):
    from src.vector_store import ChromaVectorStore
    return ChromaVectorStore(str(tmp_path / "chroma"), "test_docs", embedder)

@pytest.fixture()
def sample_bytes(): return SAMPLE_PDF.read_bytes()

@pytest.fixture()
def ingested(store, sample_bytes):
    from src.ingestion import ingest_pdf
    return store, ingest_pdf(sample_bytes, "hr_policy.pdf", store, 800, 150)
