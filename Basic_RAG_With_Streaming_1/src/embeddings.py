"""One embedder object is shared by ingestion and querying, so both always use the same model and dimension."""
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer     # lazy: tests/CLI help do not need the model
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32 matrix of L2-normalised embeddings (cosine similarity == dot product)."""
        emb = self._load().encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

if __name__ == "__main__":
    e = SentenceEmbedder()
    m = e.encode(["Employees receive 12 paid leaves annually.", "What is the refund policy?"])
    print("model:", e.model_name, "| dimension:", e.dimension, "| matrix shape:", m.shape)
