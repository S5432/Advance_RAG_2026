"""One-off: create (if missing) and validate the Day 11 Pinecone index.   python -m scripts.create_index"""
from src.config import get_settings
from src.pinecone_service import connect_pinecone

if __name__ == "__main__":
    s = get_settings()
    connect_pinecone(s, create=True)
    print(f"OK index={s.pinecone_index_name} model={s.embedding_model} dim={s.embedding_dimension} "
          f"metric={s.embedding_metric} namespace={s.pinecone_namespace}")
