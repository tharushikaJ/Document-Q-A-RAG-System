"""
A minimal, persistent vector store backed by FAISS.

Uses cosine similarity (via inner product on L2-normalized vectors),
which is the standard choice for sentence-transformer embeddings.
Metadata (source text, filename, chunk id) is stored alongside the
index in a JSONL file, keyed by row order.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine sim on normalized vecs
        self.metadata: list[dict] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings and L2-normalize for cosine similarity."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        faiss.normalize_L2(embeddings)
        return embeddings.astype("float32")

    def add(self, records: list[dict]):
        """
        records: list of {"source", "chunk_id", "text"}
        Embeds the text field and adds to the index + metadata store.
        """
        if not records:
            return
        texts = [r["text"] for r in records]
        vectors = self.embed(texts)
        self.index.add(vectors)
        self.metadata.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k metadata records with similarity scores, best first."""
        if self.index.ntotal == 0:
            return []
        query_vec = self.embed([query])
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(score)
            results.append(record)
        return results

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            for record in self.metadata:
                f.write(json.dumps(record) + "\n")

    def load(self):
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                "No saved index found. Run `python ingest.py` first."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        self.metadata = []
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def __len__(self):
        return self.index.ntotal
