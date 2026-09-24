"""Small persistent FAISS knowledge store used by the RAG workflow.

The index is deliberately generated locally from the versioned knowledge-base
file.  Generated files are ignored, keeping source ZIPs portable and clean.
"""
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from services.matcher import get_embedding_model

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = BASE_DIR / "data" / "career_knowledge.md"
STORE_DIR = BASE_DIR / "data" / "vector_store"
INDEX_PATH = STORE_DIR / "career_knowledge.faiss"
MANIFEST_PATH = STORE_DIR / "career_knowledge.json"


def chunk_text(text: str, chunk_size: int = 160, overlap: int = 30) -> list[str]:
    """Split text on word boundaries with overlap so context survives chunks."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[start:start + chunk_size]) for start in range(0, len(words), step)]


def _documents() -> list[dict]:
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    return [
        {"id": index, "source": KNOWLEDGE_PATH.name, "text": chunk}
        for index, chunk in enumerate(chunks)
    ]


def build_index(force: bool = False) -> int:
    """Embed knowledge chunks and persist FAISS plus a readable metadata manifest."""
    if not force and INDEX_PATH.exists() and MANIFEST_PATH.exists():
        return len(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))
    documents = _documents()
    if not documents:
        raise ValueError("The career knowledge base contains no indexable text.")
    vectors = get_embedding_model().encode(
        [document["text"] for document in documents], normalize_embeddings=True
    )
    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])  # normalized embeddings => cosine similarity
    index.add(matrix)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    MANIFEST_PATH.write_text(json.dumps(documents, indent=2), encoding="utf-8")
    return len(documents)


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """Return the most semantically relevant career/skill knowledge chunks."""
    build_index()
    index = faiss.read_index(str(INDEX_PATH))
    documents = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    query_vector = get_embedding_model().encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(query_vector, dtype="float32"), min(top_k, len(documents)))
    return [
        {**documents[item_id], "score": round(float(score), 4)}
        for score, item_id in zip(scores[0], ids[0]) if item_id != -1
    ]
