"""FAISS-backed chunk store: full-text embeddings, metadata, retrieval."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
DIM = 384
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 80


class ChunkStore:
    """In-memory FAISS index over text chunks with paper metadata."""

    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBED_MODEL)
        self._index = faiss.IndexFlatL2(DIM)
        self._chunks: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self._index = faiss.IndexFlatL2(DIM)
        self._chunks = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        faiss.normalize_L2(arr)
        return arr

    def add_chunks(
        self,
        texts: List[str],
        paper_title: str,
        link: str,
        section: str,
        source: str,
    ) -> int:
        """Add non-empty chunks. Returns number of vectors added."""
        added = 0
        batch_texts: List[str] = []
        batch_meta: List[Dict[str, Any]] = []

        for t in texts:
            t = (t or "").strip()
            if len(t) < MIN_CHUNK_CHARS:
                continue
            batch_texts.append(t)
            batch_meta.append(
                {
                    "text": t,
                    "paper_title": paper_title,
                    "link": link,
                    "section": section,
                    "source": source,
                }
            )

        if not batch_texts:
            return 0

        mat = self._embed(batch_texts)
        self._index.add(mat)
        self._chunks.extend(batch_meta)
        added = len(batch_meta)
        logger.debug("Added %d chunks for paper=%r section=%s", added, paper_title, section)
        return added

    def search(self, query: str, k: int = 16) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q or self._index.ntotal == 0:
            return []

        k = min(k, int(self._index.ntotal))
        if k <= 0:
            return []

        qv = self._embed([q])
        scores, indices = self._index.search(qv, k)
        out: List[Dict[str, Any]] = []
        for dist, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            row = dict(self._chunks[idx])
            row["distance"] = float(dist)
            out.append(row)
        return out


_active_store: Optional[ChunkStore] = None


def set_active_store(store: Optional[ChunkStore]) -> None:
    global _active_store
    _active_store = store


def get_active_store() -> Optional[ChunkStore]:
    return _active_store
