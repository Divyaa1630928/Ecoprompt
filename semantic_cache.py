"""
ecoprompt/semantic_cache.py
──────────────────────────────────────────────────────────────────────────────
Semantic Caching Module for EcoPrompt
──────────────────────────────────────────────────────────────────────────────
How it works:
  1. Every incoming query is converted to a dense embedding vector using the
     lightweight all-MiniLM-L6-v2 sentence-transformer model (22 MB, runs
     fully on CPU).
  2. The embedding is compared against a local FAISS flat-index of previously
     seen queries using cosine similarity.
  3. If the nearest neighbour exceeds `similarity_threshold`, the cached
     response is returned immediately — no LLM call is made.
  4. On a cache miss the query is forwarded to the LLM, the response is
     stored in the cache, and metrics are updated.
  5. A TTL sweep removes entries older than `ttl_seconds` (default 24 h).

Water / energy saving rationale
  ─────────────────────────────
  Google's 2023 Environmental Report discloses ~0.3 L of water per Wh of
  energy used in its data centres.  Microsoft's 2023 Sustainability Report
  cites roughly 0.5 ml of water consumed per typical ChatGPT-class query
  (4 0-50 input + output tokens, moderate inference).  We adopt a
  CONSERVATIVE estimate of 0.5 ml per avoided LLM call for illustrative
  purposes. Energy saving is approximated at 0.003 kWh per avoided call
  (based on publicly reported figures for mid-size transformer inference).

  ⚠️  These numbers are approximations derived from published third-party
  disclosures and intended solely for educational illustration.  Actual
  values vary by model size, hardware, and data-centre efficiency.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── constants ──────────────────────────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"   # 22 MB, 384-dim embeddings, MIT licence
_EMBEDDING_DIM = 384

# Published approximations (see docstring above)
WATER_PER_SAVED_CALL_ML = 0.5       # millilitres
ENERGY_PER_SAVED_CALL_KWH = 0.003  # kilowatt-hours


# ── data structures ────────────────────────────────────────────────────────
@dataclass
class CacheEntry:
    """A single record stored in the semantic cache."""
    entry_id: str
    query: str
    response: str
    embedding: np.ndarray          # shape (384,)
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


@dataclass
class CacheResult:
    """Returned by SemanticCache.query()."""
    hit: bool                       # True  → served from cache
    response: str                   # The response text
    similarity: float               # Cosine similarity to nearest cached query (0–1)
    matched_query: Optional[str]    # The original cached query that matched
    entry_id: Optional[str]         # ID of the cache entry that was used


# ── main class ─────────────────────────────────────────────────────────────
class SemanticCache:
    """
    In-process semantic cache backed by a FAISS flat L2 index.

    Parameters
    ----------
    similarity_threshold : float
        Cosine similarity above which a cached answer is reused (0–1).
        Recommended: 0.85.  Lower = more cache hits but riskier accuracy.
    ttl_seconds : float
        Time-to-live for each cache entry in seconds.  Default 86 400 (24 h).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        ttl_seconds: float = 86_400,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        # Model is loaded lazily on first use so the Streamlit app starts instantly.
        self._model: SentenceTransformer | None = None

        # FAISS inner-product index on L2-normalised vectors → cosine similarity
        self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)

        # Ordered list of entries — index position maps to FAISS row id
        self._entries: list[CacheEntry] = []

        # ── metrics counters ──────────────────────────────────────────────
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def query(self, user_query: str) -> CacheResult:
        """
        Check whether a semantically similar response already exists in cache.

        Returns a CacheResult.  If `hit` is True, use the `response` field
        directly.  If False, call the LLM and then call `store()`.
        """
        self.total_queries += 1
        self._evict_expired()

        embedding = self._embed(user_query)

        if self._index.ntotal == 0:
            # Cache is empty — guaranteed miss
            self.cache_misses += 1
            return CacheResult(hit=False, response="", similarity=0.0,
                               matched_query=None, entry_id=None)

        # Search for the single nearest neighbour
        distances, indices = self._index.search(
            embedding.reshape(1, -1).astype(np.float32), k=1
        )
        similarity = float(distances[0][0])   # inner-product on unit vecs = cosine sim
        idx = int(indices[0][0])

        if idx < 0 or idx >= len(self._entries):
            # FAISS returned a ghost index (can happen after eviction rebuild)
            self.cache_misses += 1
            return CacheResult(hit=False, response="", similarity=similarity,
                               matched_query=None, entry_id=None)

        entry = self._entries[idx]

        if similarity >= self.similarity_threshold:
            self.cache_hits += 1
            return CacheResult(
                hit=True,
                response=entry.response,
                similarity=similarity,
                matched_query=entry.query,
                entry_id=entry.entry_id,
            )

        self.cache_misses += 1
        return CacheResult(hit=False, response="", similarity=similarity,
                           matched_query=None, entry_id=None)

    def store(self, query: str, response: str) -> str:
        """
        Add a new query+response pair to the cache.

        Returns the new entry's ID.
        """
        embedding = self._embed(query)
        entry = CacheEntry(
            entry_id=str(uuid.uuid4()),
            query=query,
            response=response,
            embedding=embedding,
        )
        self._entries.append(entry)
        self._index.add(embedding.reshape(1, -1).astype(np.float32))
        return entry.entry_id

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()
        self._index.reset()
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def save(self, path: str = "ecoprompt_cache.pkl") -> None:
        """Persist the cache to disk so it survives restarts."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "entries": self._entries,
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            }, f)

    def load(self, path: str = "ecoprompt_cache.pkl") -> None:
        """Load a previously saved cache from disk."""
        import os, pickle
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._entries = data["entries"]
        self.total_queries = data["total_queries"]
        self.cache_hits = data["cache_hits"]
        self.cache_misses = data["cache_misses"]
        # Rebuild FAISS index from loaded entries
        self._index.reset()
        if self._entries:
            import numpy as np
            matrix = np.stack([e.embedding for e in self._entries]).astype("float32")
            self._index.add(matrix)

    # ── derived metrics ────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0–1)."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    @property
    def water_saved_ml(self) -> float:
        """Estimated water saved (ml) from avoided LLM calls."""
        return self.cache_hits * WATER_PER_SAVED_CALL_ML

    @property
    def energy_saved_kwh(self) -> float:
        """Estimated energy saved (kWh) from avoided LLM calls."""
        return self.cache_hits * ENERGY_PER_SAVED_CALL_KWH

    @property
    def cache_size(self) -> int:
        return len(self._entries)

    # ── internal helpers ───────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Return a unit-norm embedding vector for `text`. Loads model on first call."""
        if self._model is None:
            self._model = SentenceTransformer(_MODEL_NAME)
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def _evict_expired(self) -> None:
        """Remove entries older than ttl_seconds and rebuild the FAISS index."""
        live = [e for e in self._entries if not e.is_expired(self.ttl_seconds)]
        if len(live) == len(self._entries):
            return   # nothing expired

        self._entries = live
        self._index.reset()
        if live:
            matrix = np.stack([e.embedding for e in live]).astype(np.float32)
            self._index.add(matrix)
