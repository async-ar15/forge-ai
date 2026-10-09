"""Named retrieval constants (units and semantics in symbol names)."""

from __future__ import annotations

from typing import Final

# --- Semantic cache (PCR) ---
CACHE_SIMILARITY_THRESHOLD: Final[float] = 0.92
CACHE_CANDIDATE_ZSET_SCAN_LIMIT: Final[int] = 64
SEMANTIC_CACHE_KEY_PREFIX: Final[str] = "forgeai:pcr:v1"
SEMANTIC_CACHE_HITS_ZSET_SUFFIX: Final[str] = "hits"

# --- Prefetch ---
PREFETCH_MAX_CONCURRENT: Final[int] = 3
PREFETCH_TOP_SIMILAR_ENTRIES: Final[int] = 3

# --- Qdrant ---
QDRANT_DEFAULT_TOP_K: Final[int] = 5

# --- Embeddings (all-MiniLM-L6-v2) ---
EMBEDDING_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: Final[int] = 384
EMBEDDING_LRU_MAXSIZE: Final[int] = 1024

# --- Bloom (feature extraction + retrieval recording) ---
BLOOM_FILTER_KEY_DEFAULT: Final[str] = "forgeai:bloom:retrieval"

__all__ = [
    "BLOOM_FILTER_KEY_DEFAULT",
    "CACHE_CANDIDATE_ZSET_SCAN_LIMIT",
    "CACHE_SIMILARITY_THRESHOLD",
    "EMBEDDING_DIM",
    "EMBEDDING_LRU_MAXSIZE",
    "EMBEDDING_MODEL_NAME",
    "PREFETCH_MAX_CONCURRENT",
    "PREFETCH_TOP_SIMILAR_ENTRIES",
    "QDRANT_DEFAULT_TOP_K",
    "SEMANTIC_CACHE_HITS_ZSET_SUFFIX",
    "SEMANTIC_CACHE_KEY_PREFIX",
]
