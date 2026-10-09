"""Retrieved passage representation (internal; serialized to gRPC strings)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One ranked passage from Qdrant or semantic cache."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


_PASSAGE_VERSION: Final[str] = "forgeai.chunk.v1"


def chunk_to_passage_utf8(chunk: RetrievedChunk) -> str:
    """JSON line shape for ``context_passage_utf8`` (stable for downstream)."""

    payload = {
        "v": _PASSAGE_VERSION,
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": chunk.score,
        "metadata": chunk.metadata,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def chunks_from_passages(passages: list[str]) -> list[RetrievedChunk]:
    """Parse passages back to chunks (best-effort for tests)."""

    out: list[RetrievedChunk] = []
    for p in passages:
        d = json.loads(p)
        out.append(
            RetrievedChunk(
                chunk_id=str(d["chunk_id"]),
                text=str(d["text"]),
                score=float(d["score"]),
                metadata=dict(d.get("metadata", {})),
            ),
        )
    return out


__all__ = ["RetrievedChunk", "chunk_to_passage_utf8", "chunks_from_passages"]
