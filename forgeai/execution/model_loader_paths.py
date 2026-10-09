"""Filesystem layout for mirrored registry artifacts."""

from __future__ import annotations

from pathlib import Path


def local_artifact_directory(
    cache_dir: Path,
    model_version_id: str,
    object_id: str,
) -> Path:
    """Return the directory vLLM should open as ``model=``."""

    rel = object_id.strip().strip("/").replace("..", "")
    if rel:
        return cache_dir / model_version_id / rel
    return cache_dir / model_version_id


__all__ = ["local_artifact_directory"]
