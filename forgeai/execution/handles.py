"""Handles returned after registry resolution and engine materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forgeai.execution.quantization_config import QuantizationConfig


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """Stable binding between registry rows, on-disk weights, and a live engine."""

    model_version_id: str
    artifact_path: Path
    quant_profile_id: str | None
    quant_config: QuantizationConfig
    loaded_engine: object


__all__ = ["ModelHandle"]
