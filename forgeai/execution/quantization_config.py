"""Quantization metadata attached to a loaded engine (Section 5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class QuantizationMethodKind(StrEnum):
    """Coarse method label mirrored from registry ``quant_profiles.method``."""

    NONE = "none"
    BITSANDBYTES_INT8 = "bitsandbytes_int8"
    AWQ_INT4 = "awq_int4"


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Observed quantization load outcome for auditing and registry write-back."""

    method: QuantizationMethodKind
    bits: int
    loaded_at_utc: datetime
    memory_mb: float


__all__ = ["QuantizationConfig", "QuantizationMethodKind"]
