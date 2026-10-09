"""On-disk checks before declaring INT4/AWQ loads (Section 5)."""

from __future__ import annotations

from pathlib import Path

from forgeai.enums import Precision
from forgeai.execution.exceptions import QuantizationMismatchError

_AWQ_MARKERS: tuple[str, ...] = (
    "quant_config.json",
    "quantize_config.json",
)


def require_awq_layout(artifact_dir: Path) -> None:
    """Fail fast when INT4 is requested but AWQ metadata is absent."""

    if not artifact_dir.is_dir():
        msg = "artifact path is not a directory"
        raise QuantizationMismatchError(
            precision=Precision.INT4,
            artifact_path=str(artifact_dir),
            detail=msg,
        )
    found = any((artifact_dir / name).is_file() for name in _AWQ_MARKERS)
    if not found:
        raise QuantizationMismatchError(
            precision=Precision.INT4,
            artifact_path=str(artifact_dir),
            detail=f"missing_awq_marker expected_one_of={list(_AWQ_MARKERS)}",
        )


__all__ = ["require_awq_layout"]
