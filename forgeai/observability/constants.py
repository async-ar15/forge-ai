"""Owns: observability-local configuration keys and metric label names.

Does not own: service SLO targets, on-call rotations, or trace sampling policies.
"""

from __future__ import annotations

from enum import StrEnum


class MetricsLabel(StrEnum):
    """Shared label keys attached to exported Prometheus series."""

    TENANT_ID = "tenant_id"
    SERVICE_NAME = "service_name"


__all__ = ["MetricsLabel"]
