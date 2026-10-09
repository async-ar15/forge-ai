"""Append-only JSONL sink for decision durability when Postgres is unavailable."""

from __future__ import annotations

import json
import logging
from typing import Any, Final

_LOG = logging.getLogger(__name__)

# JSONL envelope field (not a SQL column); names the durability failure class.
JSONL_FALLBACK_REASON_KEY: Final[str] = "fallback_reason"


def append_fallback_jsonl(path: str, payload: dict[str, Any]) -> None:
    """Append one JSON object; raises ``OSError`` on I/O failure."""

    line = json.dumps(payload, default=str, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)


def try_append_fallback_jsonl(path: str, payload: dict[str, Any]) -> bool:
    """Return True on success; on failure log CRITICAL and return False."""

    try:
        append_fallback_jsonl(path, payload)
    except OSError as exc:
        try:
            blob = json.dumps(payload, default=str, sort_keys=True)
        except (TypeError, ValueError):
            blob = repr(payload)
        _LOG.critical(
            "decision_fallback_jsonl_failed path=%s err=%s payload=%s",
            path,
            exc,
            blob,
        )
        return False
    return True


__all__ = [
    "JSONL_FALLBACK_REASON_KEY",
    "append_fallback_jsonl",
    "try_append_fallback_jsonl",
]
