"""Server-Sent Events framing for inference and error responses."""

from __future__ import annotations

import json
from typing import Any


def format_sse_event(payload: dict[str, Any]) -> str:
    """Return one SSE message (includes trailing blank line)."""

    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def sse_infer_chunk(
    *,
    token: str,
    is_final: bool,
    cost_microdollars: int | None,
    request_id: str,
) -> str:
    """One inference stream event."""

    body: dict[str, Any] = {
        "token": token,
        "is_final": is_final,
        "cost_microdollars": cost_microdollars,
        "request_id": request_id,
    }
    return format_sse_event(body)


def sse_error_chunk(*, error: str, request_id: str) -> str:
    """Terminal error shape for SSE responses."""

    return format_sse_event(
        {"error": error, "request_id": request_id, "is_final": True},
    )


__all__ = ["format_sse_event", "sse_error_chunk", "sse_infer_chunk"]
