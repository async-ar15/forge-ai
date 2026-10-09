"""Request-scoped logging context (request_id binding)."""

from __future__ import annotations

import contextvars
import logging
from typing import Final

_REQUEST_ID: Final[contextvars.ContextVar[str]] = contextvars.ContextVar(
    "gateway_request_id",
    default="",
)


def set_request_id(value: str) -> contextvars.Token[str]:
    """Bind ``request_id`` for downstream log filters."""

    return _REQUEST_ID.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """Restore prior context after the request completes."""

    _REQUEST_ID.reset(token)


def get_bound_request_id() -> str:
    """Return the active request id or empty string."""

    return _REQUEST_ID.get()


class RequestIdFilter(logging.Filter):
    """Inject ``request_id`` on every log record (empty when unset)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_bound_request_id()
        return True


__all__ = [
    "RequestIdFilter",
    "get_bound_request_id",
    "reset_request_id",
    "set_request_id",
]
