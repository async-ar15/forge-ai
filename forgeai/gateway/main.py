"""ASGI entrypoint for ``uvicorn forgeai.gateway.main:app``."""

from __future__ import annotations

from forgeai.gateway.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
