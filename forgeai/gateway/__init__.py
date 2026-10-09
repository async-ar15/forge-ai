"""Owns: the Inference Gateway Python package boundary (FastAPI HTTPS entrypoint).

Does not own: Policy Engine gRPC implementation, Execution Engine, or Retrieval Engine.
"""

from forgeai.gateway.app import create_app

__all__ = ["create_app"]
