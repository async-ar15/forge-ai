"""Sync gRPC ``RetrievalService`` over async PCR/Qdrant engine."""

from __future__ import annotations

import asyncio
import logging
from concurrent import futures
from typing import Final

import grpc

from forgeai.proto import retrieval_service_pb2, retrieval_service_pb2_grpc
from forgeai.retrieval.engine import RetrievalEngine

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


class ForgeRetrievalServicer(retrieval_service_pb2_grpc.RetrievalServiceServicer):
    """Unary ``Retrieve``; drives ``asyncio.run`` per request (v1 simplicity)."""

    __slots__ = ("_engine",)

    def __init__(self, engine: RetrievalEngine) -> None:
        self._engine = engine

    def Retrieve(
        self,
        request: retrieval_service_pb2.RetrievalRetrieveRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_service_pb2.RetrievalRetrieveResponse:
        """Synchronous gRPC entry; async engine isolated in a fresh event loop."""

        try:
            return asyncio.run(self._engine.retrieve(request))
        except Exception:
            _LOG.exception("retrieval_rpc_failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("retrieval_internal_error")
            return retrieval_service_pb2.RetrievalRetrieveResponse(
                cache_hit=False,
                retrieval_internal_latency_ms=0,
            )


def serve(engine: RetrievalEngine, *, port: int, max_workers: int = 8) -> None:
    """Start insecure ``RetrievalService`` until process signal."""

    servicer = ForgeRetrievalServicer(engine)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    retrieval_service_pb2_grpc.add_RetrievalServiceServicer_to_server(
        servicer,
        server,
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    _LOG.info("retrieval_grpc_listening port=%s", port)
    server.wait_for_termination()


def main() -> None:
    """Load settings, build engine, bind ``retrieval_engine_grpc_port``."""

    from forgeai.config import get_settings
    from forgeai.retrieval.clients import build_retrieval_clients
    from forgeai.retrieval.engine import build_engine

    cfg = get_settings()
    clients = build_retrieval_clients(cfg)
    engine = build_engine(cfg, clients)
    serve(engine, port=cfg.retrieval_engine_grpc_port)


if __name__ == "__main__":
    main()

__all__ = ["ForgeRetrievalServicer", "main", "serve"]
