"""Blocking RegistryService call for execution model resolution."""

from __future__ import annotations

from typing import cast

from forgeai.proto import common_pb2, registry_service_pb2, registry_service_pb2_grpc


def resolve_execution_model_sync(
    stub: registry_service_pb2_grpc.RegistryServiceStub,
    *,
    model_tier: common_pb2.ModelTier.ValueType,
    execution_precision: common_pb2.Precision.ValueType,
    serving_environment: common_pb2.ServingEnvironment.ValueType,
    timeout_seconds: float = 5.0,
) -> registry_service_pb2.RegistryResolveExecutionModelResponse:
    """Invoke ``RegistryResolveExecutionModel`` (sync gRPC stub)."""

    req = registry_service_pb2.RegistryResolveExecutionModelRequest(
        model_tier=model_tier,
        execution_precision=execution_precision,
        serving_environment=serving_environment,
    )
    raw = stub.RegistryResolveExecutionModel(req, timeout=timeout_seconds)
    return cast(registry_service_pb2.RegistryResolveExecutionModelResponse, raw)


__all__ = ["resolve_execution_model_sync"]
