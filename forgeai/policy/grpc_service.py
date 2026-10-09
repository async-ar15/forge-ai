"""gRPC ``PolicyService`` implementation (Decide RPC, latency interceptor).

Does not own: gateway HTTP, TLS termination, or bandit training jobs.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import grpc

from forgeai.enums import ExplorationType as ExplorationTypeStr
from forgeai.policy.bandit import LinUCBBandit
from forgeai.policy.bandit_actions import action_spec_from_proto, action_spec_to_proto
from forgeai.policy.bandit_metrics import POLICY_RPC_DURATION_SECONDS
from forgeai.policy.features import FeatureVector
from forgeai.proto import common_pb2, policy_service_pb2, policy_service_pb2_grpc

_DECIDE_PATH: Final[str] = "/forgeai.v1.PolicyService/Decide"

if TYPE_CHECKING:

    class _TypedServerInterceptor(grpc.ServerInterceptor[Any, Any]):
        pass

else:

    class _TypedServerInterceptor(grpc.ServerInterceptor):
        pass


_EXPLORATION_TO_PROTO: Final[
    dict[ExplorationTypeStr, common_pb2.ExplorationType.ValueType]
] = {
    ExplorationTypeStr.NONE: common_pb2.EXPLORATION_TYPE_NONE,
    ExplorationTypeStr.EPSILON: common_pb2.EXPLORATION_TYPE_EPSILON,
    ExplorationTypeStr.UCB: common_pb2.EXPLORATION_TYPE_UCB,
    ExplorationTypeStr.FORCED: common_pb2.EXPLORATION_TYPE_FORCED,
}


class PolicyDecideLatencyInterceptor(_TypedServerInterceptor):
    """Record ``forgeai_policy_rpc_duration_seconds`` for unary ``Decide`` RPCs."""

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler[Any, Any] | None:
        """Wrap only ``Decide``; other handlers pass through unchanged."""

        base = continuation(handler_call_details)
        if base is None or handler_call_details.method != _DECIDE_PATH:
            return cast(grpc.RpcMethodHandler[Any, Any] | None, base)
        if base.request_streaming or base.response_streaming:
            return cast(grpc.RpcMethodHandler[Any, Any], base)
        orig = base.unary_unary

        def timed(request: object, context: grpc.ServicerContext) -> object:
            start = time.perf_counter()
            try:
                return orig(request, context)
            finally:
                POLICY_RPC_DURATION_SECONDS.observe(time.perf_counter() - start)

        return grpc.unary_unary_rpc_method_handler(
            timed,
            request_deserializer=base.request_deserializer,
            response_serializer=base.response_serializer,
        )


class ForgePolicyServicer(policy_service_pb2_grpc.PolicyServiceServicer):
    """Build ``PolicyDecideResponse`` from ``StateVector`` via ``LinUCBBandit``."""

    __slots__ = ("_bandit",)

    def __init__(self, bandit: LinUCBBandit) -> None:
        """Wire a shared bandit instance (process-wide posterior)."""

        self._bandit = bandit

    def Decide(
        self,
        request: policy_service_pb2.PolicyDecideRequest,
        _context: object,
    ) -> policy_service_pb2.PolicyDecideResponse:
        """O(81·d³) scoring; must stay within gateway pre-execution budget."""

        features = FeatureVector.from_proto(request.routing_state_features)
        decision = self._bandit.select_action(features)
        et = _EXPLORATION_TO_PROTO[decision.exploration_type]
        return policy_service_pb2.PolicyDecideResponse(
            chosen_action=action_spec_to_proto(decision.chosen_action),
            exploration_flag=decision.exploration_flag,
            exploration_type=et,
            greedy_action=action_spec_to_proto(decision.greedy_action),
            action_scores=dict(decision.action_scores),
            policy_version=self._bandit.policy_version,
        )

    def RecordReward(
        self,
        request: policy_service_pb2.RecordRewardRequest,
        _context: object,
    ) -> policy_service_pb2.RecordRewardResponse:
        """Apply synchronous online reward update on authoritative bandit."""

        try:
            action = action_spec_from_proto(request.action)
            features = FeatureVector.from_proto(request.feature_vector)
            self._bandit.update(action, float(request.r_online), features)
            return policy_service_pb2.RecordRewardResponse(
                acknowledged=True,
                policy_version=self._bandit.policy_version,
            )
        except Exception:
            return policy_service_pb2.RecordRewardResponse(
                acknowledged=False,
                policy_version=self._bandit.policy_version,
            )

    def LoadPolicy(
        self,
        request: policy_service_pb2.LoadPolicyRequest,
        _context: object,
    ) -> policy_service_pb2.LoadPolicyResponse:
        """Replace current policy with serialized LinUCB artifact."""

        blob = bytes(request.serialized_policy_bytes)
        if not blob:
            return policy_service_pb2.LoadPolicyResponse(
                acknowledged=False,
                new_policy_version=self._bandit.policy_version,
            )
        try:
            loaded = _load_bandit_from_bytes(blob)
            self._bandit = loaded
            return policy_service_pb2.LoadPolicyResponse(
                acknowledged=True,
                new_policy_version=self._bandit.policy_version,
            )
        except Exception:
            return policy_service_pb2.LoadPolicyResponse(
                acknowledged=False,
                new_policy_version=self._bandit.policy_version,
            )


def _load_bandit_from_bytes(blob: bytes) -> LinUCBBandit:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp.write(blob)
        path = Path(tmp.name)
    try:
        return LinUCBBandit.load(path)
    finally:
        path.unlink(missing_ok=True)


def serve(port: int, alpha: float, epsilon: float) -> None:
    """Start insecure PolicyService; ``port``, ``alpha``, ``epsilon`` are explicit.

    No ``get_settings()`` here — CLI passes env-derived values; tests pass literals.
    """

    bandit = LinUCBBandit(ridge_scale=alpha, ucb_alpha=alpha, epsilon=epsilon)
    servicer = ForgePolicyServicer(bandit)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        interceptors=[PolicyDecideLatencyInterceptor()],
    )
    policy_service_pb2_grpc.add_PolicyServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


__all__ = [
    "ForgePolicyServicer",
    "PolicyDecideLatencyInterceptor",
    "serve",
]


def main() -> None:
    """CLI module entry: load env, then ``serve(port, alpha, epsilon)``."""

    import os

    from forgeai.config import get_settings
    from forgeai.policy.constants import (
        DEFAULT_LINUCB_ALPHA_STR,
        DEFAULT_LINUCB_EXPLORATION_EPSILON_STR,
        LinUCBConfigKey,
    )

    cfg = get_settings()
    alpha = float(os.environ.get(LinUCBConfigKey.ALPHA, DEFAULT_LINUCB_ALPHA_STR))
    epsilon = float(
        os.environ.get(
            LinUCBConfigKey.EXPLORATION_EPSILON,
            DEFAULT_LINUCB_EXPLORATION_EPSILON_STR,
        )
    )
    serve(cfg.policy_service_port, alpha, epsilon)


if __name__ == "__main__":
    main()
