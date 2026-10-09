"""Section 3 joint action space: 81 arms, canonical keys, protobuf mapping.

Does not own: LinUCB algebra, feature extraction, or persistence layout.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from forgeai.enums import ModelTier, OutputBudget, Precision, RetrievalMode
from forgeai.proto import common_pb2

ACTION_SPACE_SIZE: Final[int] = 81


class ActionSpec(NamedTuple):
    """Joint action; field order matches ``forgeai.v1.ActionSpec`` on the wire."""

    model_tier: ModelTier
    precision: Precision
    retrieval_mode: RetrievalMode
    output_budget: OutputBudget


_MODEL_PROTO: Final[dict[ModelTier, common_pb2.ModelTier.ValueType]] = {
    ModelTier.SMALL: common_pb2.MODEL_TIER_SMALL,
    ModelTier.MEDIUM: common_pb2.MODEL_TIER_MEDIUM,
    ModelTier.LARGE: common_pb2.MODEL_TIER_LARGE,
}
_PRECISION_PROTO: Final[dict[Precision, common_pb2.Precision.ValueType]] = {
    Precision.FP16: common_pb2.PRECISION_FP16,
    Precision.INT8: common_pb2.PRECISION_INT8,
    Precision.INT4: common_pb2.PRECISION_INT4,
}
_RETRIEVAL_PROTO: Final[dict[RetrievalMode, common_pb2.RetrievalMode.ValueType]] = {
    RetrievalMode.OFF: common_pb2.RETRIEVAL_MODE_OFF,
    RetrievalMode.CACHE_ONLY: common_pb2.RETRIEVAL_MODE_CACHE_ONLY,
    RetrievalMode.FULL: common_pb2.RETRIEVAL_MODE_FULL,
}
_OUTPUT_PROTO: Final[dict[OutputBudget, common_pb2.OutputBudget.ValueType]] = {
    OutputBudget.SHORT: common_pb2.OUTPUT_BUDGET_SHORT,
    OutputBudget.MEDIUM: common_pb2.OUTPUT_BUDGET_MEDIUM,
    OutputBudget.LONG: common_pb2.OUTPUT_BUDGET_LONG,
}

_PROTO_MODEL: Final[dict[int, ModelTier]] = {v: k for k, v in _MODEL_PROTO.items()}
_PROTO_PRECISION: Final[dict[int, Precision]] = {
    v: k for k, v in _PRECISION_PROTO.items()
}
_PROTO_RETRIEVAL: Final[dict[int, RetrievalMode]] = {
    v: k for k, v in _RETRIEVAL_PROTO.items()
}
_PROTO_OUTPUT: Final[dict[int, OutputBudget]] = {v: k for k, v in _OUTPUT_PROTO.items()}


def canonical_action_key(action: ActionSpec) -> str:
    """Return the routing_decisions / proto map key (lower_snake, no enum prefixes)."""

    return (
        f"{action.model_tier.value}_{action.precision.value}_"
        f"{action.retrieval_mode.value}_{action.output_budget.value}"
    )


def all_actions_in_order() -> tuple[ActionSpec, ...]:
    """Fixed arm index order: nested loops model → precision → retrieval → output."""

    out: list[ActionSpec] = []
    for mt in ModelTier:
        for pr in Precision:
            for rm in RetrievalMode:
                for ob in OutputBudget:
                    out.append(ActionSpec(mt, pr, rm, ob))
    assert len(out) == ACTION_SPACE_SIZE
    return tuple(out)


ALL_ACTIONS: Final[tuple[ActionSpec, ...]] = all_actions_in_order()
_ACTION_INDEX: Final[dict[str, int]] = {
    canonical_action_key(a): i for i, a in enumerate(ALL_ACTIONS)
}


def action_spec_to_proto(action: ActionSpec) -> common_pb2.ActionSpec:
    """Map domain ``ActionSpec`` to ``forgeai.v1.ActionSpec``."""

    return common_pb2.ActionSpec(
        model_tier=_MODEL_PROTO[action.model_tier],
        precision=_PRECISION_PROTO[action.precision],
        retrieval_mode=_RETRIEVAL_PROTO[action.retrieval_mode],
        output_budget=_OUTPUT_PROTO[action.output_budget],
    )


def action_spec_from_proto(message: common_pb2.ActionSpec) -> ActionSpec:
    """Parse protobuf ``ActionSpec``; unknown wire values raise ``KeyError``."""

    return ActionSpec(
        model_tier=_PROTO_MODEL[int(message.model_tier)],
        precision=_PROTO_PRECISION[int(message.precision)],
        retrieval_mode=_PROTO_RETRIEVAL[int(message.retrieval_mode)],
        output_budget=_PROTO_OUTPUT[int(message.output_budget)],
    )


def action_index(action: ActionSpec) -> int:
    """Stable arm index in ``[0, 80]`` for LinUCB state."""

    return _ACTION_INDEX[canonical_action_key(action)]


__all__ = [
    "ACTION_SPACE_SIZE",
    "ALL_ACTIONS",
    "ActionSpec",
    "action_index",
    "action_spec_from_proto",
    "action_spec_to_proto",
    "canonical_action_key",
]
