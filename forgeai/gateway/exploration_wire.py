"""Map policy exploration enum ints to persisted wire strings."""

from __future__ import annotations

from typing import Final

from forgeai.enums import ExplorationType
from forgeai.proto import common_pb2

_PROTO_TO_WIRE: Final[dict[int, str]] = {
    int(common_pb2.EXPLORATION_TYPE_NONE): ExplorationType.NONE.value,
    int(common_pb2.EXPLORATION_TYPE_EPSILON): ExplorationType.EPSILON.value,
    int(common_pb2.EXPLORATION_TYPE_UCB): ExplorationType.UCB.value,
    int(common_pb2.EXPLORATION_TYPE_FORCED): ExplorationType.FORCED.value,
}


def exploration_type_int_to_wire(value: int) -> str:
    """Best-effort wire string; unknown ints map to ``none``."""

    return _PROTO_TO_WIRE.get(value, ExplorationType.NONE.value)


__all__ = ["exploration_type_int_to_wire"]
