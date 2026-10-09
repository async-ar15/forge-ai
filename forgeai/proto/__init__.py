"""gRPC/protobuf stubs as ``forgeai.proto.<module>``.

Run ``make proto`` to emit into ``proto/generated/``.
This package prepends that directory to ``sys.path`` and registers module aliases under
``forgeai.proto`` (e.g. ``from forgeai.proto import policy_service_pb2``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATED_DIR = _REPO_ROOT / "proto" / "generated"
_gen_path = str(_GENERATED_DIR)
if _gen_path not in sys.path:
    sys.path.insert(0, _gen_path)

_EXPORTS: tuple[str, ...] = (
    "common_pb2",
    "common_pb2_grpc",
    "policy_service_pb2",
    "policy_service_pb2_grpc",
    "execution_service_pb2",
    "execution_service_pb2_grpc",
    "retrieval_service_pb2",
    "retrieval_service_pb2_grpc",
    "eval_service_pb2",
    "eval_service_pb2_grpc",
    "registry_service_pb2",
    "registry_service_pb2_grpc",
)

__all__ = list(_EXPORTS)

for _name in _EXPORTS:
    _mod = importlib.import_module(_name)
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod
