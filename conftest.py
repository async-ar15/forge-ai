"""Pytest bootstrap: put ``proto/generated`` on ``sys.path`` for protobuf modules."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_PROTO_GENERATED = _REPO_ROOT / "proto" / "generated"
_p = str(_PROTO_GENERATED)
if _p not in sys.path:
    sys.path.insert(0, _p)
