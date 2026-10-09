#!/usr/bin/env python3
"""Post-process mypy-protobuf ``.pyi`` output for strict-clean stubs.

``mypy-protobuf`` may emit ``# noqa: Y015`` and ``# type: ignore``; this script removes
them and widens gRPC ``*Stub`` RPC types so async subclasses stay Liskov-safe.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATED = _REPO_ROOT / "proto" / "generated"

_SERVICER_CTX_PATTERN = re.compile(
    r"class _ServicerContext\(_grpc\.ServicerContext, _aio\.ServicerContext\):"
    r"(?: # type: ignore\[misc, type-arg\])?\n    \.\.\.\n",
    re.MULTILINE,
)
_SERVICER_CTX_REPL = (
    "class _ServicerContext(_typing.Protocol):\n"
    '    """Structural type accepted by generated servicers (sync or aio context)."""\n'
    "    ...\n"
)

_GRPC_MULTICALLABLE = re.compile(
    r"^    (?P<name>[A-Za-z0-9_]+): "
    r"(?P<grpc>_grpc\.(?P<kind>(?:UnaryUnary|UnaryStream|StreamUnary|StreamStream))"
    r"MultiCallable)\[(?P<args>[^\]]+)\]\s*$"
)

_NOQA = re.compile(r"\s+# noqa:.*$")
_TYPE_IGNORE_ASSIGNMENT = re.compile(r"\s+# type: ignore\[assignment]\s*$")
_TYPE_IGNORE_MISC = re.compile(r"\s+# type: ignore\[misc, type-arg]\s*$")


def _strip_trailing_suppressions(line: str) -> str:
    has_nl = line.endswith("\n")
    body = line[:-1] if has_nl else line
    body = _NOQA.sub("", body)
    body = _TYPE_IGNORE_ASSIGNMENT.sub("", body)
    body = _TYPE_IGNORE_MISC.sub("", body)
    body = body.rstrip()
    return body + ("\n" if has_nl else "")


def _widen_sync_stub_multicallables(lines: list[str]) -> list[str]:
    """Inside each non-async ``*Stub`` class body, union grpc and aio mult callables."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^class (.+Stub):\s*$", line)
        if m and not m.group(1).endswith("AsyncStub"):
            out.append(line)
            i += 1
            while i < len(lines):
                inner = lines[i]
                if re.match(r"^class .+AsyncStub", inner) or inner.strip().startswith(
                    "@_typing.type_check_only"
                ):
                    break
                mm = _GRPC_MULTICALLABLE.match(inner.rstrip("\n"))
                if mm:
                    name = mm.group("name")
                    grpc = mm.group("grpc")
                    kind = mm.group("kind")
                    args = mm.group("args")
                    aio = f"_aio.{kind}MultiCallable[{args}]"
                    out.append(f"    {name}: {grpc}[{args}] | {aio}\n")
                else:
                    out.append(inner)
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def sanitize_pyi(text: str, *, is_grpc: bool) -> str:
    text = _SERVICER_CTX_PATTERN.sub(_SERVICER_CTX_REPL, text)
    lines = text.splitlines(keepends=True)
    if is_grpc:
        lines = _widen_sync_stub_multicallables(lines)
    cleaned = [_strip_trailing_suppressions(L) for L in lines]
    return "".join(cleaned)


def main() -> int:
    if not _GENERATED.is_dir():
        print(f"error: missing {_GENERATED}", file=sys.stderr)
        return 1
    for path in sorted(_GENERATED.glob("*.pyi")):
        raw = path.read_text(encoding="utf-8")
        fixed = sanitize_pyi(raw, is_grpc=path.name.endswith("_grpc.pyi"))
        path.write_text(fixed, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
