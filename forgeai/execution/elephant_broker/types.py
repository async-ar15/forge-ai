"""Wire types for ElephantBroker I/O (Section 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroundedPrompt:
    """Prompt text after grounding pass (RAG + safety hooks in v2)."""

    text_utf8: str


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Tenant/session memory snippets for conditioning (v2 retrieval-backed)."""

    snippets_utf8: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Single tool invocation audit row (v2 persistence)."""

    tool_name: str
    inputs_json_utf8: str
    outputs_json_utf8: str


__all__ = ["GroundedPrompt", "MemoryContext", "ToolCallRecord"]
