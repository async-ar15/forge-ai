"""Pure translation helpers for OpenAI-compatible gateway endpoints."""

from __future__ import annotations

import re
from typing import Final

from forgeai.gateway.openai_constants import (
    OPENAI_ROLE_ASSISTANT,
    OPENAI_ROLE_SYSTEM,
    OPENAI_ROLE_USER,
)
from forgeai.gateway.schemas import OpenAIChatMessage, QueryTypeLiteral

_CODE_BLOCK_RE: Final[re.Pattern[str]] = re.compile(r"```[\s\S]*?```")


def infer_query_type_heuristic(messages: list[OpenAIChatMessage]) -> QueryTypeLiteral:
    """Infer query_type with a message-structure heuristic.

    This is not semantic classification; a classifier is a v2 feature.
    """

    if _contains_code_signal(messages):
        return "code"
    if len(messages) > 3:
        return "chat"
    return "unknown"


def messages_to_chatml_prompt(messages: list[OpenAIChatMessage]) -> str:
    chunks: list[str] = []
    for msg in messages:
        chunks.append(_chatml_segment(msg.role, msg.content))
    chunks.append(f"<|im_start|>{OPENAI_ROLE_ASSISTANT}\n")
    return "".join(chunks)


def last_user_message(messages: list[OpenAIChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == OPENAI_ROLE_USER:
            return msg.content
    return messages[-1].content


def estimate_token_count(text: str) -> int:
    parts = [p for p in text.strip().split() if p]
    return len(parts)


def split_stream_deltas(text: str) -> list[str]:
    if text == "":
        return []
    words = text.split(" ")
    out: list[str] = []
    for idx, word in enumerate(words):
        suffix = " " if idx < len(words) - 1 else ""
        out.append(f"{word}{suffix}")
    return out


def _contains_code_signal(messages: list[OpenAIChatMessage]) -> bool:
    if not messages:
        return False
    system_text = " ".join(
        msg.content.lower() for msg in messages if msg.role == OPENAI_ROLE_SYSTEM
    )
    if "code" in system_text:
        return True
    user = last_user_message(messages)
    return _CODE_BLOCK_RE.search(user) is not None


def _chatml_segment(role: str, content: str) -> str:
    safe_role = (
        role
        if role in {OPENAI_ROLE_SYSTEM, OPENAI_ROLE_USER, OPENAI_ROLE_ASSISTANT}
        else OPENAI_ROLE_USER
    )
    return f"<|im_start|>{safe_role}\n{content}<|im_end|>\n"
