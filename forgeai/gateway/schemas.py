"""Pydantic models for the public HTTP surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from forgeai.gateway.constants import (
    QUERY_MAX_CHARS,
    TOKEN_BUDGET_DEFAULT,
    TOKEN_BUDGET_MAX_INCLUSIVE,
    TOKEN_BUDGET_MIN_INCLUSIVE,
)

QueryTypeLiteral = Literal["rag", "code", "chat", "summarize", "unknown"]


class InferRequest(BaseModel):
    """POST /v1/infer JSON body; query_type is client-supplied only (never inferred)."""

    query: str = Field(..., max_length=QUERY_MAX_CHARS)
    query_type: QueryTypeLiteral = "unknown"
    token_budget: int = Field(default=TOKEN_BUDGET_DEFAULT)
    latency_slo_ms: int | None = None
    session_id: str | None = None
    stream: bool = True

    @field_validator("token_budget")
    @classmethod
    def _token_budget_bounds(cls, v: int) -> int:
        if v < TOKEN_BUDGET_MIN_INCLUSIVE or v > TOKEN_BUDGET_MAX_INCLUSIVE:
            raise ValueError("token_budget out of bounds")
        return v

    @field_validator("session_id")
    @classmethod
    def _session_uuid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        from uuid import UUID

        UUID(v)
        return v


OpenAIRoleLiteral = Literal["system", "user", "assistant"]


class OpenAIChatMessage(BaseModel):
    role: OpenAIRoleLiteral
    content: str


class OpenAIChatCompletionsRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[OpenAIChatMessage]) -> list[OpenAIChatMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


__all__ = [
    "InferRequest",
    "OpenAIChatCompletionsRequest",
    "OpenAIChatMessage",
    "QueryTypeLiteral",
]
