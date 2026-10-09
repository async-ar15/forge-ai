"""OpenAI-compatible gateway constants (field names, objects, and headers)."""

from __future__ import annotations

from typing import Final

OPENAI_OBJECT_CHAT_COMPLETION: Final[str] = "chat.completion"
OPENAI_OBJECT_CHAT_COMPLETION_CHUNK: Final[str] = "chat.completion.chunk"
OPENAI_OBJECT_LIST: Final[str] = "list"
OPENAI_OBJECT_MODEL: Final[str] = "model"
OPENAI_ROLE_SYSTEM: Final[str] = "system"
OPENAI_ROLE_USER: Final[str] = "user"
OPENAI_ROLE_ASSISTANT: Final[str] = "assistant"
OPENAI_DONE_TOKEN: Final[str] = "[DONE]"
OPENAI_MODEL_ROUTED: Final[str] = "forgeai-routed"

OPENAI_HEADER_MODEL: Final[str] = "x-forgeai-model"
OPENAI_HEADER_PRECISION: Final[str] = "x-forgeai-precision"
OPENAI_HEADER_RETRIEVAL: Final[str] = "x-forgeai-retrieval"
OPENAI_HEADER_COST: Final[str] = "x-forgeai-cost"
OPENAI_HEADER_LATENCY: Final[str] = "x-forgeai-latency"
OPENAI_HEADER_POLICY_VERSION: Final[str] = "x-forgeai-policy-version"
OPENAI_HEADER_REQUEST_ID: Final[str] = "x-forgeai-request-id"

OPENAI_ENDPOINT_INFER: Final[str] = "/v1/infer"
OPENAI_ENDPOINT_CHAT_COMPLETIONS: Final[str] = "/v1/chat/completions"
