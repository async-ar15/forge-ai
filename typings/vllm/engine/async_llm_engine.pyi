"""Stub for ``vllm.engine.async_llm_engine``."""

from typing import Any

class AsyncLLMEngine:
    @staticmethod
    def from_engine_args(args: Any) -> AsyncLLMEngine: ...
    def generate(self, *args: Any, **kwargs: Any) -> Any: ...
