"""One loaded vLLM ``AsyncLLMEngine`` shard with warmup and streaming helpers.

The ``vllm`` package is imported only inside ``create``, ``_warmup``, and
``stream_text`` — never at module import time — so environments without vLLM
can still load other ``forgeai.execution`` modules.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from forgeai.enums import Precision
from forgeai.execution.quantization_config import (
    QuantizationConfig,
    QuantizationMethodKind,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)


def _memory_mb_from_engine(engine: object) -> float:
    """Best-effort RSS from vLLM internals; 0.0 when the runtime omits hooks."""

    llm = getattr(engine, "engine", engine)
    ex = getattr(llm, "model_executor", None)
    if ex is None:
        return 0.0
    fn = getattr(ex, "get_memory_usage_gb", None)
    if callable(fn):
        try:
            return float(fn()) * 1024.0
        except Exception:
            return 0.0
    return 0.0


def _queue_depth_from_engine(engine: object) -> int:
    """Prefer unfinished request count; fall back to scheduler waiting queue."""

    llm = getattr(engine, "engine", engine)
    fn = getattr(llm, "get_num_unfinished_requests", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0
    sched = getattr(llm, "scheduler", None)
    if sched is None:
        return 0
    waiting = getattr(sched, "waiting", None)
    if waiting is None:
        return 0
    try:
        return len(waiting)
    except TypeError:
        return 0


def _quant_kind_and_bits(precision: Precision) -> tuple[QuantizationMethodKind, int]:
    if precision is Precision.FP16:
        return QuantizationMethodKind.NONE, 16
    if precision is Precision.INT8:
        return QuantizationMethodKind.BITSANDBYTES_INT8, 8
    if precision is Precision.INT4:
        return QuantizationMethodKind.AWQ_INT4, 4
    return QuantizationMethodKind.NONE, 16


def _vllm_extra_kwargs(precision: Precision) -> dict[str, Any]:
    if precision is Precision.INT8:
        return {"quantization": "bitsandbytes", "load_format": "bitsandbytes"}
    if precision is Precision.INT4:
        return {"quantization": "awq"}
    return {}


async def _warmup(engine: Any) -> None:  # noqa: ANN401
    from vllm import SamplingParams

    warm_id = f"warmup-{uuid.uuid4()}"
    warm_params = SamplingParams(max_tokens=1, temperature=0.0)
    gen = engine.generate("forgeai", warm_params, warm_id)
    async for _ in gen:
        break


@dataclass(slots=True)
class VllmEngineShard:
    """Thin async wrapper so tests can substitute a compatible double."""

    _engine: Any
    _quant: QuantizationConfig

    @classmethod
    async def create(
        cls,
        artifact_dir: Path,
        precision: Precision,
        *,
        model_version_id: str,
    ) -> VllmEngineShard:
        """Load weights asynchronously; safe to await concurrently per distinct path."""

        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError as exc:
            msg = "vllm is not installed — execution requires vLLM in runtime images."
            raise RuntimeError(msg) from exc

        model_path = str(artifact_dir)
        extra = _vllm_extra_kwargs(precision)
        method, bits = _quant_kind_and_bits(precision)
        args = AsyncEngineArgs(model=model_path, **extra)
        engine = AsyncLLMEngine.from_engine_args(args)
        loaded_at = datetime.now(UTC)
        await _warmup(engine)
        mem = _memory_mb_from_engine(engine)
        qconf = QuantizationConfig(
            method=method,
            bits=bits,
            loaded_at_utc=loaded_at,
            memory_mb=mem,
        )
        _LOG.info(
            "vllm_engine_loaded version=%s precision=%s memory_mb=%.2f",
            model_version_id,
            precision.value,
            mem,
        )
        return cls(_engine=engine, _quant=qconf)

    async def shutdown(self) -> None:
        """Release GPU memory held by this shard."""

        fn = getattr(self._engine, "shutdown", None)
        if callable(fn):
            out = fn()
            if inspect.isawaitable(out):
                await out

    def quantization_config(self) -> QuantizationConfig:
        """Return the post-warmup quantization snapshot."""

        return self._quant

    def queue_depth(self) -> int:
        """Best-effort depth for policy probes."""

        return _queue_depth_from_engine(self._engine)

    async def stream_text(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Yield decoder text fragments for one completion."""

        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        gen = self._engine.generate(prompt, sp, request_id)
        async for ro in gen:
            for out in ro.outputs:
                if out.text:
                    yield out.text


__all__ = ["VllmEngineShard"]
