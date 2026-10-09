"""Execution engine: vLLM shards, registry resolution, streaming gRPC (Sections 2/5).

Does not own: Policy Engine scoring, Retrieval Engine fan-out, or gateway HTTP.
"""

from __future__ import annotations

from forgeai.execution.constants import (
    COST_PER_OUTPUT_TOKEN_MICRODOLLARS,
    VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS,
    VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS,
    VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS,
    ExecutionConfigKey,
    cost_per_output_token_microdollars,
    read_output_token_cap,
)
from forgeai.execution.elephant_broker import (
    ElephantBrokerClient,
    GroundedPrompt,
    MemoryContext,
)
from forgeai.execution.engine_pool import EnginePool
from forgeai.execution.exceptions import (
    PrecisionUnavailableError,
    QuantizationMismatchError,
)
from forgeai.execution.gpu_probe import ProductionGpuLoadProbe
from forgeai.execution.grpc_servicer import ForgeExecutionServicer
from forgeai.execution.handles import ModelHandle
from forgeai.execution.model_loader import ModelLoader
from forgeai.execution.quantization_config import (
    QuantizationConfig,
    QuantizationMethodKind,
)
from forgeai.execution.queue_depth_probe import ProductionQueueDepthProbe
from forgeai.execution.runtime import ExecutionRuntime, ExecutionSpec
from forgeai.execution.vllm_shard import VllmEngineShard

__all__ = [
    "COST_PER_OUTPUT_TOKEN_MICRODOLLARS",
    "ElephantBrokerClient",
    "EnginePool",
    "ExecutionConfigKey",
    "ExecutionRuntime",
    "ExecutionSpec",
    "ForgeExecutionServicer",
    "GroundedPrompt",
    "MemoryContext",
    "ModelHandle",
    "ModelLoader",
    "PrecisionUnavailableError",
    "ProductionGpuLoadProbe",
    "ProductionQueueDepthProbe",
    "QuantizationConfig",
    "QuantizationMismatchError",
    "QuantizationMethodKind",
    "VLLM_OUTPUT_BUDGET_LONG_MAX_TOKENS",
    "VLLM_OUTPUT_BUDGET_MEDIUM_MAX_TOKENS",
    "VLLM_OUTPUT_BUDGET_SHORT_MAX_TOKENS",
    "VllmEngineShard",
    "cost_per_output_token_microdollars",
    "read_output_token_cap",
]
