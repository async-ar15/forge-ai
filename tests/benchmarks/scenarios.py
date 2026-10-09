from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    name: str
    n_requests: int
    concurrency: int
    query_type_distribution: dict[str, float]
    repeated_query_ratio: float
    latency_slo_ms: int
    baseline_forced_action: str | None = None


SCENARIO_BASELINE = BenchmarkScenario(
    name="baseline",
    n_requests=500,
    concurrency=20,
    query_type_distribution={"chat": 0.4, "rag": 0.3, "code": 0.2, "summarize": 0.1},
    repeated_query_ratio=0.0,
    latency_slo_ms=1000,
    baseline_forced_action="large|fp16|full|medium",
)

SCENARIO_ADAPTIVE = BenchmarkScenario(
    name="adaptive_routing",
    n_requests=1000,
    concurrency=20,
    query_type_distribution={"chat": 0.4, "rag": 0.3, "code": 0.2, "summarize": 0.1},
    repeated_query_ratio=0.2,
    latency_slo_ms=1000,
)

SCENARIO_CACHE_HEAVY = BenchmarkScenario(
    name="cache_heavy",
    n_requests=500,
    concurrency=20,
    query_type_distribution={"rag": 0.5, "chat": 0.3, "code": 0.1, "summarize": 0.1},
    repeated_query_ratio=0.8,
    latency_slo_ms=1000,
)

SCENARIO_COST_OPT = BenchmarkScenario(
    name="cost_optimization",
    n_requests=200,
    concurrency=10,
    query_type_distribution={"chat": 0.5, "rag": 0.2, "code": 0.2, "summarize": 0.1},
    repeated_query_ratio=0.4,
    latency_slo_ms=2000,
)

SCENARIO_SLO_STRESS = BenchmarkScenario(
    name="slo_stress",
    n_requests=200,
    concurrency=10,
    query_type_distribution={"chat": 0.5, "rag": 0.2, "code": 0.2, "summarize": 0.1},
    repeated_query_ratio=0.1,
    latency_slo_ms=100,
)

ALL_SCENARIOS = [
    SCENARIO_BASELINE,
    SCENARIO_ADAPTIVE,
    SCENARIO_CACHE_HEAVY,
    SCENARIO_COST_OPT,
    SCENARIO_SLO_STRESS,
]

__all__ = [
    "ALL_SCENARIOS",
    "BenchmarkScenario",
    "SCENARIO_ADAPTIVE",
    "SCENARIO_BASELINE",
    "SCENARIO_CACHE_HEAVY",
    "SCENARIO_COST_OPT",
    "SCENARIO_SLO_STRESS",
]
