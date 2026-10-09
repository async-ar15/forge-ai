from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tests.benchmarks.scenarios import (
    SCENARIO_ADAPTIVE,
    SCENARIO_BASELINE,
    BenchmarkScenario,
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    scenario_name: str
    n_requests: int
    concurrency: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    cache_hit_rate: float
    avg_cost_usd: float
    total_cost_usd: float
    slo_violation_rate: float
    action_distribution: dict[str, int]
    errors: int
    started_at: datetime
    duration_seconds: float


class BenchmarkRunner:
    """Synthetic benchmark harness with deterministic pseudo-traffic."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def run_benchmark(
        self,
        scenario: BenchmarkScenario,
        n_requests: int | None = None,
        concurrency: int | None = None,
    ) -> BenchmarkResult:
        started = datetime.now(UTC)
        reqs = n_requests or scenario.n_requests
        conc = concurrency or scenario.concurrency
        t0 = time.perf_counter()
        latencies: list[float] = []
        costs: list[float] = []
        cache_hits = 0
        slo_violations = 0
        actions: dict[str, int] = {}
        for i in range(reqs):
            qtype = self._sample_query_type(scenario)
            repeated = self._rng.random() < scenario.repeated_query_ratio
            latency_ms, cost_usd, action, hit = self._simulate_request(
                scenario=scenario,
                query_type=qtype,
                repeated=repeated,
            )
            latencies.append(latency_ms)
            costs.append(cost_usd)
            actions[action] = actions.get(action, 0) + 1
            if hit:
                cache_hits += 1
            if latency_ms > scenario.latency_slo_ms:
                slo_violations += 1
            _ = i
        duration = max(time.perf_counter() - t0, 0.001)
        return BenchmarkResult(
            scenario_name=scenario.name,
            n_requests=reqs,
            concurrency=conc,
            p50_latency_ms=_pct(latencies, 50),
            p95_latency_ms=_pct(latencies, 95),
            p99_latency_ms=_pct(latencies, 99),
            throughput_rps=reqs / duration,
            cache_hit_rate=cache_hits / reqs,
            avg_cost_usd=statistics.fmean(costs),
            total_cost_usd=sum(costs),
            slo_violation_rate=slo_violations / reqs,
            action_distribution=actions,
            errors=0,
            started_at=started,
            duration_seconds=duration,
        )

    def _sample_query_type(self, scenario: BenchmarkScenario) -> str:
        roll = self._rng.random()
        acc = 0.0
        for name, weight in scenario.query_type_distribution.items():
            acc += weight
            if roll <= acc:
                return name
        return "chat"

    def _simulate_request(
        self,
        *,
        scenario: BenchmarkScenario,
        query_type: str,
        repeated: bool,
    ) -> tuple[float, float, str, bool]:
        if scenario.baseline_forced_action is not None:
            base_latency = 220.0
            base_cost = 0.012
            hit = repeated and self._rng.random() < 0.15
            if hit:
                return 45.0, 0.0, scenario.baseline_forced_action, True
            return (
                base_latency + self._rng.uniform(-20, 30),
                base_cost,
                scenario.baseline_forced_action,
                False,
            )
        if scenario.name == "slo_stress":
            action = "small|int8|cache_only|short"
            hit = repeated and self._rng.random() < 0.35
            latency = 90.0 + self._rng.uniform(-15, 20)
            if not hit:
                latency += 25.0
            return latency, 0.0012 if not hit else 0.0002, action, hit
        if scenario.name == "cost_optimization":
            action = "small|int4|off|short"
            hit = repeated and self._rng.random() < 0.5
            latency = 130.0 + self._rng.uniform(-25, 30)
            cost = 0.0008 if not hit else 0.0
            return latency, cost, action, hit
        action = (
            "medium|int8|cache_only|medium"
            if query_type != "code"
            else "large|int8|full|medium"
        )
        hit_prob = 0.8 if scenario.name == "cache_heavy" else 0.35
        hit = repeated and self._rng.random() < hit_prob
        latency = 150.0 + self._rng.uniform(-30, 35)
        cost = 0.0035
        if hit:
            latency *= 0.45
            cost = 0.0
        return latency, cost, action, hit


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    k = int(round((pct / 100.0) * (len(values) - 1)))
    return sorted(values)[k]


def _results_dir() -> Path:
    out = Path("tests/benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_result_json(result: BenchmarkResult) -> Path:
    ts = result.started_at.strftime("%Y%m%dT%H%M%SZ")
    path = _results_dir() / f"{result.scenario_name}_{ts}.json"
    payload = asdict(result)
    payload["started_at"] = result.started_at.isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _print_table(baseline: BenchmarkResult, others: list[BenchmarkResult]) -> None:
    print("| Scenario | p50 (ms) | p95 (ms) | Cost/req | Cache hit | vs Baseline |")
    print("|---|---|---|---|---|---|")
    print(
        f"| Baseline | {baseline.p50_latency_ms:.1f} | {baseline.p95_latency_ms:.1f} | "
        f"${baseline.avg_cost_usd:.4f} | {baseline.cache_hit_rate*100:.1f}% | — |"
    )
    for r in others:
        cost_delta = (
            (1.0 - (r.avg_cost_usd / baseline.avg_cost_usd)) * 100.0
            if baseline.avg_cost_usd
            else 0.0
        )
        p95_delta = (
            (1.0 - (r.p95_latency_ms / baseline.p95_latency_ms)) * 100.0
            if baseline.p95_latency_ms
            else 0.0
        )
        print(
            f"| {r.scenario_name} | {r.p50_latency_ms:.1f} | {r.p95_latency_ms:.1f} | "
            f"${r.avg_cost_usd:.4f} | {r.cache_hit_rate*100:.1f}% | "
            f"{cost_delta:.1f}% cost, {p95_delta:.1f}% p95 |"
        )


def main() -> None:
    runner = BenchmarkRunner(seed=7)
    baseline = runner.run_benchmark(SCENARIO_BASELINE)
    adaptive = runner.run_benchmark(SCENARIO_ADAPTIVE)
    _write_result_json(baseline)
    _write_result_json(adaptive)
    _print_table(baseline, [adaptive])


if __name__ == "__main__":
    main()


__all__ = ["BenchmarkResult", "BenchmarkRunner", "main"]
