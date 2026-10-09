from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from openai import OpenAI

BASELINE_COST = Decimal("0.0120")


@dataclass(frozen=True, slots=True)
class DemoResult:
    prompt: str
    response_head: str
    model: str
    precision: str
    cost: Decimal
    latency_ms: int


def main() -> None:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="demo-key")
    prompts: list[tuple[str, int | None]] = [
        ("What is 2+2?", 20),
        ("Write a Python hello world function", None),
        ("What is RAG in one sentence?", None),
    ]
    results = [
        _run_prompt(client, prompt=prompt, max_tokens=max_tokens)
        for prompt, max_tokens in prompts
    ]
    for item in results:
        _print_result(item)
    _print_summary(results)


def _run_prompt(client: OpenAI, prompt: str, max_tokens: int | None = None) -> DemoResult:
    raw = client.chat.completions.with_raw_response.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
    )
    parsed = raw.parse()
    text = parsed.choices[0].message.content or ""
    headers = raw.headers
    return DemoResult(
        prompt=prompt,
        response_head=text[:100],
        model=headers.get("x-forgeai-model", "unknown"),
        precision=headers.get("x-forgeai-precision", "unknown"),
        cost=Decimal(headers.get("x-forgeai-cost", "0.000000")),
        latency_ms=int(headers.get("x-forgeai-latency", "0")),
    )


def _print_result(item: DemoResult) -> None:
    print(f'Prompt: "{item.prompt}"')
    print(f'Response: "{item.response_head}"')
    print(f"Routed to: {item.model} / {item.precision}")
    print(f"Cost: ${item.cost:.6f}")
    print(f"Latency: {item.latency_ms}ms")
    print("─────────────────────────────")


def _print_summary(results: list[DemoResult]) -> None:
    total = sum((r.cost for r in results), Decimal("0"))
    baseline = BASELINE_COST * Decimal(len(results))
    savings = Decimal("0")
    if baseline > 0:
        savings = (baseline - total) / baseline * Decimal("100")
    print(f"Total cost: ${total:.6f}")
    print(f"vs baseline (always large/fp16): ${baseline:.6f}")
    print(f"Savings: {savings:.2f}%")


if __name__ == "__main__":
    main()
