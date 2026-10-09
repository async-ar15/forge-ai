# Contributing

## Run The Stack Locally

```bash
cp .env.example .env
make dev-up
make migrate
```

## Run Tests

```bash
uv run pytest -q
```

## Add A New Model Tier

1. Add the model tier enum/value mapping in `forgeai/enums.py` and `forgeai/policy/bandit_actions.py`.
2. Add execution loading/runtime support in `forgeai/execution/`.
3. Add benchmark scenarios in `tests/benchmarks/scenarios.py`.

## Pull Request Requirements

```bash
make lint
make typecheck
uv run pytest -q
```

PRs must pass lint, typecheck, and tests before review.
