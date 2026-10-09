# Owns: developer workflow entrypoints (install, quality gates, compose, migrations, protos).
# Does not own: CI provider configuration or production release automation.

ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
  PYTHON ?= .venv/bin/python
else
  PYTHON ?= python3
endif

.PHONY: install lint typecheck test migrate dev-up dev-down dev-gateway proto proto-stubs demo

PROTO_FILES := \
	proto/common.proto \
	proto/policy_service.proto \
	proto/execution_service.proto \
	proto/retrieval_service.proto \
	proto/eval_service.proto \
	proto/registry_service.proto

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check forgeai scripts tests conftest.py
	black --check forgeai scripts tests conftest.py

typecheck:
	mypy forgeai

test:
	# override demo mode so tests always exercise real paths
	FORGEAI_DEMO_MODE=false uv run pytest -q

migrate:
	uv run alembic upgrade head

dev-up:
	docker compose up -d

dev-down:
	docker compose down

dev-gateway:
	FORGEAI_DEMO_MODE=true uv run uvicorn forgeai.gateway.app:create_app --host 0.0.0.0 --port 8000 --reload

demo:
	@echo "Starting ForgeAI demo..."
	docker compose up -d
	sleep 15
	@echo "Starting gateway in demo mode..."
	@FORGEAI_DEMO_MODE=true uv run uvicorn forgeai.gateway.app:create_app --host 0.0.0.0 --port 8000 > /tmp/forgeai-demo-gateway.log 2>&1 & echo $$! > .forgeai_demo_gateway.pid
	sleep 2
	@echo "Running 3 demo prompts through /v1/chat/completions..."
	uv run python3 scripts/demo.py
	@kill `cat .forgeai_demo_gateway.pid` >/dev/null 2>&1 || true
	@rm -f .forgeai_demo_gateway.pid

proto:
	mkdir -p proto/generated
	GRPC_INCLUDE=$$($(PYTHON) -c "import grpc_tools, pathlib; print(pathlib.Path(grpc_tools.__file__).parent / '_proto')"); \
	export PATH="$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$$PATH"; \
	$(PYTHON) -m grpc_tools.protoc \
		-Iproto \
		-I"$$GRPC_INCLUDE" \
		--python_out=proto/generated \
		--grpc_python_out=proto/generated \
		--mypy_out=proto/generated \
		--mypy_grpc_out=proto/generated \
		$(PROTO_FILES)
	$(PYTHON) scripts/sanitize_proto_stubs.py
	@echo "Import stubs via forgeai.proto (see forgeai/proto/__init__.py); tests use root conftest.py."

proto-stubs:
	mkdir -p proto/generated
	GRPC_INCLUDE=$$($(PYTHON) -c "import grpc_tools, pathlib; print(pathlib.Path(grpc_tools.__file__).parent / '_proto')"); \
	export PATH="$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$$PATH"; \
	$(PYTHON) -m grpc_tools.protoc \
		-Iproto \
		-I"$$GRPC_INCLUDE" \
		--mypy_out=proto/generated \
		--mypy_grpc_out=proto/generated \
		$(PROTO_FILES)
	$(PYTHON) scripts/sanitize_proto_stubs.py
