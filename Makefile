UV ?= $(shell command -v uv || echo ~/.local/bin/uv)
DETECTOR := $(UV) run detector

.PHONY: help install clean run start test lint format typecheck check smoke-test build macos

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Core Targets:"
	@echo "  install           - Install dependencies via uv"
	@echo "  run / start       - Run the detector in foreground"
	@echo ""
	@echo "Development & Quality:"
	@echo "  lint              - Run ruff for linting"
	@echo "  format            - Run ruff for formatting"
	@echo "  typecheck         - Run pyright for type checking"
	@echo "  test              - Run pytest suite"
	@echo "  check             - Run lint, typecheck, and test"
	@echo "  smoke-test        - Run ZMQ connection smoke test"
	@echo ""
	@echo "  build             - Build the macOS App"
	@echo "  clean             - Remove all build artifacts, virtualenv, and caches"

install:
	$(UV) sync --all-groups

run start:
	$(DETECTOR) start


test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

typecheck:
	$(UV) run pyright

check: 
	lint typecheck test

smoke-test:
	$(UV) run scripts/zmq_smoke_test.py

build:
	./macos/build/build_app.sh

clean:
	rm -rf .venv build/ .pytest_cache .ruff_cache macos/swift/.build macos/swift/.swiftpm
	find . -type d -name "__pycache__" -exec rm -rf {} +