.PHONY: help install setup download-model run test clean format lint

help:
	@echo "WoofWatch - Makefile commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install dependencies"
	@echo "  make setup           Full setup (venv + install + model)"
	@echo "  make download-model  Download YOLOv8 model"
	@echo ""
	@echo "Running:"
	@echo "  make run             Run CLI with default settings"
	@echo "  make run-headless    Run CLI in headless mode"
	@echo ""
	@echo "Development:"
	@echo "  make test            Run tests"
	@echo "  make test-cov        Run tests with coverage"
	@echo "  make format          Format code with black"
	@echo "  make lint            Run ruff linter"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove cache and temporary files"

install:
	pip install -r requirements.txt

setup:
	python3 -m venv venv
	@echo "Run: source venv/bin/activate"
	@echo "Then run: make install && make download-model"

download-model:
	python scripts/download_model.py

run:
	python scripts/cli_runner.py

run-headless:
	python scripts/cli_runner.py --no-display

test:
	pytest tests/ -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term tests/

format:
	black src/ tests/ scripts/

lint:
	ruff check src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
