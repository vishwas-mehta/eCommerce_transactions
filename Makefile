.PHONY: install test lint format clean run-eda run-clustering run-lookalike all help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Remove generated files"
	@echo "  make run-eda       - Run EDA analysis"
	@echo "  make run-clustering- Run customer clustering"
	@echo "  make run-lookalike - Run lookalike model"
	@echo "  make all           - Run full analysis pipeline"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev,notebook]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

# Analysis pipelines
run-eda:
	python -c "from src.eda import run_full_eda; run_full_eda()"

run-clustering:
	python -c "from src.clustering import main; main()"

run-lookalike:
	python -c "from src.lookalike import main; main()"

all: run-eda run-clustering run-lookalike
	@echo "Full analysis pipeline completed!"
