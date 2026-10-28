PYTHON ?= python

.PHONY: gate lint test check build

gate: lint check test

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

check:
	$(PYTHON) scripts/check_repository.py

test:
	JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $(PYTHON) -m pytest tests/ -q

build:
	$(PYTHON) -m build
