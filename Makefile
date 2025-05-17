.PHONY: install dev test lint format clean docker-build

install:
	python -m pip install -e .

dev:
	python -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff src

format:
	ruff format src

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist

docker-build:
	docker build -t rnf-experiments .
