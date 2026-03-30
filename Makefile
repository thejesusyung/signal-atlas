PYTHON ?= python3

.PHONY: setup up down test lint db-migrate db-upgrade db-revision

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

up:
	docker compose up --build -d

down:
	docker compose down --remove-orphans

test:
	pytest

lint:
	ruff check src tests dags

db-migrate:
	alembic revision --autogenerate -m "$(message)"

db-upgrade:
	alembic upgrade head

db-revision:
	alembic revision -m "$(message)"

