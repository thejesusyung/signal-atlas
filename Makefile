PYTHON ?= python3

.PHONY: setup up down up-dev down-dev up-prod down-prod up-airflow down-airflow \
        test lint db-migrate db-upgrade db-revision

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Local development (full stack: postgres, airflow, mlflow, api)
up-dev:
	docker compose -f docker-compose.dev.yml up --build -d

down-dev:
	docker compose -f docker-compose.dev.yml down --remove-orphans

# Aliases for convenience
up: up-dev
down: down-dev

# Production 2GB EC2 (API + MLflow only, postgres on RDS)
up-prod:
	docker compose -f docker-compose.prod.yml up --build -d

down-prod:
	docker compose -f docker-compose.prod.yml down --remove-orphans

# On-demand Airflow EC2 (Airflow only, postgres on RDS, mlflow on 2GB EC2)
up-airflow:
	docker compose -f docker-compose.airflow.yml up --build -d

down-airflow:
	docker compose -f docker-compose.airflow.yml down --remove-orphans

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
