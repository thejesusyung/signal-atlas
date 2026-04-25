PYTHON ?= python3
AWS_REGION ?= us-east-2
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text)
ECR_BASE = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

.PHONY: setup up down up-dev down-dev up-prod down-prod up-airflow down-airflow \
        build-api build-pipeline push-ecr \
        run-ingestion-local run-extraction-local run-simulation-local \
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

# AWS-native build & deploy targets
build-api:
	docker build -f Dockerfile.api -t pet-signal-atlas/api:latest .

build-pipeline:
	docker build -f Dockerfile.pipeline -t pet-signal-atlas/pipeline:latest .

push-ecr: build-api build-pipeline
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_BASE)
	docker tag pet-signal-atlas/api:latest $(ECR_BASE)/pet-signal-atlas/api:latest
	docker tag pet-signal-atlas/pipeline:latest $(ECR_BASE)/pet-signal-atlas/pipeline:latest
	docker push $(ECR_BASE)/pet-signal-atlas/api:latest
	docker push $(ECR_BASE)/pet-signal-atlas/pipeline:latest

# Run pipeline tasks locally against real RDS (requires .env.aws)
run-ingestion-local:
	docker run --rm --env-file .env.aws pet-signal-atlas/pipeline:latest \
	    python -m news_pipeline.pipelines.ingestion

run-extraction-local:
	docker run --rm --env-file .env.aws pet-signal-atlas/pipeline:latest \
	    python -m news_pipeline.pipelines.extraction

run-simulation-local:
	docker run --rm --env-file .env.aws pet-signal-atlas/pipeline:latest \
	    python -m news_pipeline.pipelines.simulation

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
