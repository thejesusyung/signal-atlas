FROM apache/airflow:2.10.5-python3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/project/src

WORKDIR /opt/project

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    libgomp1 \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY --chown=airflow:root requirements.txt pyproject.toml README.md ./
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root dags ./dags
COPY --chown=airflow:root config ./config
COPY --chown=airflow:root alembic.ini ./
COPY --chown=airflow:root alembic ./alembic
COPY --chown=airflow:root scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.3.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

CMD ["uvicorn", "news_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
