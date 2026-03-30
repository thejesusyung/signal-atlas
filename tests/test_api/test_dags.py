from __future__ import annotations

from pathlib import Path

import pytest


def _prepare_airflow_env(monkeypatch, tmp_path) -> None:
    airflow_home = tmp_path / "airflow-home"
    logs = airflow_home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIRFLOW_HOME", str(airflow_home))
    monkeypatch.setenv("AIRFLOW__LOGGING__BASE_LOG_FOLDER", str(logs))
    monkeypatch.setenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", f"sqlite:///{airflow_home / 'airflow.db'}")
    monkeypatch.setenv(
        "AIRFLOW__CORE__DAGS_FOLDER",
        str(Path(__file__).resolve().parents[2] / "dags"),
    )
    monkeypatch.setenv("AIRFLOW__CORE__LOAD_EXAMPLES", "False")


def test_ingestion_dag_imports(monkeypatch, tmp_path):
    _prepare_airflow_env(monkeypatch, tmp_path)
    pytest.importorskip("airflow")
    from dags.ingestion_dag import ingestion_dag

    assert ingestion_dag.dag_id == "ingestion_dag"
    assert "fetch_candidates" in ingestion_dag.task_ids


def test_extraction_dag_imports(monkeypatch, tmp_path):
    _prepare_airflow_env(monkeypatch, tmp_path)
    pytest.importorskip("airflow")
    from dags.extraction_dag import extraction_dag

    assert extraction_dag.dag_id == "extraction_dag"
    assert "fetch_pending_article_ids" in extraction_dag.task_ids
