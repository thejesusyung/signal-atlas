#!/usr/bin/env bash
# scripts/db_migrate.sh
#
# Migrates all three databases from the current postgres container to RDS.
# Run this FROM the current (6GB) server while it is still live.
#
# Usage:
#   export PGPASSWORD_RDS=<rds-master-password>
#   ./scripts/db_migrate.sh <rds-endpoint> [rds-port] [rds-master-user]
#
# Prerequisites on this machine:
#   - postgresql-client (pg_dump, pg_restore, psql) installed
#   - Docker running with the current postgres container up
#   - ~500MB free in /tmp for dump files
#
# RDS must be prepped before running this script:
#   CREATE DATABASE news_pipeline;
#   CREATE DATABASE airflow_metadata;
#   CREATE DATABASE mlflow_tracking;
#   CREATE USER news_pipeline WITH PASSWORD '...';
#   GRANT ALL PRIVILEGES ON DATABASE news_pipeline TO news_pipeline;
#   GRANT ALL PRIVILEGES ON DATABASE airflow_metadata TO news_pipeline;
#   GRANT ALL PRIVILEGES ON DATABASE mlflow_tracking TO news_pipeline;
#   \c news_pipeline
#   CREATE EXTENSION IF NOT EXISTS vector;   -- MUST happen before restore
#   GRANT ALL ON SCHEMA public TO news_pipeline;
#   \c airflow_metadata
#   GRANT ALL ON SCHEMA public TO news_pipeline;
#   \c mlflow_tracking
#   GRANT ALL ON SCHEMA public TO news_pipeline;

set -euo pipefail

RDS_HOST="${1:?Usage: $0 <rds-endpoint> [port] [master-user]}"
RDS_PORT="${2:-5432}"
RDS_MASTER_USER="${3:-postgres}"

LOCAL_USER="news_pipeline"
LOCAL_PASS="news_pipeline"
LOCAL_HOST="127.0.0.1"
LOCAL_PORT="5432"

DUMP_DIR="/tmp/pg_dumps_$(date +%Y%m%d_%H%M%S)"
DATABASES=("news_pipeline" "airflow_metadata" "mlflow_tracking")

: "${PGPASSWORD_RDS:?Set PGPASSWORD_RDS to the RDS master password before running}"

mkdir -p "${DUMP_DIR}"
echo "Dump directory: ${DUMP_DIR}"

# ---- STEP 1: Dump from current local postgres --------------------------------
echo ""
echo "=== STEP 1: Dumping databases from local postgres ==="
for DB in "${DATABASES[@]}"; do
  echo -n "  Dumping ${DB}... "
  PGPASSWORD="${LOCAL_PASS}" pg_dump \
    -h "${LOCAL_HOST}" \
    -p "${LOCAL_PORT}" \
    -U "${LOCAL_USER}" \
    -Fc \
    --no-owner \
    --no-privileges \
    "${DB}" \
    > "${DUMP_DIR}/${DB}.dump"
  echo "done ($(du -sh "${DUMP_DIR}/${DB}.dump" | cut -f1))"
done

# ---- STEP 2: Restore to RDS --------------------------------------------------
echo ""
echo "=== STEP 2: Restoring to RDS (${RDS_HOST}:${RDS_PORT}) ==="
for DB in "${DATABASES[@]}"; do
  echo -n "  Restoring ${DB}... "
  PGPASSWORD="${PGPASSWORD_RDS}" pg_restore \
    -h "${RDS_HOST}" \
    -p "${RDS_PORT}" \
    -U "${RDS_MASTER_USER}" \
    -d "${DB}" \
    --no-owner \
    --no-privileges \
    --exit-on-error \
    "${DUMP_DIR}/${DB}.dump"
  echo "done"
done

# ---- STEP 3: Grant permissions to app user -----------------------------------
echo ""
echo "=== STEP 3: Granting permissions to news_pipeline user ==="
for DB in "${DATABASES[@]}"; do
  echo -n "  Granting on ${DB}... "
  PGPASSWORD="${PGPASSWORD_RDS}" psql \
    -h "${RDS_HOST}" \
    -p "${RDS_PORT}" \
    -U "${RDS_MASTER_USER}" \
    -d "${DB}" \
    -q <<-SQL
      GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO news_pipeline;
      GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO news_pipeline;
      GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO news_pipeline;
      ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT ALL ON TABLES TO news_pipeline;
      ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT ALL ON SEQUENCES TO news_pipeline;
SQL
  echo "done"
done

# ---- STEP 4: Verify row counts -----------------------------------------------
echo ""
echo "=== STEP 4: Row count verification ==="
echo "--- Local ---"
PGPASSWORD="${LOCAL_PASS}" psql \
  -h "${LOCAL_HOST}" -p "${LOCAL_PORT}" -U "${LOCAL_USER}" -d news_pipeline \
  -c "SELECT 'raw_articles' AS tbl, COUNT(*) FROM raw_articles UNION ALL SELECT 'entities', COUNT(*) FROM entities UNION ALL SELECT 'signals', COUNT(*) FROM signals;" \
  2>/dev/null || echo "  (local query failed — postgres may be stopping)"

echo "--- RDS ---"
PGPASSWORD="${PGPASSWORD_RDS}" psql \
  -h "${RDS_HOST}" -p "${RDS_PORT}" -U "${RDS_MASTER_USER}" -d news_pipeline \
  -c "SELECT 'raw_articles' AS tbl, COUNT(*) FROM raw_articles UNION ALL SELECT 'entities', COUNT(*) FROM entities UNION ALL SELECT 'signals', COUNT(*) FROM signals;"

echo ""
echo "=== Migration complete ==="
echo "Dump files retained at: ${DUMP_DIR}"
echo ""
echo "Next step: transfer mlruns artifacts to the 2GB EC2:"
echo "  rsync -avz ./mlruns/ ec2-user@<2gb-ip>:/opt/pet-signal-atlas/mlruns/"
