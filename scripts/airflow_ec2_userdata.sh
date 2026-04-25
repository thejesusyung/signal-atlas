#!/usr/bin/env bash
# scripts/airflow_ec2_userdata.sh
#
# EC2 User Data script for the Airflow on-demand instance.
# Runs on every instance start (idempotent).
# Assumes Amazon Linux 2023. Switch dnf → apt-get for Ubuntu.
#
# Required IAM permissions on the EC2 instance role:
#   - ssm:GetParameter on /pet-signal/airflow/.env
#   - (optional) ecr:GetAuthorizationToken if pulling from ECR instead of building

set -euo pipefail

LOG_FILE="/var/log/airflow-startup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -u)] === Airflow EC2 startup ==="

REPO_DIR="/opt/pet-signal-atlas"
REPO_URL="https://github.com/<your-org>/PET_SIGNAL_ATLAS.git"
BRANCH="infra/2gb-migration"
SSM_PARAM="/pet-signal/airflow/.env"

# ---- Install Docker (idempotent) -----------------------------------------------
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  dnf update -y
  dnf install -y docker git
  systemctl enable docker
  systemctl start docker
  usermod -aG docker ec2-user

  # Docker Compose plugin
  COMPOSE_VERSION=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
    | grep '"tag_name"' | cut -d'"' -f4)
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -SL \
    "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  echo "Docker installed: $(docker --version)"
fi

# Ensure Docker is running (may have been stopped between boots)
systemctl start docker

# ---- Pull or clone repo --------------------------------------------------------
if [ -d "${REPO_DIR}/.git" ]; then
  echo "Pulling latest code on branch ${BRANCH}..."
  git -C "${REPO_DIR}" fetch origin
  git -C "${REPO_DIR}" checkout "${BRANCH}"
  git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
else
  echo "Cloning repo..."
  git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

# ---- Pull secrets from SSM Parameter Store ------------------------------------
echo "Fetching .env.airflow from SSM (${SSM_PARAM})..."
aws ssm get-parameter \
  --name "${SSM_PARAM}" \
  --with-decryption \
  --query "Parameter.Value" \
  --output text \
  > "${REPO_DIR}/.env.airflow"
echo "Secrets written."

# ---- Ensure data directory exists (bind-mounted by compose) --------------------
mkdir -p "${REPO_DIR}/data"

# ---- Build and start Airflow services ------------------------------------------
cd "${REPO_DIR}"
echo "Building Airflow image..."
docker compose -f docker-compose.airflow.yml build --pull

echo "Starting Airflow services..."
docker compose -f docker-compose.airflow.yml up -d

echo "[$(date -u)] === Airflow EC2 startup complete ==="
echo "Airflow webserver: http://$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4):8080"
