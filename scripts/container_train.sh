#!/usr/bin/env bash
# GCE container workflow: stage data on SSD and train against Cloud Run MLflow.
#
# Required env vars:
#   GCS_BUCKET   - GCS bucket name (no gs:// prefix), e.g. "my-feral-bucket"
#
# Optional env vars:
#   DATA_DIR     - local SSD mount (default: /data)
#   RUN_RECIPE   - complete Hydra Run Recipe (default: baseline)
#   GCS_DATA_PREFIX - GCS prefix containing images/ and annotations/
#   MLFLOW_TRACKING_URI - Cloud Run MLflow tracking endpoint (required)
#   HYDRA_OVERRIDES - extra Hydra CLI overrides passed to the trainer
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET env var required}"
RUN_RECIPE="${RUN_RECIPE:-baseline}"
GCS_DATA_PREFIX="${GCS_DATA_PREFIX:-}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:?MLFLOW_TRACKING_URI env var required}"

if [[ -n "${GCS_DATA_PREFIX}" ]]; then
    GCS_DATA_PREFIX="${GCS_DATA_PREFIX%/}/"
fi
echo "=== [1/4] Syncing images from gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}images/ ==="
gcloud storage rsync -r \
    "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}images/" \
    "${DATA_DIR}/images/"

echo "=== [2/4] Syncing annotations from gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}annotations/ ==="
gcloud storage rsync -r \
    "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}annotations/" \
    "${DATA_DIR}/annotations/"

echo "=== [3/4] Syncing DVC tracker from gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}data.dvc ==="
gcloud storage cp \
    "gs://${GCS_BUCKET}/${GCS_DATA_PREFIX}data.dvc" \
    "${DATA_DIR}/data.dvc"

echo "=== [4/4] Starting training ==="
python -m feral_vision.training.trainer \
    --config-name "runs/${RUN_RECIPE}" \
    data.root="${DATA_DIR}" \
    tracking.tracking_uri="${MLFLOW_TRACKING_URI}" \
    ${HYDRA_OVERRIDES:-}

echo "=== Done. ==="
