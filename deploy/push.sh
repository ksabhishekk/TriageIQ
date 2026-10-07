#!/bin/sh
# ============================================================
# deploy/push.sh — ship TriageIQ to the server and (re)start it, from the Mac:
#     sh deploy/push.sh <server-ip>          (key: ~/.ssh/triageiq-key.pem, or set KEY=...)
# Copies ONLY what the server needs — API and deploy files, five production model files, and the database
# dump (~330 MB); XAI artifacts are added only with XAI_ENABLED=1. Never .env, the 16 GB database or raw
# data. Then, on the server: one-time setup
# (deploy/server_setup.sh) and `docker compose --profile public up -d --build` (NGINX on port 80).
# Re-run it after any change: only changed files are sent, only changed image layers are rebuilt.
# ============================================================
set -e
HOST="$1"
KEY="${KEY:-$HOME/.ssh/triageiq-key.pem}"
[ -n "$HOST" ] || { echo "usage: sh deploy/push.sh <server-ip>"; exit 1; }
cd "$(dirname "$0")/.."                                 # the repo root

FILES="api/__init__.py api/main.py api/scorer.py api/db.py api/schemas.py api/requirements.txt
       api/xai.py
       deploy/api.Dockerfile deploy/docker-compose.yml deploy/initdb/01_restore.sh deploy/nginx/default.conf.template
       deploy/server_setup.sh .dockerignore
       training/outputs/serving_v3/model.onnx training/outputs/serving_v3/tokenizer.json
       training/outputs/serving_v3/preprocessing.json training/outputs/serving_v3/calibration.json
       training/outputs/serving_v3/model_card.json training/outputs/serving_v3/serving_db.dump"

if [ "${XAI_ENABLED:-0}" = "1" ]; then
    for artifact in training/outputs/serving_v3/xai_encoder.onnx \
                    training/outputs/serving_v3/xai_head.onnx \
                    training/outputs/serving_v3/xai_background.npz; do
        [ -f "$artifact" ] || { echo "XAI is enabled but $artifact is missing"; exit 1; }
    done
    FILES="$FILES training/serving/export_xai_onnx.py
           training/outputs/serving_v3/xai_encoder.onnx
           training/outputs/serving_v3/xai_head.onnx
           training/outputs/serving_v3/xai_background.npz"
fi

echo "== copying files to $HOST"
# --relative keeps the folder structure under ~/triageiq on the server
rsync -az --relative -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" $FILES "ubuntu@$HOST:triageiq/"

echo "== setting up and starting on the server"
ssh -i "$KEY" "ubuntu@$HOST" 'cd triageiq && sh deploy/server_setup.sh &&
    sudo docker compose -f deploy/docker-compose.yml --profile public up -d --build &&
    sudo docker compose -f deploy/docker-compose.yml ps'

echo "== done: http://$HOST/docs"
