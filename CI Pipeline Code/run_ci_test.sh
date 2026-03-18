#!/usr/bin/env bash
# Local CI smoke test runner.
# Mirrors what the GitHub Actions workflow does so you can test before pushing.
#
# Usage (from repo root or anywhere):
#   bash "CI Pipeline Code/run_ci_test.sh"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[ci] Repo root: $REPO_ROOT"
echo "[ci] Checking for edith-glasses:jp64 image..."
if ! sudo docker image inspect edith-glasses:jp64 >/dev/null 2>&1; then
  echo "[ci] ERROR: edith-glasses:jp64 image not found. Load it first:"
  echo "       sudo docker load -i \"Jetson Code/Docker/edith-glasses_jp64.tar\""
  exit 1
fi

echo "[ci] Running smoke test inside container..."
sudo docker run --rm \
  --entrypoint /bin/bash \
  -v "$REPO_ROOT:/workspace" \
  -w /workspace \
  -e PYTHONPATH="/workspace/Jetson Code/Code" \
  edith-glasses:jp64 \
  -lc 'python3 "CI Pipeline Code/smoke_test.py"'

echo "[ci] Smoke test passed."
