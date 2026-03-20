#!/bin/sh
set -e

APP_DIR="/home/CI_Pipeline/AI_Glasses/Jetson Code"

echo "[entrypoint] whoami: $(whoami)"
echo "[entrypoint] initial pwd: $(pwd)"
echo "[entrypoint] listing /home/CI_Pipeline/AI_Glasses/Jetson Code:"
ls -la /home/CI_Pipeline/AI_Glasses/Jetson Code || true
echo "[entrypoint] listing APP_DIR:"
ls -la "$APP_DIR" || true
echo "[entrypoint] listing model:"
ls -la "$APP_DIR/model" || true
echo "[entrypoint] listing model dir:"
ls -la "$APP_DIR/model/Hide" || true

while true; do
  echo "[entrypoint] cd to $APP_DIR"
  cd "$APP_DIR"
  echo "[entrypoint] pwd now: $(pwd)"
  python3 -u main.py
  code=$?
  echo "[entrypoint] main.py exited code=${code}, restarting..."
  sleep 2
done

