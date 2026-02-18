#!/bin/sh
set -e

APP_DIR="/home/erick/EDITH_CODE"

echo "[entrypoint] whoami: $(whoami)"
echo "[entrypoint] initial pwd: $(pwd)"
echo "[entrypoint] listing /home/erick:"
ls -la /home/erick || true
echo "[entrypoint] listing APP_DIR:"
ls -la "$APP_DIR" || true
echo "[entrypoint] listing outputs:"
ls -la "$APP_DIR/outputs" || true
echo "[entrypoint] listing model dir:"
ls -la "$APP_DIR/outputs/Hide" || true

while true; do
  echo "[entrypoint] cd to $APP_DIR"
  cd "$APP_DIR"
  echo "[entrypoint] pwd now: $(pwd)"
  python3 -u Demo.py
  code=$?
  echo "[entrypoint] Demo.py exited code=${code}, restarting..."
  sleep 2
done

