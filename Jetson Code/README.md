# Jetson Code

This folder holds the Linux / Jetson side of the project.

## Contents

- `Code/`: Python runtime that connects to the glasses over BLE, receives frames from the ESP32 stream, runs inference, and sends display updates back to the glasses.
- `Docker/`: Container deployment assets and service notes for running the Jetson runtime in Docker.
- `Even Glasses.md`: Protocol and behavior notes for the Even glasses BLE / display interface.
- `edith-glasses_jp64.tar`: Jetson Docker image archive used for deployment and CI.

## Entry Points

- Direct Python runtime: `python3 "Jetson Code/Code/main.py"`
- Containerized runtime: use the assets documented in `Jetson Code/Docker/README.md`

## Important Notes

- The repo history and some comments still refer to `Demo.py`, but the current split runtime entry point is `Code/main.py`.
- The Docker image archive in this folder is the same deployment artifact referenced in CI and setup notes.
- Some CI helpers currently look for the tar archive under `Jetson Code/Docker/`, so keep the archive location in mind when setting up automation.
