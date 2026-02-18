# Repository Guidelines

## Project Structure & Module Organization
- `main/`: ESP-IDF firmware for the XIAO ESP32-S3 Sense (camera capture, AP/networking, device config).
- `Jetson Code/Code/`: Jetson runtime app (`Demo.py`) for BLE messaging, model inference, and frame processing.
- `Jetson Code/Docker/`: deployment assets for Jetson (`entrypoint.sh`, `dockerrun`, systemd notes).
- `README.md`: high-level project overview and feature direction.

Keep ESP32 changes isolated to `main/` and Jetson/Linux changes under `Jetson Code/`.

## Build, Test, and Development Commands
Firmware (ESP-IDF):
- `idf.py set-target esp32s3`: set the MCU target once per workspace.
- `idf.py build`: compile firmware from the repository root.
- `idf.py -p <PORT> flash monitor`: flash and open serial logs.

Jetson runtime:
- `python3 "Jetson Code/Code/Demo.py"`: run the Jetson app directly.
- `sudo docker load -i edith-glasses_jp64.tar`: import deployment image (if available).
- `sudo docker run ... edith-glasses:jp64`: run container using the pattern in `Jetson Code/Docker/README.md`.

## Coding Style & Naming Conventions
- C code: follow existing style in `main/` (4-space indentation, snake_case functions/variables, UPPER_SNAKE_CASE macros/constants).
- Python code: PEP 8-style naming (`snake_case` functions, `PascalCase` classes), keep module-level config grouped at top.
- Prefer short, focused files; place hardware-specific constants in config/header files (`configs.h`, `macros.h`).

## Testing Guidelines
- No formal automated test suite is currently committed.
- For firmware PRs, include: build success (`idf.py build`) and a short runtime verification note (boot, camera init, AP/TCP behavior).
- For Jetson PRs, include: startup result, BLE connect status, and inference/display behavior observed.

## Commit & Pull Request Guidelines
- Use concise, imperative commit messages (example: `Add Jetson Docker deployment files`).
- Keep subject lines <= 72 characters; one logical change per commit when possible.
- PRs should include:
  - what changed and why,
  - affected area (`main/` firmware, Jetson runtime, or Docker/deployment),
  - validation steps/commands run,
  - logs or screenshots for UI/device-visible behavior changes.

## Security & Configuration Tips
- Do not commit secrets, private keys, or environment-specific credentials.
- Treat MAC addresses, IPs, and model paths in `Demo.py` as deployment config; document overrides in PRs.
