# GitHub Workflows

This folder contains the CI workflows used by this repository.

## Workflows

- `firmware-build.yml`: Builds the ESP32-S3 firmware inside an ESP-IDF container when firmware-related files change. It generates `main/configs.h` during CI from the `AP_SSID` and `AP_PASS` GitHub secrets.
- `jetson-smoke-test.yml`: Runs the Jetson Python smoke tests on a self-hosted `linux/arm64` runner. It loads `edith-glasses:jp64` from a tar archive if needed, then runs `CI Pipeline Code/smoke_test.py` inside the container.

## Prerequisites

- The firmware workflow needs the Wi-Fi secrets configured in GitHub.
- The Jetson workflow needs a self-hosted ARM64 runner with Docker access.
- The Jetson workflow expects the deployment image archive at `Jetson Code/Docker/edith-glasses_jp64.tar` unless the image is already preloaded on the runner.

## Notes

- These workflows are the CI entry points; helper scripts and setup notes live under `CI Pipeline Code/`.
- If you change runtime paths or image names, update both the workflows and the local CI helper scripts.
