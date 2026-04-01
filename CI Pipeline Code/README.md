# CI Pipeline Code

This folder contains the Jetson-side CI helpers and setup notes.

## Contents

- `run_ci_test.sh`: Local wrapper that runs the Jetson smoke test inside the `edith-glasses:jp64` Docker image.
- `smoke_test.py`: Pure-Python smoke tests that validate the split Jetson runtime modules without requiring real BLE, camera, or network hardware.
- `CI Pipeline SetUp.md`: Personal setup notes for reproducing the self-hosted GitHub runner environment.

## Typical Use

Load the Jetson image first, then run the local smoke test:

```bash
sudo docker load -i "Jetson Code/edith-glasses_jp64.tar"
bash "CI Pipeline Code/run_ci_test.sh"
```

If the image archive lives somewhere else locally, load it from that path before running the script. The GitHub workflow separately looks for a copy under `Jetson Code/Docker/edith-glasses_jp64.tar`.

## Notes

- The GitHub workflow and the local helper both expect the image name `edith-glasses:jp64`.
- The setup notes mention downloading the archive from the Hugging Face artifact `BlueDoorE/edith-glasses_jp64.tar`.
- The smoke tests are meant to catch import and interface regressions before you need Jetson hardware.
