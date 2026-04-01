# Jetson Runtime Code

This folder contains the Python runtime that runs on Jetson or another Linux host.

## Runtime Flow

The runtime:

1. Connects to the left and right glasses arms over BLE.
2. Opens the ESP32 video stream over TCP.
3. Loads the newest available TorchScript model, if one exists.
4. Runs inference on incoming frames.
5. Sends formatted status text back to the glasses display.
6. Optionally saves uncertain or periodic frames for later review.

## Files

- `main.py`: Main async entry point and live state management.
- `BLE.py`: Dual-BLE connection handling and packet writes to the glasses.
- `Display.py`: TCP frame receiver, inference loop, save logic, and text composition.
- `ML.py`: Model discovery and prediction wrapper.
- `config.py`: Deployment-specific local configuration.
- `config_sample.py`: Template for creating `config.py`.

## Setup

Create `config.py` from the sample and fill in the real values for:

- BLE MAC addresses
- NUS UUIDs / address type
- ESP32 AP host and port
- Model directory and save settings

Run the app with:

```bash
python3 "Jetson Code/Code/main.py"
```

## Notes

- `main.py` is the current entry point even though some older docs still say `Demo.py`.
- `__pycache__/` is generated output and should not be treated as hand-maintained source.
- This folder is what the local smoke tests target through `PYTHONPATH`.
