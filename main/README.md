# Firmware (`main`)

This folder contains the ESP-IDF application code for the XIAO ESP32-S3 Sense.

## Responsibilities

The firmware currently handles:

- camera initialization
- frame capture
- Wi-Fi access point startup
- TCP streaming support for the Jetson / Linux side
- optional SD-card-related helpers

## Files

- `main.c`: Firmware entry point. Starts the camera task and AP task.
- `cam.c` / `cam.h`: Camera setup, capture tasks, and SD-card helpers.
- `AP.c` / `AP.h`: Wi-Fi AP startup and access-point-side task logic.
- `configs.c` / `configs.h`: Local or generated configuration values such as AP credentials.
- `sample_config.c` / `sample_config.h`: Template config files.
- `macros.h`: Shared constants and macros.
- `idf_component.yml`: ESP-IDF component dependency definition.
- `Kconfig.projbuild`: ESP-IDF project configuration entries.

## Build

From the repository root:

```bash
idf.py set-target esp32s3
idf.py build
```

Flash and monitor:

```bash
idf.py -p <PORT> flash monitor
```

## Config Notes

- CI generates `configs.h` from GitHub secrets during the firmware workflow.
- For local work, use the sample config files or provide your own local values without committing secrets.
- The firmware is tightly coupled to the ESP32-S3 camera + AP streaming path, so validate on hardware after any behavior change.

## Subfolders

- `rtos/`: Reserved for future RTOS-specific helpers and task split-outs.
