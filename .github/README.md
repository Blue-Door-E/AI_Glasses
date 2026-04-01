# RTOS Edge Inference — AI Smart-Glasses with On-Device ML

## Demo

[![Watch the demo](https://img.youtube.com/vi/N8S3p4ECKG8/maxresdefault.jpg)](https://youtu.be/N8S3p4ECKG8)


A portable, fully offline AI smart-glasses system that performs real-time machine learning inference on the edge. Built with a custom FreeRTOS firmware running on an ESP32-S3 for camera capture and an NVIDIA Jetson for GPU-accelerated gesture classification — all running locally with zero cloud dependency.

## Overview

This project combines embedded systems, real-time operating systems, and machine learning into a wearable that can recognize hand gestures and map them to arbitrary actions — skipping songs, taking photos, browsing content — all through a pair of smart-glasses connected to a portable Jetson compute unit carried in a fanny pack.

### How It Works

```
┌─────────────────────┐        Wi-Fi (TCP)        ┌─────────────────────┐
│   ESP32-S3 Sense    │ ──────────────────────────▶│    NVIDIA Jetson    │
│   (Camera Module)   │    JPEG stream @ ~5 fps    │  (ML Inference)     │
│                     │                            │                     │
│  FreeRTOS firmware  │                            │  PyTorch ResNet-18  │
│  Zero-copy DMA      │        BLE (NUS)           │  CUDA accelerated   │
│  PSRAM frame buffers│ ◀──────────────────────────│                     │
└─────────────────────┘    Status text to glasses  └─────────────────────┘
```

1. The **ESP32-S3 Sense** captures JPEG frames using a zero-copy DMA pipeline through PSRAM and streams them over a local Wi-Fi access point via TCP
2. The **Jetson** receives frames, decodes them, and runs a custom-trained ResNet-18 classifier on the GPU
3. Classification results are sent back to the **smart-glasses display** over Bluetooth Low Energy in real-time

The entire pipeline runs locally. No internet connection is required for core functionality. Frame-to-prediction latency is typically under 100ms.

## Key Features

- **Real-time edge inference**: Hand-gesture classification at ~5 fps with sub-100ms latency
- **Fully offline**: All processing happens on-device — no cloud APIs, no data leaves the fanny pack
- **Privacy-first design**: Biometric hand data is stored locally and never transmitted externally, similar to how Face ID keeps your face data on your phone
- **Custom RTOS firmware**: Dual-core FreeRTOS design with dedicated cores for capture and streaming
- **Zero-copy camera pipeline**: DMA transfers from camera sensor → PSRAM → TCP socket
- **Automatic BLE reconnection**: Robust Bluetooth handling with MTU negotiation and chunked writes
- **Active learning**: Optional frame saving for dataset expansion and model retraining
- **Portable**: Entire compute stack fits in a fanny pack with the Jetson and battery

## Project Structure

```
├── Main/          ESP32-S3 FreeRTOS camera + Wi-Fi streaming firmware (C)
├── jetson code/            Jetson ML inference engine + BLE glasses driver (Python)
├── LICENSE            MIT License
└── README.md
```

See each folder's README for detailed documentation, build instructions, and configuration.

## Hardware

| Component | Role |
|-----------|------|
| XIAO ESP32-S3 Sense | Camera capture + Wi-Fi AP |
| NVIDIA Jetson (Orin Nano / Xavier NX) | GPU inference + BLE communication |
| Smart-glasses with BLE display | Wearable output (Nordic UART Service) |
| Portable battery pack | Powers the Jetson and ESP32 |

## Getting Started

### Firmware (ESP32)

```bash
cd firmware/
# Requires ESP-IDF v5.x toolchain
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

### Jetson Software

```bash
cd jetson/
cp config_sample.py config.py
# Edit config.py with your BLE MAC addresses
pip install bleak opencv-python numpy
python3 main.py
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
