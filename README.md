# AI Glasses

## Overview
This repo contains two main components:

- `main/`: C/C++ firmware for the ESP32-S3 Sense. It captures images and streams video over TCP.
- `Jetson Code/`: Jetson-side machine learning code. It runs models and communicates results back to the glasses for display.

## Hand-Signal Functions (In Progress)
Each hand sign maps to a different function. This mapping is under active development.

Planned / in-progress functions:

- **Person lookup**: perform a set of hand inputs to capture a high-quality photo, search the person online via the web, and display a summary on the glasses.
- **Reddit browser**: browse and navigate Reddit using hand signals.
- **Live translate**: listen and translate everything locally and live.
- **GPS map**: show the user’s location and map view.
- **Text messages & notifications**: under consideration, dependent on Linux support.

## Notes
If you’re looking for the ESP32 firmware, start in `main/`. If you’re working on model inference or UI feedback on the glasses, start in `Jetson Code/`.
