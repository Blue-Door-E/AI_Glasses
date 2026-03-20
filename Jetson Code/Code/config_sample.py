"""Runtime configuration constants for Jetson Demo.
MOCK. Please fill in with your information 

Keep deployment-specific values here (BLE MACs, Wi-Fi/network settings).
"""

import os

# BLE device MAC addresses
LEFT_MAC =  os.getenv("GLASSES_LEFT_MAC",  "xx:xx:xx:xx:xx:xx")
RIGHT_MAC = os.getenv("GLASSES_RIGHT_MAC", "xx:xx:xx:xx:xx:xx")

# Bleak address type. Use "random" when bluetoothctl shows "(random)".
ADDRESS_TYPE = os.getenv("GLASSES_ADDRESS_TYPE", "random")

# Nordic UART Service UUIDs (Even uses NUS)
NUS_RX = os.getenv("GLASSES_NUS_RX", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
NUS_TX = os.getenv("GLASSES_NUS_TX", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")

# ESP32 AP / Wi-Fi details
WIFI_SSID = os.getenv("GLASSES_WIFI_SSID", "")
WIFI_PASSWORD = os.getenv("GLASSES_WIFI_PASSWORD", "")
HOST = os.getenv("GLASSES_HOST", "xxx.xxx.x.x")
PORT = int(os.getenv("GLASSES_PORT", "xxxx"))

# Runtime defaults shared across split modules
IMG_SIZE = 400
MODEL_DIR = "outputs/Hide"
SAVE_DIR = "dataset/live_capture"
BLE_CHUNK = 180
CONF_THRESHOLD = 0.70
SAVE_COOLDOWN_SEC = 2.0
SAVE_CONFIDENT_EVERY_N = 20
SAVE_UNCERTAIN = True
SAVE_EVERY_N_GLOBAL = SAVE_CONFIDENT_EVERY_N
PRINT_EVERY_N = 30
UI_REFRESH_SEC = 0.3
