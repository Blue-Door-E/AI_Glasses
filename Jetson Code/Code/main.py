#!/usr/bin/env python3
# Demo.py — Jetson/Linux BLE-stable version (Docker-safe; model loading unchanged)

import asyncio
import os
import threading
import time
from dataclasses import dataclass

from BLE import DualBLE
from Display import compose_text, stream_loop
from ML import OptionalModel
from config import (
    LEFT_MAC,
    RIGHT_MAC,
    SAVE_EVERY_N_GLOBAL,
    SAVE_UNCERTAIN,
    UI_REFRESH_SEC,
)

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

@dataclass
class LiveState:
    connecting: bool = True
    connected: bool = False
    ip_status: str = ""
    prediction: str = "N/A"
    confidence: float = 0.0
    file_index: int = 0
    stop: bool = False
    attempt: int = 0
    last_infer_ms: float = 0.0
    last_frame_ms: float = 0.0
    last_ble_write_ms: float = 0.0
    last_pred_tperf: float = 0.0
    
    # --- NEW (display-only) ---
    model_name: str = "NoModel"
    torch_device: str = "CPU"     # "CUDA" or "CPU"
    saving_on: bool = False
    last_saved_frame: int = -1
    last_saved_label: str = ""
    last_saved_conf: float = 0.0
    last_saved_kind: str = ""     # "global" or "uncertain"

STATE = LiveState()


async def main():
    ble = DualBLE(LEFT_MAC, RIGHT_MAC)
    print("[BLE] Connecting...")
    await ble.connect()
    print("[BLE] Connected.")

    await ble.prompt_0x11("Loading.")
    await ble.safe_text_0x4E("EDITH starting...\nPlease wait.\nPreparing subsystems...")
    await ble.prompt_0x11("Connecting")

    model = OptionalModel()

    # --- NEW (display-only) ---
    # Row1: model filename + CUDA/CPU
    if getattr(model, "model_path", ""):
        STATE.model_name = os.path.basename(model.model_path)
    else:
        STATE.model_name = "NoModel"

    STATE.torch_device = getattr(model, "device_tag", "CPU")

    # Row3: whether saving is enabled (based on existing config)
    STATE.saving_on = bool(
        model.ready and CV2_OK and (
            (SAVE_UNCERTAIN is True) or (SAVE_EVERY_N_GLOBAL and SAVE_EVERY_N_GLOBAL > 0)
        )
    )

    t = threading.Thread(target=stream_loop, args=(STATE, model), daemon=True)
    t.start()
    try:
        last_txt = ""
        last_sent_t = 0.0
        last_sent_frame = -1

        while True:
            txt = compose_text(STATE)

            now = time.perf_counter()
            frame = STATE.file_index

            # Send when:
            #  - text changed, OR
            #  - frame advanced (so row4 updates), OR
            #  - it’s been > 1.0s since last send (keeps display alive after reconnect)
            force_periodic = (now - last_sent_t) >= 1.0
            should_send = (txt != last_txt) or (frame != last_sent_frame) or force_periodic

            if should_send:
                t_ble0 = time.perf_counter()
                try:
                    await ble.safe_text_0x4E(txt)   # IMPORTANT: always use safe write
                except Exception as e:
                    print("[BLE] Write failed:", repr(e))
                finally:
                    t_ble1 = time.perf_counter()
                    STATE.last_ble_write_ms = (t_ble1 - t_ble0) * 1000.0

                last_txt = txt
                last_sent_t = now
                last_sent_frame = frame

            await asyncio.sleep(UI_REFRESH_SEC)

    except KeyboardInterrupt:
        pass

    finally:
        STATE.stop = True
        try:
            await ble.text_0x4E("Shutting down...\nGoodbye.")
        except Exception:
            pass
        await ble.disconnect()
        print("[BLE] Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())
