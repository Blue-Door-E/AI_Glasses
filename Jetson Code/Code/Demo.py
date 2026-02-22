#!/usr/bin/env python3
# Demo.py — Jetson/Linux BLE-stable version (Docker-safe; model loading unchanged)

import asyncio, socket, struct, time, os, threading
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from datetime import datetime

from config import (
    ADDRESS_TYPE,
    HOST,
    LEFT_MAC,
    NUS_RX,
    NUS_TX,
    PORT,
    RIGHT_MAC,
)

# -------- Optional deps --------
try:
    import torch
    from torchvision import transforms
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

from bleak import BleakClient

# ====== CONFIG ======
# Deployment-specific BLE/network values are imported from config.py.
# If bluetoothctl shows "(random)" for the device, try "random".
# If your Bleak doesn't support address_type, we'll ignore it safely.
IMG_SIZE   = 400
MODEL_DIR  = "outputs/Hide"
SAVE_DIR   = "dataset/live_capture"
BLE_CHUNK  = 180

CONF_THRESHOLD = 0.70
SAVE_COOLDOWN_SEC = 2.0

# Existing behavior (kept)
SAVE_CONFIDENT_EVERY_N = 20
SAVE_UNCERTAIN = True

# NEW: global saver ("every N picture saved with respected label")
# If you want it different from SAVE_CONFIDENT_EVERY_N, set it explicitly.
SAVE_EVERY_N_GLOBAL = SAVE_CONFIDENT_EVERY_N

PRINT_EVERY_N = 30
UI_REFRESH_SEC = 0.3
# =====================

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

def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ------------- model loader (UNCHANGED) -------------
def find_latest_model(model_dir="outputs") -> Optional[Tuple[str, str]]:
    try:
        if not os.path.exists(model_dir):
            return None
        subs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))]
        subs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        for d in subs:
            m, l = os.path.join(d, "resnet18_scripted.pt"), os.path.join(d, "labels.txt")
            if os.path.exists(m) and os.path.exists(l):
                return m, l
        m, l = os.path.join(model_dir, "resnet18_scripted.pt"), os.path.join(model_dir, "labels.txt")
        if os.path.exists(m) and os.path.exists(l):
            return m, l
    except Exception:
        pass
    return None

class OptionalModel:
    def __init__(self):
        self.ready = False
        self.labels = []
        self.model = None
        self.tf = None
        self.device = "cpu"
   
        # --- NEW (display-only) ---
        self.model_path = ""
        self.device_tag = "CPU"

        if not TORCH_OK:
            print("[Model] torch not available; running without inference.")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Model] Using device: {self.device}")
        except Exception:
            self.device = "cpu"
	# --- NEW (display-only) ---
        self.device_tag = "CUDA" if self.device == "cuda" else "CPU"
        found = find_latest_model(MODEL_DIR)
        if not found:
            print("[Model] No model found; predictions will be N/A.")
            return

        try:
            mfile, lfile = found
            self.model_path = mfile
            self.model = torch.jit.load(mfile, map_location=self.device)
            self.model.eval()
            with open(lfile, "r", encoding="utf-8") as f:
                self.labels = [x.strip() for x in f if x.strip()]
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.ready = True
            print(f"[Model] Loaded {mfile} with {len(self.labels)} labels.")
        except Exception as e:
            print("[Model] Load failed:", e)
            self.ready = False

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> Tuple[str, float, int]:
        if not self.ready or bgr is None:
            return "N/A", 0.0, -1
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if CV2_OK else bgr[:, :, ::-1]
        x = self.tf(img).unsqueeze(0)
        if self.device != "cpu":
            x = x.to(self.device, non_blocking=True)
        y = self.model(x)
        if self.device != "cpu":
            y = y.to("cpu")
        p = torch.softmax(y, dim=1)[0]
        conf, idx = torch.max(p, dim=0)
        idxi = int(idx)
        label = self.labels[idxi] if 0 <= idxi < len(self.labels) else f"class_{idxi}"
        return label, float(conf), idxi

# ------------- BLE control -------------
class DualBLE:
    def __init__(self, left_id: str, right_id: str):
        self.left_id = left_id
        self.right_id = right_id
        self._tx_seq = 1  # rolling seq for 0x4E packets

        self.left: Optional[BleakClient] = None
        self.right: Optional[BleakClient] = None

        self.left_write = NUS_RX
        self.right_write = NUS_RX

        self.left_notify = NUS_TX
        self.right_notify = NUS_TX

        self._left_last_notify: bytes = b""
        self._left_notify_event = asyncio.Event()
        
        self._ble_dead = asyncio.Event()   # set when any client disconnects
        self._reconnect_lock = asyncio.Lock()
        self._payload = 20  # negotiated max payload (will update after connect)

    async def _update_payload(self):
        # try to acquire MTU (BlueZ backend); ignore if not supported
        for cli in (self.left, self.right):
            if not cli:
                continue
            try:
                await cli._acquire_mtu()
            except Exception:
                pass

        # pick the smallest payload both sides can handle
        def max_payload(cli):
            mtu = getattr(cli, "mtu_size", None) or 23
            return max(20, mtu - 3)

        if self.left and self.right:
            self._payload = min(max_payload(self.left), max_payload(self.right), BLE_CHUNK)
        elif self.left:
            self._payload = min(max_payload(self.left), BLE_CHUNK)
        else:
            self._payload = min(20, BLE_CHUNK)

        # safety floor
        if self._payload < 20:
            self._payload = 20


    def _make_client(self, addr: str, timeout: float) -> BleakClient:
        if ADDRESS_TYPE:
            try:
                return BleakClient(addr, timeout=timeout, address_type=ADDRESS_TYPE)
            except TypeError:
                pass
        return BleakClient(addr, timeout=timeout)
    def _on_disconnected(self, _client):
        # Called by Bleak on disconnect
        try:
            self._ble_dead.set()
        except Exception:
            pass

    async def _wait_left_ack(self, timeout: float = 0.35) -> bool:
        """
        Wait for *any* notify from left arm as an 'ack' gate.
        Even docs: send left, then right after left ack. :contentReference[oaicite:1]{index=1}
        """
        try:
            self._left_notify_event.clear()
            await asyncio.wait_for(self._left_notify_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


    def _on_left_notify(self, _char, data: bytearray):
        self._left_last_notify = bytes(data)
        self._left_notify_event.set()

    async def _force_discovery(self, client: BleakClient):
        """
        Key fix for: 'Service Discovery has not been performed yet'
        Tries get_services() when available; otherwise just yields time.
        """
        get_services = getattr(client, "get_services", None)
        if callable(get_services):
            try:
                await get_services()
            except Exception:
                pass

    async def _has_char(self, client: BleakClient, uuid: str) -> bool:
        svcs = getattr(client, "services", None)
        if not svcs:
            return False

        # Newer bleak: services.get_characteristic
        try:
            ch = svcs.get_characteristic(uuid)
            if ch is not None:
                return True
        except Exception:
            pass

        # Fallback: brute scan
        try:
            for svc in svcs:
                for ch in getattr(svc, "characteristics", []):
                    if str(getattr(ch, "uuid", "")).lower() == uuid.lower():
                        return True
        except Exception:
            pass

        return False

    async def _wait_for_char(self, client: BleakClient, uuid: str, tries: int = 40, delay: float = 0.15):
        """
        Wait until NUS_RX exists in discovered services.
        """
        for _ in range(tries):
            await self._force_discovery(client)
            if await self._has_char(client, uuid):
                return
            await asyncio.sleep(delay)

        # Don't hard-fail here; we will still try writing and handle errors.
        print(f"[BLE] Warning: characteristic not visible yet: {uuid} (continuing)")

    async def _wait_for_services(self, client: BleakClient, *, tries: int = 20, delay: float = 0.15):
        """
        Compatible across Bleak versions:
        - Newer Bleak: may support await client.get_services()
        - Older Bleak: services appear in client.services after connect + delay
        """
        await self._force_discovery(client)

        for _ in range(tries):
            svcs = getattr(client, "services", None)
            if svcs is not None:
                inner = getattr(svcs, "services", None)
                if inner is None:
                    try:
                        if len(list(svcs)) > 0:
                            return
                    except Exception:
                        pass
                else:
                    try:
                        if len(inner) > 0:
                            return
                    except Exception:
                        pass
            await asyncio.sleep(delay)

        print("[BLE] Warning: services not populated (continuing anyway).")

    async def _connect_one(self, addr: str, timeout: float = 30.0, attempts: int = 8) -> BleakClient:
        last_err: Optional[Exception] = None
        for k in range(1, attempts + 1):
            client = self._make_client(addr, timeout=timeout)
            try:
                await client.connect()
                try:
                    client.set_disconnected_callback(self._on_disconnected)
                except Exception:
                    pass

                await asyncio.sleep(0.8)  # wearable settle

                # IMPORTANT: force/await discovery
                await self._wait_for_services(client)
                await self._wait_for_char(client, NUS_RX)

                return client
            except Exception as e:
                last_err = e
                try:
                    if getattr(client, "is_connected", False):
                        await client.disconnect()
                except Exception:
                    pass
                await asyncio.sleep(min(3.0, 0.5 * k))

        msg = f"BLE connect failed for {addr}"
        if last_err:
            msg += f": {last_err!r}"
        raise RuntimeError(msg)

    async def connect(self):
        self.left = await self._connect_one(self.left_id)
        try: self.left.set_disconnected_callback(self._on_disconnected)
        except Exception: pass
        try:
            await self.left.start_notify(self.left_notify, self._on_left_notify)
        except Exception:
            pass

        await asyncio.sleep(0.8)  # IMPORTANT between left/right on wearables

        self.right = await self._connect_one(self.right_id)
        try: self.right.set_disconnected_callback(self._on_disconnected)
        except Exception: pass
        try:
            await self.right.start_notify(self.right_notify, lambda c, d: None)
        except Exception:
            pass
        await self._update_payload()
        print(f"[BLE] Using payload={self._payload} bytes (mtu-based)")

    async def disconnect(self):
        for cli in (self.left, self.right):
            try:
                if cli and cli.is_connected:
                    await cli.disconnect()
            except Exception:
                pass

    @staticmethod
    def _max_payload(client: BleakClient) -> int:
        mtu = getattr(client, "mtu_size", None) or 23
        return max(20, mtu - 3)

    async def _write_pkt(self, client: BleakClient, char_uuid: str, pkt: bytes, *, timeout: float = 1.5):
        if not client or not getattr(client, "is_connected", False):
            raise RuntimeError("BLE client not connected")

        # IMPORTANT: pkt must already be <= self._payload
        if len(pkt) > self._payload:
            raise RuntimeError(f"Packet too large for payload: {len(pkt)} > {self._payload}")

        await asyncio.wait_for(
            client.write_gatt_char(char_uuid, pkt, response=False),
            timeout=timeout
        )



    async def send_both(self, pkt: bytes):
        # pkt MUST be <= self._payload
        await self._write_pkt(self.left, self.left_write, pkt)

        _ = await self._wait_left_ack(timeout=0.35)
        await asyncio.sleep(0.02)

        await self._write_pkt(self.right, self.right_write, pkt)

    async def prompt_0x11(self, txt: str):
        b = txt.encode("utf-8")
        pkt = bytes([0x11, len(b)]) + b
        await self.send_both(pkt)
    async def text_0x4E(self, text: str):
        data = text.encode("utf-8")

        header_len = 9
        max_data = max(1, self._payload - header_len)  # MTU-safe
        parts = [data[i:i + max_data] for i in range(0, len(data), max_data)]
        total = len(parts)

        seq = self._tx_seq & 0xFF
        if seq == 0:
            seq = 1
        self._tx_seq = (seq + 1) & 0xFF

        for idx, part in enumerate(parts):
            hdr = bytes([
                0x4E,
                seq & 0xFF,
                total & 0xFF,
                idx & 0xFF,
                0x71,
                0, 0, 0,
                max(1, total)
            ])
            pkt = hdr + part

            # sanity: must fit in one BLE write
            if len(pkt) > self._payload:
                raise RuntimeError(f"Internal error: pkt {len(pkt)} > payload {self._payload}")

            await self.send_both(pkt)

    async def ensure_connected(self):
        async with self._reconnect_lock:
            if self._ble_dead.is_set():
                # Something dropped; start fresh
                self._ble_dead.clear()
                await self.disconnect()

            if not (self.left and self.left.is_connected):
                self.left = await self._connect_one(self.left_id)
                try:
                    self.left.set_disconnected_callback(self._on_disconnected)
                except Exception:
                    pass
                try:
                    await self.left.start_notify(self.left_notify, self._on_left_notify)
                except Exception:
                    pass
                await asyncio.sleep(0.4)

            if not (self.right and self.right.is_connected):
                self.right = await self._connect_one(self.right_id)
                try:
                    self.right.set_disconnected_callback(self._on_disconnected)
                except Exception:
                    pass
                try:
                    await self.right.start_notify(self.right_notify, lambda c, d: None)
                except Exception:
                    pass
                await asyncio.sleep(0.2)

            # cheap re-discovery to avoid BlueZ races
            await self._force_discovery(self.left)
            await self._force_discovery(self.right)
            await self._wait_for_char(self.left, NUS_RX)
            await self._wait_for_char(self.right, NUS_RX)
    async def safe_text_0x4E(self, text: str):
        try:
            await self.ensure_connected()
            await self.text_0x4E(text)
            return
        except asyncio.TimeoutError:
            # write hung/slow -> force reconnect
            await self.disconnect()
            await self.connect()
            await self.text_0x4E(text)
        except Exception as e:
            msg = str(e) or repr(e)
            if "disconnected" in msg.lower() or "not connected" in msg.lower():
                await self.disconnect()
                await self.connect()
                await self.text_0x4E(text)
                return
            if "Service Discovery has not been performed yet" in msg:
                await self._force_discovery(self.left)
                await self._force_discovery(self.right)
                await self._wait_for_char(self.left, NUS_RX)
                await self._wait_for_char(self.right, NUS_RX)
                await self.text_0x4E(text)
                return
            # fallback: one reconnect attempt
            await self.disconnect()
            await self.connect()
            await self.text_0x4E(text)

# ------------- Stream loop w/ saving + metrics -------------
def stream_loop(state: LiveState, model: OptionalModel):
    ensure_dir(SAVE_DIR)
    uncertain_path = os.path.join(SAVE_DIR, "uncertain")
    ensure_dir(uncertain_path)

    # Ensure class directories exist (so saving never silently fails)
    class_dirs = {}
    last_saved_time = {"__uncertain__": 0.0, "__global__": 0.0}
    if model.ready:
        for lbl in model.labels:
            p = os.path.join(SAVE_DIR, lbl)
            ensure_dir(p)
            class_dirs[lbl] = p

    backoff = 1.0
    while not state.stop:
        state.attempt += 1
        state.connecting = True
        state.connected = False
        state.ip_status = f"Attempting {HOST}:{PORT} (try {state.attempt})"

        s = socket.socket()
        s.settimeout(4)
        try:
            s.connect((HOST, PORT))
            s.settimeout(None)

            state.connected = True
            state.connecting = False
            state.ip_status = f"Connected {HOST}:{PORT}"
            print("[Stream] Connected", HOST, PORT)
            backoff = 1.0
            count = state.file_index

            while not state.stop:
                t0 = time.perf_counter()

                hdr = s.recv(4)
                if not hdr or len(hdr) < 4:
                    raise RuntimeError("EOF header")

                frame_len = struct.unpack(">I", hdr)[0]
                data = b""
                while len(data) < frame_len:
                    pkt = s.recv(frame_len - len(data))
                    if not pkt:
                        raise RuntimeError("EOF mid-frame")
                    data += pkt

                img = None
                if CV2_OK:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

                t_infer0 = time.perf_counter()

                if img is not None and model.ready:
                    pred_label, conf, pred_idx = model.predict(img)
                    state.prediction = pred_label
                    state.confidence = conf
                else:
                    pred_label, conf, pred_idx = "N/A", 0.0, -1
                    state.prediction, state.confidence = "N/A", 0.0

                t_infer1 = time.perf_counter()

                state.last_infer_ms = (t_infer1 - t_infer0) * 1000.0
                state.last_frame_ms = (t_infer1 - t0) * 1000.0
                state.last_pred_tperf = t_infer1

                ts = time.time()
                ts_str = _now_str()

                # ---- SAVING ----
                if model.ready and CV2_OK and img is not None and pred_label != "N/A":
                    # 1) Uncertain saving (kept)
                    if conf < CONF_THRESHOLD and SAVE_UNCERTAIN:
                        if ts - last_saved_time["__uncertain__"] >= SAVE_COOLDOWN_SEC:
                            fname = f"{ts_str}_pred-{pred_label}_conf-{conf:.3f}.jpg"
                            cv2.imwrite(os.path.join(uncertain_path, fname), img)
                            last_saved_time["__uncertain__"] = ts
                            # --- NEW (display-only) ---
                            state.last_saved_frame = count + 1
                            state.last_saved_label = pred_label
                            state.last_saved_conf = conf
                            state.last_saved_kind = "uncertain"

                    # 2) NEW: Global every-N saving to label folder
                    if SAVE_EVERY_N_GLOBAL > 0:
                        # count below is incremented after saving logic, so use (count+1)
                        next_count = count + 1
                        if (next_count % SAVE_EVERY_N_GLOBAL) == 0:
                            if ts - last_saved_time["__global__"] >= SAVE_COOLDOWN_SEC:
                                outdir = class_dirs.get(pred_label) or os.path.join(SAVE_DIR, pred_label)
                                ensure_dir(outdir)
                                fname = f"{ts_str}_{pred_label}_conf-{conf:.3f}_frame-{next_count}.jpg"
                                cv2.imwrite(os.path.join(outdir, fname), img)
                                last_saved_time["__global__"] = ts
                                # --- NEW (display-only) ---
                                state.last_saved_frame = next_count
                                state.last_saved_label = pred_label
                                state.last_saved_conf = conf
                                state.last_saved_kind = "global"

                count += 1
                state.file_index = count

                if PRINT_EVERY_N > 0 and (count % PRINT_EVERY_N) == 0:
                    print(f"[Perf] frame#{count} net+decode+infer_ms={state.last_frame_ms:.1f} "
                          f"(infer={state.last_infer_ms:.1f})")

        except Exception as e:
            print("[Stream] Not connected:", repr(e))
            state.connected = False
            state.connecting = True
            state.ip_status = f"Attempting {HOST}:{PORT} (retry in {int(backoff)}s)"

            try:
                s.close()
            except Exception:
                pass

            sleep_s = min(10.0, backoff)
            for _ in range(int(sleep_s * 10)):
                if state.stop:
                    break
                time.sleep(0.1)

            backoff = min(10.0, backoff * 1.8)
            continue

        try:
            s.close()
        except Exception:
            pass

def compose_text(state: LiveState) -> str:
    # Row 1: IP + model + CUDA/CPU
    if state.connected:
        ip_part = f"{HOST}:{PORT}"
    else:
        ip_part = state.ip_status or f"{HOST}:{PORT}"

    row1 = f"{ip_part} | {state.model_name} | {state.torch_device}"

    # Row 2: Prediction (confidence)
    if state.prediction == "N/A":
        row2 = "Prediction: N/A"
    else:
        row2 = f"Prediction: {state.prediction} ({state.confidence:.2f})"

    # Row 3: Saving + last frame saved
    if state.saving_on:
        if state.last_saved_frame >= 0:
            row3 = (
                f"Saving: ON | Last: {state.last_saved_kind} "
                f"#{state.last_saved_frame} {state.last_saved_label} ({state.last_saved_conf:.2f})"
            )
        else:
            row3 = "Saving: ON | Last: none"
    else:
        row3 = "Saving: OFF | Last: none"

    # Row 4: Frame count (live)
    row4 = f"Frame #: {state.file_index}"

    return f"{row1}\n{row2}\n{row3}\n{row4}"


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
