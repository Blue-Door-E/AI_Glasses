#!/usr/bin/env python3
"""CI smoke tests for the split Jetson runtime modules.

These tests intentionally avoid real BLE/camera/network hardware. They verify
that core interfaces exist and that pure-Python logic behaves as expected.
"""

import importlib
import importlib.util
import inspect
import os
import sys
import tempfile
import types
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(REPO_ROOT, "Jetson Code", "Code")
CONFIG_PATH = os.path.join(CODE_DIR, "config.py")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.ndarray = object
    numpy_stub.uint8 = "uint8"

    def _frombuffer(data, *_args, **_kwargs):
        return data

    numpy_stub.frombuffer = _frombuffer
    sys.modules["numpy"] = numpy_stub


# ML.py decorates methods with @torch.no_grad() at import time. If torch is
# absent, provide a tiny compatibility stub so import can still proceed.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")

    def _no_grad_decorator():
        def _wrap(func):
            return func
        return _wrap

    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    torch_stub.no_grad = _no_grad_decorator
    torch_stub.cuda = _CudaStub()
    sys.modules["torch"] = torch_stub


def _load_local_config():
    if os.path.exists(CONFIG_PATH):
        spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load config from {CONFIG_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
        return module

    config_stub = types.ModuleType("config")
    config_stub.ADDRESS_TYPE = "random"
    config_stub.BLE_CHUNK = 180
    config_stub.CONF_THRESHOLD = 0.70
    config_stub.HOST = "111.0.0.1"
    config_stub.IMG_SIZE = 400
    config_stub.LEFT_MAC = "00:00:00:00:00:00"
    config_stub.MODEL_DIR = "outputs/Hide"
    config_stub.NUS_RX = "11111111-1111-1111-1111-111111111111"
    config_stub.NUS_TX = "11111111-1111-1111-1111-111111111111"
    config_stub.PORT = 5000
    config_stub.PRINT_EVERY_N = 30
    config_stub.RIGHT_MAC = "00:00:00:00:00:00"
    config_stub.SAVE_COOLDOWN_SEC = 2.0
    config_stub.SAVE_DIR = "dataset/live_capture"
    config_stub.SAVE_EVERY_N_GLOBAL = 20
    config_stub.SAVE_UNCERTAIN = True
    config_stub.UI_REFRESH_SEC = 0.3
    sys.modules["config"] = config_stub
    return config_stub


config = _load_local_config()


# BLE.py imports bleak at module load. Provide a minimal fallback so this
# smoke test can still run in constrained CI environments.
try:
    import bleak  # noqa: F401
except ModuleNotFoundError:
    bleak_stub = types.ModuleType("bleak")

    class _BleakClientStub:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self.is_connected = False
            self.mtu_size = 23

        async def connect(self):
            self.is_connected = True

        async def disconnect(self):
            self.is_connected = False

        async def start_notify(self, *_args, **_kwargs):
            return None

        async def write_gatt_char(self, *_args, **_kwargs):
            return None

        def set_disconnected_callback(self, *_args, **_kwargs):
            return None

    bleak_stub.BleakClient = _BleakClientStub
    sys.modules["bleak"] = bleak_stub


main_module = importlib.import_module("main")
ble_module = importlib.import_module("BLE")
display_module = importlib.import_module("Display")
ml_module = importlib.import_module("ML")


class JetsonRuntimeSmokeTest(unittest.TestCase):
    def test_core_constants_exist(self):
        self.assertIsInstance(config.LEFT_MAC, str)
        self.assertIsInstance(config.RIGHT_MAC, str)
        self.assertIsInstance(config.HOST, str)
        self.assertIsInstance(config.PORT, int)
        self.assertGreater(config.BLE_CHUNK, 0)

    def test_live_state_contract(self):
        state = main_module.LiveState()
        self.assertTrue(hasattr(state, "prediction"))
        self.assertTrue(hasattr(state, "confidence"))
        self.assertTrue(hasattr(state, "file_index"))
        self.assertEqual(state.prediction, "N/A")
        self.assertEqual(state.file_index, 0)

    def test_find_latest_model_handles_missing_path(self):
        with tempfile.TemporaryDirectory() as tdir:
            missing_dir = os.path.join(tdir, "does_not_exist")
            self.assertIsNone(ml_module.find_latest_model(missing_dir))

    def test_find_latest_model_prefers_newest_valid_dir(self):
        with tempfile.TemporaryDirectory() as tdir:
            old_dir = os.path.join(tdir, "old")
            new_dir = os.path.join(tdir, "new")
            os.makedirs(old_dir, exist_ok=True)
            os.makedirs(new_dir, exist_ok=True)

            old_model = os.path.join(old_dir, "resnet18_scripted.pt")
            old_labels = os.path.join(old_dir, "labels.txt")
            new_model = os.path.join(new_dir, "resnet18_scripted.pt")
            new_labels = os.path.join(new_dir, "labels.txt")

            for path in (old_model, old_labels, new_model, new_labels):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("ok\n")

            os.utime(old_dir, (1, 1))
            os.utime(new_dir, (2, 2))

            found = ml_module.find_latest_model(tdir)
            self.assertIsNotNone(found)
            model_path, labels_path = found
            self.assertEqual(model_path, new_model)
            self.assertEqual(labels_path, new_labels)

    def test_compose_text_shape(self):
        state = main_module.LiveState(
            connected=True,
            prediction="person",
            confidence=0.93,
            file_index=42,
            model_name="resnet18_scripted.pt",
            torch_device="CUDA",
            saving_on=True,
            last_saved_frame=40,
            last_saved_label="person",
            last_saved_conf=0.93,
            last_saved_kind="global",
        )
        text = display_module.compose_text(state)
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("Prediction: person", lines[1])
        self.assertIn("Frame #: 42", lines[3])

    def test_main_and_stream_entrypoints_exist(self):
        self.assertTrue(callable(display_module.stream_loop))
        self.assertTrue(callable(main_module.main))
        self.assertTrue(inspect.iscoroutinefunction(main_module.main))


class JetsonBlePacketSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_0x4e_chunking_and_header(self):
        ble = ble_module.DualBLE("left", "right")
        ble._payload = 20
        sent = []

        async def capture(pkt: bytes):
            sent.append(pkt)

        ble.send_both = capture
        await ble.text_0x4E("Smoke test packet chunking")

        self.assertGreaterEqual(len(sent), 2)

        first = sent[0]
        total = first[2]
        seq = first[1]
        self.assertGreaterEqual(total, 2)

        reconstructed = bytearray()
        for idx, pkt in enumerate(sent):
            self.assertLessEqual(len(pkt), ble._payload)
            self.assertEqual(pkt[0], 0x4E)
            self.assertEqual(pkt[1], seq)
            self.assertEqual(pkt[2], total)
            self.assertEqual(pkt[3], idx)
            reconstructed.extend(pkt[9:])

        self.assertEqual(total, len(sent))
        self.assertEqual(reconstructed.decode("utf-8"), "Smoke test packet chunking")

    async def test_safe_text_0x4e_retries_on_not_connected(self):
        ble = ble_module.DualBLE("left", "right")
        calls = {"ensure": 0, "text": 0, "disconnect": 0, "connect": 0}

        async def ensure_connected():
            calls["ensure"] += 1

        async def disconnect():
            calls["disconnect"] += 1

        async def connect():
            calls["connect"] += 1

        async def text_0x4e(_text: str):
            calls["text"] += 1
            if calls["text"] == 1:
                raise RuntimeError("BLE client not connected")

        ble.ensure_connected = ensure_connected
        ble.disconnect = disconnect
        ble.connect = connect
        ble.text_0x4E = text_0x4e

        await ble.safe_text_0x4E("hello")

        self.assertEqual(calls["ensure"], 1)
        self.assertEqual(calls["disconnect"], 1)
        self.assertEqual(calls["connect"], 1)
        self.assertEqual(calls["text"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
