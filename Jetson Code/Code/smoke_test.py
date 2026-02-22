#!/usr/bin/env python3
"""CI smoke tests for Jetson runtime expectations in Demo.py.

These tests intentionally avoid real BLE/camera/network hardware. They verify
that core interfaces exist and that pure-Python logic behaves as expected.
"""

import asyncio
import inspect
import os
import sys
import tempfile
import types
import unittest


# Demo.py decorates methods with @torch.no_grad() at import time. If torch is
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


# Demo.py imports bleak at module load. Provide a minimal fallback so this
# smoke test can still run in constrained CI environments.
try:
    import bleak  # noqa: F401
except ModuleNotFoundError:
    bleak_stub = types.ModuleType("bleak")

    class _BleakClientStub:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            self.is_connected = False
            self.mtu_size = 23

    bleak_stub.BleakClient = _BleakClientStub
    sys.modules["bleak"] = bleak_stub


import Demo  # noqa: E402


class DemoSmokeTest(unittest.TestCase):
    def test_core_constants_exist(self):
        self.assertIsInstance(Demo.LEFT_MAC, str)
        self.assertIsInstance(Demo.RIGHT_MAC, str)
        self.assertIsInstance(Demo.HOST, str)
        self.assertIsInstance(Demo.PORT, int)
        self.assertGreater(Demo.BLE_CHUNK, 0)

    def test_live_state_contract(self):
        state = Demo.LiveState()
        self.assertTrue(hasattr(state, "prediction"))
        self.assertTrue(hasattr(state, "confidence"))
        self.assertTrue(hasattr(state, "file_index"))
        self.assertEqual(state.prediction, "N/A")
        self.assertEqual(state.file_index, 0)

    def test_find_latest_model_handles_missing_path(self):
        with tempfile.TemporaryDirectory() as tdir:
            missing_dir = os.path.join(tdir, "does_not_exist")
            self.assertIsNone(Demo.find_latest_model(missing_dir))

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

            found = Demo.find_latest_model(tdir)
            self.assertIsNotNone(found)
            model_path, labels_path = found
            self.assertEqual(model_path, new_model)
            self.assertEqual(labels_path, new_labels)

    def test_compose_text_shape(self):
        state = Demo.LiveState(
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
        text = Demo.compose_text(state)
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("Prediction: person", lines[1])
        self.assertIn("Frame #: 42", lines[3])

    def test_main_and_stream_entrypoints_exist(self):
        self.assertTrue(callable(Demo.stream_loop))
        self.assertTrue(callable(Demo.main))
        self.assertTrue(inspect.iscoroutinefunction(Demo.main))


class DemoBlePacketSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_text_0x4e_chunking_and_header(self):
        ble = Demo.DualBLE("left", "right")
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
        ble = Demo.DualBLE("left", "right")
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
