"""No-op BLE stub that mirrors the DualBLE async interface.

Drop-in replacement for BLE.DualBLE used during benchmark runs so that
the real main.py runtime path can execute without BLE hardware.
"""

from __future__ import annotations


class DualBLE:
    """No-op stand-in for the real DualBLE class."""

    def __init__(self, left_id: str = "", right_id: str = ""):
        self.left_id = left_id
        self.right_id = right_id
        self._tx_seq = 1
        self._payload = 180

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def prompt_0x11(self, txt: str) -> None:
        pass

    async def text_0x4E(self, text: str) -> None:
        pass

    async def safe_text_0x4E(self, text: str) -> None:
        pass

    async def ensure_connected(self) -> None:
        pass
