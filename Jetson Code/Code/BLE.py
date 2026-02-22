from __future__ import annotations

import asyncio
from typing import Optional

from bleak import BleakClient

from config import ADDRESS_TYPE, BLE_CHUNK, NUS_RX, NUS_TX


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
