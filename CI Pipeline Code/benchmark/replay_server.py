#!/usr/bin/env python3
"""TCP replay server that sends video frames using production framing.

Reads a video file, encodes each frame as JPEG, and sends it over TCP
using the same protocol as the ESP32 camera:
    4-byte big-endian frame length | JPEG bytes

Usage:
    python replay_server.py --video PATH [--host 127.0.0.1] [--port 8080]
                            [--fps 5] [--jpeg-quality 85] [--loop]
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import cv2


def _encode_jpeg(frame, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def serve_once(
    video_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    fps: float = 5.0,
    jpeg_quality: int = 85,
    loop: bool = False,
) -> dict:
    """Serve frames to exactly one client, then return stats."""

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[Replay] Listening on {host}:{port}  video={video_path}  fps={fps}")

    conn, addr = srv.accept()
    print(f"[Replay] Client connected from {addr}")

    frame_interval = 1.0 / fps
    total_sent = 0
    stats = {"frames_sent": 0, "bytes_sent": 0, "errors": 0}

    try:
        while True:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Replay] ERROR: cannot open {video_path}", file=sys.stderr)
                stats["errors"] += 1
                break

            while True:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    break

                jpeg = _encode_jpeg(frame, jpeg_quality)
                header = struct.pack(">I", len(jpeg))
                try:
                    conn.sendall(header + jpeg)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    print("[Replay] Client disconnected")
                    cap.release()
                    return stats

                stats["frames_sent"] += 1
                stats["bytes_sent"] += 4 + len(jpeg)

                elapsed = time.monotonic() - t0
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            cap.release()

            if not loop:
                break

    finally:
        try:
            conn.close()
        except OSError:
            pass
        srv.close()

    print(f"[Replay] Done. Sent {stats['frames_sent']} frames, "
          f"{stats['bytes_sent'] / 1024 / 1024:.1f} MB")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="TCP video replay server")
    ap.add_argument("--video", required=True, help="Path to benchmark video")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--loop", action="store_true", help="Loop video indefinitely")
    args = ap.parse_args()

    serve_once(
        video_path=args.video,
        host=args.host,
        port=args.port,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
