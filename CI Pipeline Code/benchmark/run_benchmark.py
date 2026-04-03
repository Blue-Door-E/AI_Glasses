#!/usr/bin/env python3
"""Jetson benchmark harness — orchestrates replay, runtime, metrics, and artifacts.

Modes:
    pr-short     2-minute replay, Nsight trace + tegrastats
    nightly-long 30-minute replay, tegrastats only (no Nsight)

This script:
  1. Starts tegrastats logging
  2. Starts the TCP replay server (background)
  3. Optionally wraps the runtime in Nsight Systems profiling
  4. Launches the real main.py with BLE stubbed out
  5. Waits for replay to finish or timeout
  6. Collects runtime metrics from the LiveState dataclass
  7. Parses tegrastats + runtime metrics into summary artifacts
  8. Writes summary.json, summary.csv, summary.md

Usage:
    python run_benchmark.py --mode pr-short  --video bench_2min.mp4 \\
        --artifacts-dir ./artifacts --runtime-dir "Jetson Code/Code"

    python run_benchmark.py --mode nightly-long --video bench_30min.mp4 \\
        --artifacts-dir ./artifacts --runtime-dir "Jetson Code/Code"

Environment variables (override defaults):
    BENCH_REPLAY_HOST   Replay TCP host  (default: 127.0.0.1)
    BENCH_REPLAY_PORT   Replay TCP port  (default: 8080)
    BENCH_REPLAY_FPS    Replay frame rate (default: 5)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure benchmark package is importable
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from tegrastats_parser import parse_log, summarize
from summary import build_summary, summary_to_markdown, write_summaries

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPLAY_HOST = os.getenv("BENCH_REPLAY_HOST", "127.0.0.1")
REPLAY_PORT = int(os.getenv("BENCH_REPLAY_PORT", "8080"))
REPLAY_FPS = float(os.getenv("BENCH_REPLAY_FPS", "5"))

MODE_CONFIG = {
    "pr-short": {
        "timeout_s": 180,       # 3 min safety ceiling for a 2-min video
        "nsight": True,
        "tegrastats_interval_ms": 500,
    },
    "nightly-long": {
        "timeout_s": 2100,      # 35 min safety ceiling for 30-min video
        "nsight": False,
        "tegrastats_interval_ms": 1000,
    },
}


def _start_tegrastats(log_path: Path, interval_ms: int = 1000) -> subprocess.Popen:
    """Start tegrastats logging to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"[Harness] tegrastats pid={proc.pid} -> {log_path}")
    return proc


def _start_replay(video: str, host: str, port: int, fps: float) -> subprocess.Popen:
    """Start the TCP replay server as a subprocess."""
    cmd = [
        sys.executable,
        str(BENCH_DIR / "replay_server.py"),
        "--video", video,
        "--host", host,
        "--port", str(port),
        "--fps", str(fps),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"[Harness] Replay server pid={proc.pid}")
    return proc


def _build_runtime_cmd(runtime_dir: str) -> list[str]:
    """Build the command to run main.py directly."""
    return [sys.executable, "-u", str(Path(runtime_dir) / "main.py")]


def _build_runtime_env(runtime_dir: str, host: str, port: int) -> dict[str, str]:
    """Build env vars that inject the replay endpoint."""
    env = os.environ.copy()
    env["GLASSES_HOST"] = host
    env["GLASSES_PORT"] = str(port)
    env["PYTHONPATH"] = f"{runtime_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _setup_ble_stub(runtime_dir: str) -> Path:
    """Swap the real BLE.py with the stub. Returns path to the backup.

    This is the most reliable injection method — main.py runs exactly as
    in production, but `from BLE import DualBLE` picks up the stub since
    it physically replaces the file in the same directory.
    """
    real_ble = Path(runtime_dir) / "BLE.py"
    backup = Path(runtime_dir) / "BLE.py.bench_backup"
    stub_src = BENCH_DIR / "ble_stub.py"

    # Backup the real BLE.py
    if real_ble.exists():
        shutil.copy2(real_ble, backup)
        print(f"[Harness] Backed up BLE.py -> {backup}")

    # Overwrite with stub
    shutil.copy2(stub_src, real_ble)
    print(f"[Harness] Injected BLE stub -> {real_ble}")

    # Remove any cached bytecode so Python doesn't use stale .pyc
    pycache = Path(runtime_dir) / "__pycache__"
    if pycache.exists():
        for f in pycache.glob("BLE.cpython-*.pyc"):
            f.unlink()
            print(f"[Harness] Removed cached {f.name}")

    return backup


def _cleanup_ble_stub(runtime_dir: str, backup: Path) -> None:
    """Restore the real BLE.py from backup."""
    real_ble = Path(runtime_dir) / "BLE.py"
    if backup.exists():
        shutil.copy2(backup, real_ble)
        backup.unlink()
        print(f"[Harness] Restored BLE.py from backup")


def _collect_runtime_metrics(stdout_text: str, exit_code: int, duration_s: float) -> dict:
    """Extract runtime metrics from main.py stdout (Perf lines)."""
    import re
    infer_times: list[float] = []
    frame_times: list[float] = []
    frame_count = 0
    reconnect_count = 0

    for line in stdout_text.splitlines():
        # [Perf] frame#30 net+decode+infer_ms=45.2 (infer=12.3)
        m = re.search(
            r"\[Perf\]\s+frame#(\d+)\s+net\+decode\+infer_ms=([\d.]+)\s+\(infer=([\d.]+)\)",
            line,
        )
        if m:
            frame_count = max(frame_count, int(m.group(1)))
            frame_times.append(float(m.group(2)))
            infer_times.append(float(m.group(3)))
            continue

        if "[Stream] Not connected:" in line:
            reconnect_count += 1

    def _p95(vals: list[float]) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        k = int(len(vals) * 0.95)
        return vals[min(k, len(vals) - 1)]

    return {
        "frames_processed": frame_count,
        "inference_avg_ms": round(sum(infer_times) / len(infer_times), 2) if infer_times else 0.0,
        "inference_p95_ms": round(_p95(infer_times), 2),
        "frame_time_avg_ms": round(sum(frame_times) / len(frame_times), 2) if frame_times else 0.0,
        "frame_time_p95_ms": round(_p95(frame_times), 2),
        "reconnect_count": reconnect_count,
        "exit_code": exit_code,
        "duration_s": round(duration_s, 2),
    }


def run(
    mode: str,
    video: str,
    artifacts_dir: str,
    runtime_dir: str,
    commit: str = "",
) -> int:
    """Run a full benchmark pass. Returns 0 on success."""
    cfg = MODE_CONFIG[mode]
    arts = Path(artifacts_dir)
    arts.mkdir(parents=True, exist_ok=True)

    tegra_log = arts / "tegrastats.log"
    runtime_log = arts / "runtime_stdout.log"

    # --- Setup: swap BLE.py with stub ---
    ble_backup = _setup_ble_stub(runtime_dir)

    # --- Start tegrastats (may already be running on the host) ---
    tegra_proc = None
    try:
        tegra_proc = _start_tegrastats(tegra_log, cfg["tegrastats_interval_ms"])
        time.sleep(1)  # give tegrastats a moment to start
    except FileNotFoundError:
        print("[Harness] tegrastats not found in container (expected — host runs it)")

    # --- Start replay server ---
    replay_proc = _start_replay(video, REPLAY_HOST, REPLAY_PORT, REPLAY_FPS)
    # Give the server time to bind
    time.sleep(2)

    # --- Build runtime command ---
    runtime_cmd = _build_runtime_cmd(runtime_dir)
    runtime_env = _build_runtime_env(runtime_dir, REPLAY_HOST, REPLAY_PORT)

    # --- Optionally wrap with Nsight ---
    if cfg["nsight"]:
        from nsight_capture import find_nsys
        nsys = find_nsys()
        if nsys:
            nsight_out = arts / "nsight"
            nsight_out.mkdir(parents=True, exist_ok=True)
            runtime_cmd = [
                nsys, "profile",
                "--output", str(nsight_out / "benchmark"),
                "--force-overwrite", "true",
                "--trace", "cuda,nvtx,osrt",
                "--sample", "cpu",
                "--export", "sqlite",
                "--",
            ] + runtime_cmd
            print(f"[Harness] Nsight wrapping enabled -> {nsight_out}")
        else:
            print("[Harness] WARNING: nsys not found, skipping Nsight profiling")

    # --- Launch runtime ---
    print(f"[Harness] Launching runtime: {' '.join(runtime_cmd)}")
    t_start = time.monotonic()

    runtime_proc = subprocess.Popen(
        runtime_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=runtime_env,
        cwd=runtime_dir,
    )
    print(f"[Harness] Runtime pid={runtime_proc.pid}")

    # --- Wait for replay to finish, then give runtime a grace period ---
    # The runtime reconnects forever by design, so we watch the replay
    # server: once the video is fully sent, we allow a short grace period
    # for the runtime to process remaining frames, then terminate it.
    harness_terminated = False
    try:
        # Wait for either the runtime to exit on its own or the full timeout
        # Check periodically if the replay server has finished
        deadline = time.monotonic() + cfg["timeout_s"]
        grace_deadline = None
        stdout_bytes = None

        while True:
            # Check if runtime exited on its own
            ret = runtime_proc.poll()
            if ret is not None:
                stdout_bytes, _ = runtime_proc.communicate()
                break

            # Check if replay server finished (video fully sent)
            if grace_deadline is None and replay_proc.poll() is not None:
                grace_deadline = time.monotonic() + 5  # 5s grace after replay ends
                print("[Harness] Replay finished, giving runtime 5s to wrap up")

            # Grace period expired — terminate cleanly
            if grace_deadline and time.monotonic() >= grace_deadline:
                print("[Harness] Grace period over, terminating runtime")
                runtime_proc.terminate()
                harness_terminated = True
                try:
                    stdout_bytes, _ = runtime_proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    runtime_proc.kill()
                    stdout_bytes, _ = runtime_proc.communicate()
                break

            # Hard timeout — something is stuck
            if time.monotonic() >= deadline:
                print(f"[Harness] Timeout after {cfg['timeout_s']}s, terminating runtime")
                runtime_proc.terminate()
                try:
                    stdout_bytes, _ = runtime_proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    runtime_proc.kill()
                    stdout_bytes, _ = runtime_proc.communicate()
                break

            time.sleep(0.5)

    except Exception as exc:
        print(f"[Harness] Unexpected error waiting for runtime: {exc}")
        runtime_proc.kill()
        stdout_bytes, _ = runtime_proc.communicate()

    t_end = time.monotonic()
    duration_s = t_end - t_start
    exit_code = runtime_proc.returncode
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")

    # Save runtime stdout
    runtime_log.write_text(stdout_text)
    print(f"[Harness] Runtime exited with code={exit_code} after {duration_s:.1f}s")

    # --- Stop replay ---
    if replay_proc.poll() is None:
        replay_proc.terminate()
        try:
            replay_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            replay_proc.kill()

    # --- Stop tegrastats ---
    if tegra_proc and tegra_proc.poll() is None:
        tegra_proc.terminate()
        try:
            tegra_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tegra_proc.kill()

    # --- Restore real BLE.py ---
    _cleanup_ble_stub(runtime_dir, ble_backup)

    # --- Parse metrics ---
    runtime_metrics = _collect_runtime_metrics(stdout_text, exit_code, duration_s)

    # If the harness terminated the runtime after the replay finished and
    # frames were processed, treat it as a clean run (exit 0) for reporting.
    if harness_terminated and runtime_metrics["frames_processed"] > 0:
        runtime_metrics["exit_code"] = 0
        print(f"[Harness] Replay complete with {runtime_metrics['frames_processed']} frames — marking as OK")

    runtime_json_path = arts / "runtime_metrics.json"
    runtime_json_path.write_text(json.dumps(runtime_metrics, indent=2))

    tegra_summary = {}
    if tegra_log.exists() and tegra_log.stat().st_size > 0:
        samples = parse_log(tegra_log)
        tegra_summary = summarize(samples)
        (arts / "tegrastats_summary.json").write_text(json.dumps(tegra_summary, indent=2))

    metadata = {
        "mode": mode,
        "video": os.path.basename(video),
        "commit": commit or os.getenv("GITHUB_SHA", "unknown"),
        "runner": os.getenv("RUNNER_NAME", "unknown"),
        "docker_image": os.getenv("BENCH_DOCKER_IMAGE", "edith-glasses:jp64"),
    }

    summary = build_summary(tegra_summary, runtime_metrics, metadata)
    paths = write_summaries(summary, arts)
    print(f"[Harness] Summary files: {paths}")

    # Print markdown summary
    md = summary_to_markdown(summary)
    print(md)

    # Write to GITHUB_STEP_SUMMARY if available and the file exists
    # (inside Docker the env var may point to a host path that doesn't exist)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary and os.path.exists(os.path.dirname(step_summary) or "."):
        try:
            with open(step_summary, "a") as f:
                f.write(md)
                f.write("\n")
        except OSError:
            pass

    # Harness-terminated after replay finished is a clean exit if frames
    # were actually processed (runtime loops forever reconnecting by design).
    if harness_terminated and runtime_metrics["frames_processed"] > 0:
        return 0
    return 0 if exit_code == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jetson benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=list(MODE_CONFIG.keys()),
        help="Benchmark mode: pr-short (2min + Nsight) or nightly-long (30min)",
    )
    ap.add_argument("--video", required=True, help="Path to benchmark video file")
    ap.add_argument("--artifacts-dir", default="./benchmark_artifacts",
                    help="Directory for output artifacts")
    ap.add_argument("--runtime-dir", default="Jetson Code/Code",
                    help="Path to the runtime source directory containing main.py")
    ap.add_argument("--commit", default="", help="Git commit SHA for metadata")
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"ERROR: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    runtime_main = Path(args.runtime_dir) / "main.py"
    if not runtime_main.exists():
        print(f"ERROR: Runtime not found: {runtime_main}", file=sys.stderr)
        sys.exit(1)

    rc = run(
        mode=args.mode,
        video=args.video,
        artifacts_dir=args.artifacts_dir,
        runtime_dir=args.runtime_dir,
        commit=args.commit,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
