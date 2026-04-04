#!/usr/bin/env python3
"""Nsight Systems capture helper for short benchmark runs.

Wraps nsys to:
  1. Profile the benchmark runtime with GPU/CPU sampling
  2. Produce a .nsys-rep trace file
  3. Export stats to SQLite for machine-readable analysis

Usage (standalone):
    python nsight_capture.py --output-dir ./artifacts --duration 120 \\
        -- python3 main.py

Typically invoked by the benchmark harness, not directly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_nsys() -> str | None:
    """Locate the nsys binary."""
    nsys = shutil.which("nsys")
    if nsys:
        return nsys
    # Common Jetson install paths
    for candidate in [
        "/opt/nvidia/nsight-systems/bin/nsys",
        "/usr/local/bin/nsys",
        "/opt/nvidia/nsight-systems-cli/bin/nsys",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def profile(
    cmd: list[str],
    output_dir: str | Path,
    trace_name: str = "benchmark",
    duration: int | None = None,
    sample_cpu: bool = True,
    trace_gpu: bool = True,
) -> dict:
    """Run nsys profile on the given command. Returns paths to artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nsys = find_nsys()
    if not nsys:
        print("[Nsight] WARNING: nsys not found, skipping profiling", file=sys.stderr)
        return {"skipped": True, "reason": "nsys not found"}

    rep_path = output_dir / f"{trace_name}.nsys-rep"
    sqlite_path = output_dir / f"{trace_name}.sqlite"

    # Build nsys profile command
    nsys_cmd = [
        nsys, "profile",
        "--output", str(output_dir / trace_name),
        "--force-overwrite", "true",
        "--trace", "cuda,nvtx,osrt" if trace_gpu else "osrt",
        "--sample", "cpu" if sample_cpu else "none",
        "--export", "sqlite",
    ]
    if duration:
        nsys_cmd.extend(["--duration", str(duration)])

    nsys_cmd.append("--")
    nsys_cmd.extend(cmd)

    print(f"[Nsight] Running: {' '.join(nsys_cmd)}")
    result = subprocess.run(nsys_cmd, capture_output=False)

    artifacts = {
        "nsys_exit_code": result.returncode,
        "nsys_rep": str(rep_path) if rep_path.exists() else None,
        "sqlite": str(sqlite_path) if sqlite_path.exists() else None,
    }

    # Attempt to export stats as CSV if sqlite exists
    if sqlite_path.exists():
        csv_path = output_dir / f"{trace_name}_stats.csv"
        try:
            export_cmd = [
                nsys, "stats",
                "--report", "gpukernsum",
                "--format", "csv",
                "--output", str(csv_path),
                str(rep_path),
            ]
            subprocess.run(export_cmd, capture_output=True, timeout=60)
            if csv_path.exists():
                artifacts["stats_csv"] = str(csv_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    print(f"[Nsight] Artifacts: {artifacts}")
    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="Nsight Systems capture helper")
    ap.add_argument("--output-dir", default="./nsight_artifacts")
    ap.add_argument("--trace-name", default="benchmark")
    ap.add_argument("--duration", type=int, default=None,
                    help="Max profiling duration in seconds")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="Command to profile (after --)")
    args = ap.parse_args()

    # Strip leading -- if present
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("ERROR: No command specified", file=sys.stderr)
        sys.exit(1)

    profile(
        cmd=cmd,
        output_dir=args.output_dir,
        trace_name=args.trace_name,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
