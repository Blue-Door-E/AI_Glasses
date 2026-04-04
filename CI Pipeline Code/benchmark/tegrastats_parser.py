#!/usr/bin/env python3
"""Parse tegrastats log output into structured metrics.

tegrastats line format (Jetson Orin / Xavier):
    RAM 3456/7620MB (lfb 20x4MB) SWAP 0/3810MB (cached 0MB) CPU [20%@1190,15%@1190,...]
    GR3D_FREQ 45% ... VDD_IN 4500mW/4200mW ...

This parser extracts: CPU%, GPU%, RAM MB, and power mW from each sample line.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Sample:
    timestamp_s: float = 0.0
    cpu_pcts: List[int] = field(default_factory=list)  # per-core
    cpu_avg: float = 0.0
    gpu_pct: int = 0
    ram_used_mb: int = 0
    ram_total_mb: int = 0
    power_mw: int = 0


# --- regex patterns ---
_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RE_CPU_CORES = re.compile(r"CPU\s*\[([^\]]+)\]")
_RE_CORE_PCT = re.compile(r"(\d+)%")
_RE_GPU = re.compile(r"GR3D_FREQ\s+(\d+)%")
# Power: try VDD_IN first (total board power), then VDD_CPU_GPU_CV
_RE_POWER = re.compile(r"VDD_(?:IN|CPU_GPU_CV|SOC|TOTAL)\s+(\d+)mW")


def parse_line(line: str, ts: float = 0.0) -> Optional[Sample]:
    """Parse a single tegrastats output line. Returns None if unparseable."""
    s = Sample(timestamp_s=ts)

    m = _RE_RAM.search(line)
    if m:
        s.ram_used_mb = int(m.group(1))
        s.ram_total_mb = int(m.group(2))

    m = _RE_CPU_CORES.search(line)
    if m:
        s.cpu_pcts = [int(x) for x in _RE_CORE_PCT.findall(m.group(1))]
        if s.cpu_pcts:
            s.cpu_avg = statistics.mean(s.cpu_pcts)

    m = _RE_GPU.search(line)
    if m:
        s.gpu_pct = int(m.group(1))

    m = _RE_POWER.search(line)
    if m:
        s.power_mw = int(m.group(1))

    # Need at least RAM or CPU to consider it a valid sample
    if not s.cpu_pcts and s.ram_used_mb == 0:
        return None

    return s


def parse_log(path: str | Path) -> List[Sample]:
    """Parse an entire tegrastats log file."""
    path = Path(path)
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            s = parse_line(line, ts=float(i))
            if s is not None:
                samples.append(s)
    return samples


def _percentile(vals: List[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = (len(vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    frac = k - lo
    return vals[lo] + frac * (vals[hi] - vals[lo])


def summarize(samples: List[Sample]) -> dict:
    """Compute aggregate stats from parsed samples."""
    if not samples:
        return {"error": "no samples"}

    cpu_vals = [s.cpu_avg for s in samples if s.cpu_pcts]
    gpu_vals = [float(s.gpu_pct) for s in samples]
    ram_vals = [float(s.ram_used_mb) for s in samples if s.ram_used_mb > 0]
    power_vals = [float(s.power_mw) for s in samples if s.power_mw > 0]

    def _stats(vals: List[float], unit: str = "") -> dict:
        if not vals:
            return {"avg": 0.0, "p95": 0.0, "peak": 0.0, "unit": unit}
        return {
            "avg": round(statistics.mean(vals), 2),
            "p95": round(_percentile(vals, 95), 2),
            "peak": round(max(vals), 2),
            "unit": unit,
        }

    return {
        "sample_count": len(samples),
        "cpu": _stats(cpu_vals, "%"),
        "gpu": _stats(gpu_vals, "%"),
        "ram": _stats(ram_vals, "MB"),
        "power": _stats(power_vals, "mW"),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Parse tegrastats log")
    ap.add_argument("logfile", help="Path to tegrastats log")
    ap.add_argument("--json", dest="json_out", help="Write summary JSON to path")
    args = ap.parse_args()

    samples = parse_log(args.logfile)
    summary = summarize(samples)
    print(json.dumps(summary, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
