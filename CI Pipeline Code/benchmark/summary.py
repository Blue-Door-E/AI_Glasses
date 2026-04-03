#!/usr/bin/env python3
"""Generate benchmark summary files (JSON, CSV, Markdown).

Combines tegrastats host metrics with runtime metrics collected by the
benchmark harness into unified summary artifacts.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any


def build_summary(
    tegra_summary: dict,
    runtime_metrics: dict,
    metadata: dict | None = None,
) -> dict:
    """Merge tegrastats and runtime metrics into a single summary dict."""
    summary: dict[str, Any] = {}

    if metadata:
        summary["metadata"] = metadata

    # Host metrics from tegrastats
    summary["host"] = {
        "cpu": tegra_summary.get("cpu", {}),
        "gpu": tegra_summary.get("gpu", {}),
        "ram": tegra_summary.get("ram", {}),
        "power": tegra_summary.get("power", {}),
        "sample_count": tegra_summary.get("sample_count", 0),
    }

    # Runtime metrics
    summary["runtime"] = {
        "frames_processed": runtime_metrics.get("frames_processed", 0),
        "inference_avg_ms": runtime_metrics.get("inference_avg_ms", 0.0),
        "inference_p95_ms": runtime_metrics.get("inference_p95_ms", 0.0),
        "frame_time_avg_ms": runtime_metrics.get("frame_time_avg_ms", 0.0),
        "frame_time_p95_ms": runtime_metrics.get("frame_time_p95_ms", 0.0),
        "reconnect_count": runtime_metrics.get("reconnect_count", 0),
        "exit_code": runtime_metrics.get("exit_code", None),
        "duration_s": runtime_metrics.get("duration_s", 0.0),
    }

    return summary


def summary_to_csv(summary: dict) -> str:
    """Flatten summary into a single-row CSV string."""
    flat: dict[str, Any] = {}

    # Host metrics
    for metric in ("cpu", "gpu", "ram", "power"):
        host = summary.get("host", {}).get(metric, {})
        for stat in ("avg", "p95", "peak"):
            flat[f"{metric}_{stat}"] = host.get(stat, 0.0)

    # Runtime metrics
    rt = summary.get("runtime", {})
    for key in (
        "frames_processed", "inference_avg_ms", "inference_p95_ms",
        "frame_time_avg_ms", "frame_time_p95_ms", "reconnect_count",
        "exit_code", "duration_s",
    ):
        flat[key] = rt.get(key, "")

    # Metadata
    meta = summary.get("metadata", {})
    for key in ("mode", "video", "runner", "docker_image", "commit"):
        if key in meta:
            flat[f"meta_{key}"] = meta[key]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return buf.getvalue()


def summary_to_markdown(summary: dict) -> str:
    """Render a compact markdown summary for GitHub Actions step summary."""
    host = summary.get("host", {})
    rt = summary.get("runtime", {})
    meta = summary.get("metadata", {})

    lines = ["## Jetson Benchmark Summary\n"]

    if meta:
        lines.append(f"**Mode:** {meta.get('mode', 'unknown')}  ")
        if "video" in meta:
            lines.append(f"**Video:** `{meta['video']}`  ")
        if "commit" in meta:
            lines.append(f"**Commit:** `{meta['commit'][:12]}`  ")
        lines.append("")

    lines.append("### Host Metrics\n")
    lines.append("| Metric | Avg | P95 | Peak | Unit |")
    lines.append("|--------|-----|-----|------|------|")
    for name in ("cpu", "gpu", "ram", "power"):
        m = host.get(name, {})
        lines.append(
            f"| {name.upper()} | {m.get('avg', 0):.1f} | "
            f"{m.get('p95', 0):.1f} | {m.get('peak', 0):.1f} | "
            f"{m.get('unit', '')} |"
        )

    lines.append("")
    lines.append("### Runtime Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Frames processed | {rt.get('frames_processed', 0)} |")
    lines.append(f"| Inference avg | {rt.get('inference_avg_ms', 0):.1f} ms |")
    lines.append(f"| Inference p95 | {rt.get('inference_p95_ms', 0):.1f} ms |")
    lines.append(f"| Frame time avg | {rt.get('frame_time_avg_ms', 0):.1f} ms |")
    lines.append(f"| Frame time p95 | {rt.get('frame_time_p95_ms', 0):.1f} ms |")
    lines.append(f"| Reconnect count | {rt.get('reconnect_count', 0)} |")
    lines.append(f"| Duration | {rt.get('duration_s', 0):.1f} s |")

    exit_code = rt.get("exit_code")
    status = "OK" if exit_code == 0 else f"FAIL (exit {exit_code})"
    lines.append(f"| Exit status | {status} |")

    lines.append("")
    return "\n".join(lines)


def write_summaries(
    summary: dict,
    output_dir: str | Path,
    prefix: str = "summary",
) -> dict[str, str]:
    """Write summary.json, summary.csv, summary.md. Return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    json_path = output_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    paths["json"] = str(json_path)

    csv_path = output_dir / f"{prefix}.csv"
    csv_path.write_text(summary_to_csv(summary))
    paths["csv"] = str(csv_path)

    md_path = output_dir / f"{prefix}.md"
    md_path.write_text(summary_to_markdown(summary))
    paths["md"] = str(md_path)

    return paths


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate benchmark summary")
    ap.add_argument("--tegra-json", required=True,
                    help="Path to tegrastats summary JSON")
    ap.add_argument("--runtime-json", required=True,
                    help="Path to runtime metrics JSON")
    ap.add_argument("--output-dir", default="./artifacts")
    ap.add_argument("--metadata-json", default=None,
                    help="Optional metadata JSON file")
    args = ap.parse_args()

    tegra = json.loads(Path(args.tegra_json).read_text())
    runtime = json.loads(Path(args.runtime_json).read_text())
    metadata = None
    if args.metadata_json:
        metadata = json.loads(Path(args.metadata_json).read_text())

    summary = build_summary(tegra, runtime, metadata)
    paths = write_summaries(summary, args.output_dir)
    print(f"Summaries written: {paths}")

    # Print markdown to stdout for CI
    print(summary_to_markdown(summary))


if __name__ == "__main__":
    main()
