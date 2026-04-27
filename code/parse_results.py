#!/usr/bin/env python3
"""
Unified log parser for all GNN training methods.
Extracts energy, timing, and throughput metrics from run logs.

All methods are expected to output these standardized lines:
  Part {rank}: Total GPU energy consumed: {value}J
  Part {rank}: Total CPU energy consumed: {value}J
  Part {rank}: Total energy consumed: {value}J
  Part {rank} Epoch {epoch}: Time {time}s
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_log(log_path):
    """Parse a run log and extract all metrics."""
    metrics = {
        "parts": {},  # per-partition metrics
        "epochs": [],  # per-epoch timing
    }

    if not Path(log_path).exists():
        return metrics

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Total energy lines: "Part {rank}: Total GPU energy consumed: {val}J"
        m = re.search(r"Part (\d+): Total GPU energy consumed: ([\d.]+)J", line)
        if m:
            rank = int(m.group(1))
            metrics["parts"].setdefault(rank, {})["gpu_energy_j"] = float(m.group(2))
            continue

        m = re.search(r"Part (\d+): Total CPU energy consumed: ([\d.]+)J", line)
        if m:
            rank = int(m.group(1))
            metrics["parts"].setdefault(rank, {})["cpu_energy_j"] = float(m.group(2))
            continue

        m = re.search(r"Part (\d+): Total energy consumed: ([\d.]+)J", line)
        if m:
            rank = int(m.group(1))
            metrics["parts"].setdefault(rank, {})["total_energy_j"] = float(m.group(2))
            continue

        # Epoch timing: "Part {rank} Epoch {epoch}: Time {time}s"
        # Also handles "Part {rank} Epoch {epoch:02d}: Time {time:.2f}s"
        m = re.search(r"Part (\d+) Epoch (\d+):\s*Time ([\d.]+)s", line)
        if m:
            metrics["epochs"].append({
                "part": int(m.group(1)),
                "epoch": int(m.group(2)),
                "time_s": float(m.group(3)),
            })
            continue

        # Trainer format: "Part {rank} Ep{epoch}: {time}s GPU=..."
        m = re.search(r"Part (\d+) Ep(?:och)?\s*(\d+):\s*([\d.]+)s", line)
        if m:
            metrics["epochs"].append({
                "part": int(m.group(1)),
                "epoch": int(m.group(2)),
                "time_s": float(m.group(3)),
            })
            continue

        # Also match "Part {rank}, Epoch Time(s): {val}" format (default.py)
        m = re.search(r"Part (\d+), Epoch Time\(s\): ([\d.]+)", line)
        if m:
            metrics["epochs"].append({
                "part": int(m.group(1)),
                "epoch": -1,  # epoch number not in this format
                "time_s": float(m.group(2)),
            })
            continue

        # BGL format: "[EPOCH] Part {rank} Epoch {epoch}: ... Time={val}s ..."
        m = re.search(r"\[EPOCH\] Part (\d+) Epoch (\d+):.*Time=([\d.]+)s", line)
        if m:
            metrics["epochs"].append({
                "part": int(m.group(1)),
                "epoch": int(m.group(2)),
                "time_s": float(m.group(3)),
            })
            continue

        # RapidGNN v1 format: "[METRICS] Part {rank} Epoch {epoch}: Completed in {val}s"
        m = re.search(r"\[METRICS\] Part (\d+) Epoch (\d+): Completed in ([\d.]+)s", line)
        if m:
            metrics["epochs"].append({
                "part": int(m.group(1)),
                "epoch": int(m.group(2)),
                "time_s": float(m.group(3)),
            })
            continue

        # Graphstorm format: "Epoch {epoch} take {val} seconds"
        m = re.search(r"Epoch (\d+) take ([\d.]+) seconds", line)
        if m:
            metrics["epochs"].append({
                "part": 0,
                "epoch": int(m.group(1)),
                "time_s": float(m.group(2)),
            })
            continue

        # Avg epoch time: "Part {rank}: Avg epoch time: {val}s"
        m = re.search(r"Part (\d+): Avg epoch time: ([\d.]+)s", line)
        if m:
            rank = int(m.group(1))
            metrics["parts"].setdefault(rank, {})["avg_epoch_time_s"] = float(m.group(2))
            continue

        # Presample time: "Part {rank}: Total presample time: {val}s"
        m = re.search(r"Part (\d+): Total presample time: ([\d.]+)s", line)
        if m:
            rank = int(m.group(1))
            metrics["parts"].setdefault(rank, {})["presample_time_s"] = float(m.group(2))
            continue

    # Compute aggregates
    if metrics["parts"]:
        all_gpu = [p.get("gpu_energy_j", 0) for p in metrics["parts"].values()]
        all_cpu = [p.get("cpu_energy_j", 0) for p in metrics["parts"].values()]
        all_total = [p.get("total_energy_j", 0) for p in metrics["parts"].values()]

        metrics["aggregate"] = {
            "sum_gpu_energy_j": sum(all_gpu),
            "sum_cpu_energy_j": sum(all_cpu),
            "sum_total_energy_j": sum(all_total),
            "mean_gpu_energy_j": sum(all_gpu) / len(all_gpu) if all_gpu else 0,
            "mean_cpu_energy_j": sum(all_cpu) / len(all_cpu) if all_cpu else 0,
            "mean_total_energy_j": sum(all_total) / len(all_total) if all_total else 0,
            "num_parts": len(metrics["parts"]),
        }

    if metrics["epochs"]:
        # Get epoch times for part 0 (representative)
        part0_epochs = [e["time_s"] for e in metrics["epochs"] if e["part"] == 0]
        if part0_epochs:
            # Use last 80% of epochs for stable average
            stable_start = max(0, len(part0_epochs) - int(len(part0_epochs) * 0.8))
            stable_times = part0_epochs[stable_start:]
            metrics["aggregate"] = metrics.get("aggregate", {})
            metrics["aggregate"]["avg_epoch_time_s"] = sum(stable_times) / len(stable_times) if stable_times else 0
            metrics["aggregate"]["total_epochs"] = len(part0_epochs)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parse GNN training logs")
    parser.add_argument("--log_file", required=True, help="Path to run.log")
    parser.add_argument("--output", required=True, help="Path to output metrics.json")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--batch_size", default="unknown")
    args = parser.parse_args()

    metrics = parse_log(args.log_file)
    metrics["meta"] = {
        "method": args.method,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "log_file": args.log_file,
    }

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    agg = metrics.get("aggregate", {})
    if agg:
        print(f"[{args.method}] {args.dataset} B={args.batch_size}: "
              f"GPU={agg.get('sum_gpu_energy_j', 0):.1f}J "
              f"CPU={agg.get('sum_cpu_energy_j', 0):.1f}J "
              f"Total={agg.get('sum_total_energy_j', 0):.1f}J "
              f"AvgEpoch={agg.get('avg_epoch_time_s', 0):.2f}s")
    else:
        print(f"[{args.method}] {args.dataset} B={args.batch_size}: No metrics found", file=sys.stderr)


if __name__ == "__main__":
    main()
