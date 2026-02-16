#!/usr/bin/env python3
"""Estimate GPU idle rate by sampling utilization.

The PEACE paper's Motivation section reports GPU idle rate for different
scheduling strategies (e.g., FIFO vs Reservation).

The paper defines GPU idle rate as:

  idle_rate = (sum_i g_i^I) / (sum_i (g_i^E + g_i^I))

where g_i^E is execution time on GPU i and g_i^I is idle time on GPU i.

This script approximates execution/idle time by periodically sampling
`utilization.gpu` via `nvidia-smi`:

- If utilization >= --busy-threshold, the sample counts as "busy".
- Otherwise it counts as "idle".

This is a practical, hardware-portable approximation that does not require
profilers.

Output:
- A single JSON file containing per-GPU and aggregate idle rates.

All code and comments are English-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GpuAccum:
    busy_s: float = 0.0
    idle_s: float = 0.0


def _query_utils() -> Dict[int, int]:
    """Return {gpu_index: utilization_percent} using nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    res: Dict[int, int] = {}
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Expected format: "0, 12"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        idx = int(parts[0])
        util = int(parts[1])
        res[idx] = util
    return res


def _discover_gpus() -> List[int]:
    utils = _query_utils()
    return sorted(utils.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample GPU utilization and estimate idle rate.")
    p.add_argument("--output", type=str, required=True, help="Output JSON path.")
    p.add_argument("--interval-s", type=float, default=0.2, help="Sampling interval in seconds.")
    p.add_argument("--duration-s", type=float, default=None, help="Optional duration; otherwise run until Ctrl+C.")
    p.add_argument(
        "--busy-threshold",
        type=int,
        default=1,
        help="Utilization >= threshold counts as busy; below counts as idle.",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="GPU indices to monitor (default: all visible GPUs).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    gpus = args.gpus if args.gpus else _discover_gpus()
    if not gpus:
        raise SystemExit("No GPUs detected via nvidia-smi.")

    acc: Dict[int, GpuAccum] = {int(i): GpuAccum() for i in gpus}

    interval = float(args.interval_s)
    if interval <= 0:
        raise SystemExit("--interval-s must be > 0")

    start_wall = time.time()
    last_wall = start_wall
    samples = 0

    print(f"[gpu_idle_monitor] Monitoring GPUs: {gpus}")
    print(f"[gpu_idle_monitor] interval_s={interval}, busy_threshold={args.busy_threshold}")
    if args.duration_s is None:
        print("[gpu_idle_monitor] Running until Ctrl+C...")
    else:
        print(f"[gpu_idle_monitor] duration_s={args.duration_s}")

    try:
        while True:
            now = time.time()
            elapsed = now - start_wall
            if args.duration_s is not None and elapsed >= float(args.duration_s):
                break

            try:
                utils = _query_utils()
            except Exception as e:
                print(f"[gpu_idle_monitor] nvidia-smi query failed: {e}")
                time.sleep(interval)
                continue

            # Use the actual time since last sample to reduce drift.
            dt = now - last_wall
            last_wall = now
            if dt <= 0:
                dt = interval

            for gid in gpus:
                util = int(utils.get(int(gid), 0))
                if util >= int(args.busy_threshold):
                    acc[int(gid)].busy_s += dt
                else:
                    acc[int(gid)].idle_s += dt

            samples += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        pass

    end_wall = time.time()

    per_gpu = {}
    total_busy = 0.0
    total_idle = 0.0
    for gid in gpus:
        a = acc[int(gid)]
        total = a.busy_s + a.idle_s
        idle_rate = (a.idle_s / total) if total > 0 else 0.0
        per_gpu[str(gid)] = {
            "busy_s": a.busy_s,
            "idle_s": a.idle_s,
            "idle_rate": idle_rate,
        }
        total_busy += a.busy_s
        total_idle += a.idle_s

    denom = total_busy + total_idle
    cluster_idle_rate = (total_idle / denom) if denom > 0 else 0.0

    out = {
        "gpus": gpus,
        "interval_s": interval,
        "busy_threshold": int(args.busy_threshold),
        "samples": samples,
        "start_time": start_wall,
        "end_time": end_wall,
        "duration_s": end_wall - start_wall,
        "per_gpu": per_gpu,
        "total_busy_s": total_busy,
        "total_idle_s": total_idle,
        "cluster_idle_rate": cluster_idle_rate,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[gpu_idle_monitor] Wrote: {out_path}")
    print(f"[gpu_idle_monitor] Cluster idle rate: {cluster_idle_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
