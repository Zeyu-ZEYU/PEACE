#!/usr/bin/env python3
"""
Analyze PEACE ablation experiment summaries and generate plots.

Expected layout:
  results_root/
    peace/summary.json
    pe/summary.json
    dis/summary.json
    col/summary.json
    fsp/summary.json

This tool reads the summaries and produces:
- a line plot of short-request queue delay percentiles across variants
- a bar chart of short-request throughput (RPS)
- a bar chart of long-request average JCT
- a CSV table for quick comparison

Normalization:
If --baseline-summary is provided, queue delay percentiles are divided by the
baseline's corresponding percentiles (e.g., a short-only baseline run).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from metrics import PERCENTILES


VARIANT_ORDER = ["peace", "pe", "dis", "col", "fsp"]


def _read_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ablation summaries and plot figures.")
    p.add_argument("--results-root", type=str, required=True, help="Root directory containing variant subdirs.")
    p.add_argument("--baseline-summary", type=str, default=None, help="Optional baseline summary.json for normalization.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for figures and CSV.")
    return p.parse_args()


def _get_percentiles(summary: Dict[str, Any], field: str) -> List[float]:
    d = summary.get(field, {}) or {}
    out = []
    for p in PERCENTILES:
        out.append(float(d.get(f"p{p}", float("nan"))))
    return out


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, Any]] = {}
    for key in VARIANT_ORDER:
        path = results_root / key / "summary.json"
        if not path.exists():
            print(f"[warn] Missing summary: {path}")
            continue
        summaries[key] = _read_summary(path)

    baseline = _read_summary(Path(args.baseline_summary)) if args.baseline_summary else None
    baseline_q = _get_percentiles(baseline, "short_queue_delay_s") if baseline else None

    # Build table rows.
    rows = []
    for key in VARIANT_ORDER:
        if key not in summaries:
            continue
        s = summaries[key]
        q_ps = _get_percentiles(s, "short_queue_delay_s")
        if baseline_q:
            q_ps = [q / b if b and not np.isnan(b) else np.nan for q, b in zip(q_ps, baseline_q)]
        thr = float(s.get("short_throughput_rps", float("nan")))
        long_jct_mean = float((s.get("long_jct_s") or {}).get("mean", float("nan")))
        preempt = s.get("long_preemptions_total")

        rows.append({
            "variant": key,
            "queue_delay_p99": q_ps[-1],
            "short_throughput_rps": thr,
            "long_jct_mean_s": long_jct_mean,
            "long_preemptions_total": preempt if preempt is not None else "",
        })

    # Write CSV.
    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    # Plot: queue delay percentiles (line plot).
    fig1 = out_dir / "queue_delay_percentiles.png"
    plt.figure()
    for key in VARIANT_ORDER:
        if key not in summaries:
            continue
        s = summaries[key]
        q_ps = _get_percentiles(s, "short_queue_delay_s")
        if baseline_q:
            q_ps = [q / b if b and not np.isnan(b) else np.nan for q, b in zip(q_ps, baseline_q)]
        plt.plot(PERCENTILES, q_ps, marker="o", label=key)
    plt.xlabel("Percentile")
    plt.ylabel("Normalized queue delay" if baseline_q else "Queue delay (s)")
    plt.title("Short-request queue delay percentiles (ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    # Plot: short throughput.
    fig2 = out_dir / "short_throughput_rps.png"
    plt.figure()
    keys = [k for k in VARIANT_ORDER if k in summaries]
    vals = [float(summaries[k].get("short_throughput_rps", float("nan"))) for k in keys]
    plt.bar(keys, vals)
    plt.xlabel("Variant")
    plt.ylabel("Short throughput (req/s)")
    plt.title("Short-request throughput (ablation)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    # Plot: long JCT mean.
    fig3 = out_dir / "long_jct_avg.png"
    plt.figure()
    keys = [k for k in VARIANT_ORDER if k in summaries]
    vals = [float((summaries[k].get("long_jct_s") or {}).get("mean", float("nan"))) for k in keys]
    plt.bar(keys, vals)
    plt.xlabel("Variant")
    plt.ylabel("Long-request mean JCT (s)")
    plt.title("Long-request average JCT (ablation)")
    plt.tight_layout()
    plt.savefig(fig3, dpi=200)
    plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {fig1}")
    print(f"Wrote: {fig2}")
    print(f"Wrote: {fig3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
