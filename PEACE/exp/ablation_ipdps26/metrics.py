"""
Metrics helpers for PEACE ablation experiments.

All computations are based on the `records.jsonl` format emitted by run_workload.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PERCENTILES = [1, 25, 50, 75, 99]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def percentile_dict(values: Sequence[float], percentiles: Sequence[int] = PERCENTILES) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in percentiles}
    res = np.percentile(arr, list(percentiles)).tolist()
    return {f"p{p}": float(v) for p, v in zip(percentiles, res)}


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_sum_int(values: Sequence[int | float | None]) -> Optional[int]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return int(np.sum(vals))


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize per-request records into aggregate metrics.

    Expected record fields (see run_workload.py):
      - request_type: short|long
      - arrival_time_s
      - start_time_s (client scheduled send time, relative to experiment start)
      - finish_time_s (relative to experiment start)
      - ttft_s
      - latency_s
      - queue_delay_s (optional, server instrumented)
      - max_new_tokens
      - generated_tokens (optional)
      - peace_metrics (optional)
    """
    if not records:
        return {"error": "No records"}

    makespan_s = max(float(r.get("finish_time_s", 0.0)) for r in records)

    short = [r for r in records if r.get("request_type") == "short"]
    long = [r for r in records if r.get("request_type") == "long"]

    def _jct(r: Dict[str, Any]) -> float:
        arrival = float(r["arrival_time_s"])
        finish = float(r["finish_time_s"])
        return finish - arrival

    short_jct = [_jct(r) for r in short]
    long_jct = [_jct(r) for r in long]

    # Queueing delay: prefer server-reported queue_delay_s; fallback to TTFT.
    short_qdelay = []
    short_ttft = []
    for r in short:
        ttft = r.get("ttft_s")
        if isinstance(ttft, (int, float)):
            short_ttft.append(float(ttft))
        qd = r.get("queue_delay_s")
        if isinstance(qd, (int, float)):
            short_qdelay.append(float(qd))
        elif isinstance(ttft, (int, float)):
            short_qdelay.append(float(ttft))

    # Throughput (requests per second) for short requests.
    short_throughput_rps = (len(short) / makespan_s) if makespan_s > 0 else float("nan")

    # Preemptions: best effort from record['peace_metrics']['preemptions'] if present.
    long_preemptions = []
    for r in long:
        pm = r.get("peace_metrics")
        if isinstance(pm, dict) and "preemptions" in pm:
            try:
                long_preemptions.append(int(pm["preemptions"]))
            except Exception:
                pass

    return {
        "num_requests": len(records),
        "num_short": len(short),
        "num_long": len(long),
        "makespan_s": float(makespan_s),
        "short_throughput_rps": float(short_throughput_rps),
        "short_queue_delay_s": {
            "mean": safe_mean(short_qdelay),
            **percentile_dict(short_qdelay),
        },
        "short_ttft_s": {
            "mean": safe_mean(short_ttft),
            **percentile_dict(short_ttft),
        },
        "long_jct_s": {
            "mean": safe_mean(long_jct),
            **percentile_dict(long_jct),
        },
        "short_jct_s": {
            "mean": safe_mean(short_jct),
            **percentile_dict(short_jct),
        },
        "long_preemptions_total": safe_sum_int(long_preemptions),
    }
