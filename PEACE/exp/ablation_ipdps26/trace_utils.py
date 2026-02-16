#!/usr/bin/env python3
"""
Utilities for preparing workload traces for PEACE ablation experiments.

The IPDPS'26 paper uses the Azure LLM inference trace to derive arrivals and
context/generated token distributions, and then *rescales* long requests to
represent long-input workloads (e.g., 100K–500K tokens) while preserving the
original shape of the long-input portion of the trace.

Azure trace schema (see PEACE/trace/AzureLLMInferenceDataset2024.md):
- TIMESTAMP
- ContextTokens
- GeneratedTokens

This tool converts the raw CSV into a JSONL trace used by `run_workload.py`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TraceRecord:
    """A single request record in the prepared trace."""
    request_id: int
    arrival_time_s: float
    prompt_tokens: int
    max_new_tokens: int
    request_type: str  # "short" or "long"


def _infer_timestamp_unit_scale(values: np.ndarray) -> float:
    """
    Infer a scaling factor to convert epoch-like numeric timestamps to seconds.

    Heuristic based on magnitude:
      ~1e9   => seconds
      ~1e12  => milliseconds
      ~1e15  => microseconds
      ~1e18  => nanoseconds
    """
    vmax = float(np.nanmax(values))
    if vmax >= 1e17:
        return 1e9  # ns -> s
    if vmax >= 1e14:
        return 1e6  # us -> s
    if vmax >= 1e11:
        return 1e3  # ms -> s
    return 1.0  # s -> s


def _to_relative_seconds(ts: pd.Series) -> np.ndarray:
    """Convert TIMESTAMP column into seconds relative to the first request."""
    if np.issubdtype(ts.dtype, np.number):
        arr = ts.to_numpy(dtype=np.float64)
        scale = _infer_timestamp_unit_scale(arr)
        rel = (arr - np.nanmin(arr)) / scale
        return rel

    # Try parsing as datetime-like strings.
    dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if dt.isna().all():
        raise ValueError("Failed to parse TIMESTAMP as numeric or datetime.")
    rel = (dt - dt.min()).dt.total_seconds().to_numpy(dtype=np.float64)
    return rel


def _linear_rescale(values: np.ndarray, new_min: int, new_max: int) -> np.ndarray:
    """Linearly rescale values to [new_min, new_max] preserving relative shape."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        return np.full_like(values, fill_value=new_min, dtype=np.float64)
    scaled = new_min + (values - vmin) * (new_max - new_min) / (vmax - vmin)
    return scaled


def prepare_azure_trace(
    input_csv: str,
    output_jsonl: str,
    short_threshold: int = 4096,
    long_range: Tuple[int, int] = (100_000, 500_000),
    max_requests: int | None = None,
    seed: int = 0,
) -> None:
    """
    Prepare an Azure trace CSV into a compact JSONL trace.

    Long requests (ContextTokens >= short_threshold) are linearly rescaled into
    long_range to mimic long-input workloads (paper setting: 100K–500K tokens).
    """
    rng = np.random.default_rng(seed)

    df = pd.read_csv(input_csv)
    required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if max_requests is not None:
        df = df.iloc[: int(max_requests)].copy()

    arrival_s = _to_relative_seconds(df["TIMESTAMP"])
    ctx = df["ContextTokens"].to_numpy(dtype=np.int64)
    gen = df["GeneratedTokens"].to_numpy(dtype=np.int64)

    # Ensure non-negative tokens.
    ctx = np.maximum(ctx, 0)
    gen = np.maximum(gen, 0)

    is_long = ctx >= int(short_threshold)

    # Rescale long-input context token lengths.
    if np.any(is_long):
        long_ctx = ctx[is_long].astype(np.float64)
        new_min, new_max = int(long_range[0]), int(long_range[1])
        scaled = _linear_rescale(long_ctx, new_min=new_min, new_max=new_max)

        # Round to integer tokens.
        ctx[is_long] = np.rint(scaled).astype(np.int64)

        # Optional small jitter to avoid many identical lengths after rounding.
        # This keeps the distribution similar but helps caching/bucketing logic.
        jitter = rng.integers(low=-8, high=9, size=int(np.sum(is_long)))
        ctx[is_long] = np.maximum(ctx[is_long] + jitter, new_min)

    # Build and write JSONL.
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for i, (t, in_tok, out_tok) in enumerate(zip(arrival_s, ctx, gen)):
            rec = TraceRecord(
                request_id=i,
                arrival_time_s=float(t),
                prompt_tokens=int(in_tok),
                max_new_tokens=int(out_tok),
                request_type="long" if int(in_tok) >= int(short_threshold) else "short",
            )
            f.write(json.dumps(rec.__dict__) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Azure trace CSV for PEACE experiments.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare an Azure trace CSV into JSONL.")
    p_prep.add_argument("--input-csv", type=str, required=True)
    p_prep.add_argument("--output-jsonl", type=str, required=True)
    p_prep.add_argument("--short-threshold", type=int, default=4096)
    p_prep.add_argument("--long-range", type=int, nargs=2, default=[100000, 500000], metavar=("MIN", "MAX"))
    p_prep.add_argument("--max-requests", type=int, default=None)
    p_prep.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "prepare":
        prepare_azure_trace(
            input_csv=args.input_csv,
            output_jsonl=args.output_jsonl,
            short_threshold=args.short_threshold,
            long_range=(args.long_range[0], args.long_range[1]),
            max_requests=args.max_requests,
            seed=args.seed,
        )
        print(f"Wrote prepared trace: {args.output_jsonl}")
        return 0

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
