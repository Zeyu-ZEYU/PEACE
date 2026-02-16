#!/usr/bin/env python3
"""Trace utilities for PEACE motivation measurements (real experiments).

This folder focuses on the *Motivation* section of the IPDPS'26 PEACE paper.

The Motivation experiments require two types of traces:

1) **Raw Azure trace** (CSV)
   Used to plot the request length distributions.

2) **Paper-style prepared trace** (JSONL)
   Used for real workload replay against OpenAI-compatible servers.
   The prepared trace:
   - Preserves the original arrival pattern and output-length distribution.
   - Classifies requests as short/long using the paper threshold (default: 4K).
   - Rescales long requests into the paper long range (default: 100K–500K).

This script wraps the ablation trace preparation logic and adds a few
lightweight operations (filter/split) that are convenient for Motivation
experiments.

All code and comments are English-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def _ablation_dir() -> Path:
    exp_dir = Path(__file__).resolve().parents[1]
    ablation = exp_dir / "ablation_ipdps26"
    if not ablation.exists():
        raise FileNotFoundError(f"Expected ablation harness at: {ablation}")
    return ablation


def _import_prepare_azure_trace():
    """Import prepare_azure_trace from the ablation harness."""
    ablation = _ablation_dir()
    sys.path.insert(0, str(ablation))
    from trace_utils import prepare_azure_trace  # type: ignore

    return prepare_azure_trace


def _read_jsonl(path: Path, max_lines: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= int(max_lines):
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def cmd_prepare(args: argparse.Namespace) -> int:
    prepare_azure_trace = _import_prepare_azure_trace()
    prepare_azure_trace(
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        short_threshold=int(args.short_threshold),
        long_range=(int(args.long_range[0]), int(args.long_range[1])),
        max_requests=args.max_requests,
        seed=int(args.seed),
    )
    print(f"Wrote prepared trace: {args.output_jsonl}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    inp = Path(args.input_jsonl)
    out = Path(args.output_jsonl)

    keep = args.keep_type.strip().lower()
    if keep not in ("short", "long"):
        raise SystemExit("--keep-type must be one of: short, long")

    recs = _read_jsonl(inp, max_lines=args.max_requests)
    kept = [r for r in recs if str(r.get("request_type", "")).lower() == keep]

    # Optional: reindex request_id to keep things neat.
    if args.reindex:
        for i, r in enumerate(kept):
            r["request_id"] = i

    _write_jsonl(out, kept)
    print(f"Read : {inp} ({len(recs)} records)")
    print(f"Wrote: {out} ({len(kept)} records kept: {keep})")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    inp = Path(args.input_jsonl)
    out_short = Path(args.output_short)
    out_long = Path(args.output_long)

    recs = _read_jsonl(inp, max_lines=args.max_requests)
    short = [r for r in recs if str(r.get("request_type", "")).lower() == "short"]
    long = [r for r in recs if str(r.get("request_type", "")).lower() == "long"]

    if args.reindex:
        for i, r in enumerate(short):
            r["request_id"] = i
        for i, r in enumerate(long):
            r["request_id"] = i

    _write_jsonl(out_short, short)
    _write_jsonl(out_long, long)

    print(f"Read : {inp} ({len(recs)} records)")
    print(f"Wrote: {out_short} ({len(short)} short records)")
    print(f"Wrote: {out_long} ({len(long)} long records)")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trace operations for PEACE motivation measurements.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare Azure CSV into paper-style JSONL (with long resampling).")
    p_prep.add_argument("--input-csv", type=str, required=True)
    p_prep.add_argument("--output-jsonl", type=str, required=True)
    p_prep.add_argument("--short-threshold", type=int, default=4096)
    p_prep.add_argument("--long-range", type=int, nargs=2, default=[100000, 500000], metavar=("MIN", "MAX"))
    p_prep.add_argument("--max-requests", type=int, default=None)
    p_prep.add_argument("--seed", type=int, default=0)
    p_prep.set_defaults(_handler=cmd_prepare)

    p_f = sub.add_parser("filter", help="Filter a prepared JSONL trace by request_type.")
    p_f.add_argument("--input-jsonl", type=str, required=True)
    p_f.add_argument("--output-jsonl", type=str, required=True)
    p_f.add_argument("--keep-type", type=str, required=True, help="short or long")
    p_f.add_argument("--max-requests", type=int, default=None)
    p_f.add_argument("--reindex", action="store_true", help="Reset request_id to be contiguous.")
    p_f.set_defaults(_handler=cmd_filter)

    p_s = sub.add_parser("split", help="Split a prepared JSONL trace into short-only and long-only traces.")
    p_s.add_argument("--input-jsonl", type=str, required=True)
    p_s.add_argument("--output-short", type=str, required=True)
    p_s.add_argument("--output-long", type=str, required=True)
    p_s.add_argument("--max-requests", type=int, default=None)
    p_s.add_argument("--reindex", action="store_true", help="Reset request_id to be contiguous.")
    p_s.set_defaults(_handler=cmd_split)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    handler = getattr(args, "_handler", None)
    if handler is None:
        raise RuntimeError("No handler registered")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
