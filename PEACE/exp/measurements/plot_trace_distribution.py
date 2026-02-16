#!/usr/bin/env python3
"""Plot request-length distributions for the PEACE paper (Motivation section).

This script reproduces the *Motivation → Request Length Distribution* plots
in `paper_ipdps26_camera_ready.tex`:

- Input length distribution (ContextTokens)
- Output length distribution (GeneratedTokens)

It reads the raw Azure LLM inference trace CSV (no long-input resampling).

Outputs (saved under --out-dir):
- distribution_input.pdf
- distribution_output.pdf

Notes:
- The paper bins input length at 250 tokens and output length at 50 tokens.
- Y-axis is "Proportion" (normalized histogram).

All code and comments are English-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hist_proportion(values: np.ndarray, bin_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_left_edges, proportions) for a simple fixed-width histogram."""
    v = np.asarray(values, dtype=np.int64)
    v = v[np.isfinite(v)]
    v = np.maximum(v, 0)
    if v.size == 0:
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)

    vmax = int(v.max())
    # Make sure the last bin includes vmax.
    upper = ((vmax // bin_size) + 1) * bin_size
    bins = np.arange(0, upper + bin_size, bin_size, dtype=np.int64)

    counts, edges = np.histogram(v, bins=bins)
    total = float(np.sum(counts))
    if total <= 0:
        props = np.zeros_like(counts, dtype=np.float64)
    else:
        props = counts.astype(np.float64) / total

    left_edges = edges[:-1]
    return left_edges, props


def _plot_bar_distribution(
    left_edges: np.ndarray,
    proportions: np.ndarray,
    xlabel: str,
    out_path: Path,
    tick_every: int,
    tick_label_scale: int,
) -> None:
    """Plot a bar chart for a proportion histogram."""
    bin_width = int(left_edges[1] - left_edges[0]) if left_edges.size > 1 else 1

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(left_edges, proportions, width=bin_width, align="edge")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # Reduce tick density (similar to the paper figures).
    if left_edges.size > 1:
        idx = list(range(0, len(left_edges), int(tick_every)))
        positions = [int(left_edges[i] + bin_width / 2) for i in idx]
        labels = [str(int((i) * tick_label_scale)) for i in idx]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot input/output length distributions from Azure LLM inference trace.")
    p.add_argument("--input-csv", type=str, required=True, help="Path to Azure trace CSV.")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for PDFs.")
    p.add_argument("--input-bin", type=int, default=250, help="Bin size for input length histogram (tokens).")
    p.add_argument("--output-bin", type=int, default=50, help="Bin size for output length histogram (tokens).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = {"ContextTokens", "GeneratedTokens"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {sorted(missing)}")

    ctx = df["ContextTokens"].to_numpy(dtype=np.int64)
    gen = df["GeneratedTokens"].to_numpy(dtype=np.int64)

    # Input distribution (paper uses ~250-token bins; tick labels in multiples of 250).
    x_in, p_in = _hist_proportion(ctx, bin_size=int(args.input_bin))
    _plot_bar_distribution(
        left_edges=x_in,
        proportions=p_in,
        xlabel="Input length",
        out_path=out_dir / "distribution_input.pdf",
        tick_every=4,
        tick_label_scale=int(args.input_bin),
    )

    # Output distribution (paper uses ~50-token bins; tick labels in multiples of 50).
    x_out, p_out = _hist_proportion(gen, bin_size=int(args.output_bin))
    _plot_bar_distribution(
        left_edges=x_out,
        proportions=p_out,
        xlabel="Output length",
        out_path=out_dir / "distribution_output.pdf",
        tick_every=4,
        tick_label_scale=int(args.output_bin),
    )

    print(f"Wrote: {out_dir / 'distribution_input.pdf'}")
    print(f"Wrote: {out_dir / 'distribution_output.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
