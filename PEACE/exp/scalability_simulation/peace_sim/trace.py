from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .types import Request


@dataclass(frozen=True)
class DatasetStats:
    name: str
    n_requests: int
    input_median: int
    input_p99: int
    output_median: int
    output_p99: int


# Defaults pulled from Table 1 in the PEACE paper source.
DATASETS = {
    "needlebench": DatasetStats(
        name="needlebench",
        n_requests=11326,
        input_median=113_000,
        input_p99=995_000,
        output_median=348,
        output_p99=572,
    ),
    "bookcorpus": DatasetStats(
        name="bookcorpus",
        n_requests=7185,
        input_median=97_000,
        input_p99=406_000,
        output_median=375,
        output_p99=631,
    ),
}


def _fit_lognormal_mu_sigma(median: float, p99: float) -> tuple[float, float]:
    """Fit a lognormal using median and 99th percentile.

    For lognormal:
        median = exp(mu)
        p99 = exp(mu + sigma * z99)
    """
    if median <= 0 or p99 <= 0:
        raise ValueError("median/p99 must be positive")
    z99 = 2.3263478740408408
    mu = math.log(median)
    sigma = max(1e-6, (math.log(p99) - mu) / z99)
    return mu, sigma


def generate_synthetic_trace(
    dataset: str,
    n: Optional[int] = None,
    arrival_rate_rps: float = 1.0,
    seed: int = 42,
    min_input: int = 1,
    max_input: Optional[int] = None,
    min_output: int = 1,
    max_output: Optional[int] = None,
) -> List[Request]:
    """Generate a synthetic trace matching rough dataset statistics.

    Arrival times are generated from a Poisson process with the given rate.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}'. Known: {sorted(DATASETS.keys())}")
    if arrival_rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be > 0")

    stats = DATASETS[dataset]
    n = int(n or stats.n_requests)

    rng = random.Random(seed)

    mu_in, sigma_in = _fit_lognormal_mu_sigma(stats.input_median, stats.input_p99)
    mu_out, sigma_out = _fit_lognormal_mu_sigma(stats.output_median, stats.output_p99)

    t = 0.0
    reqs: List[Request] = []
    for i in range(n):
        # Exponential inter-arrival
        t += rng.expovariate(arrival_rate_rps)

        inp = int(round(rng.lognormvariate(mu_in, sigma_in)))
        out = int(round(rng.lognormvariate(mu_out, sigma_out)))

        inp = max(min_input, inp)
        out = max(min_output, out)
        if max_input is not None:
            inp = min(max_input, inp)
        if max_output is not None:
            out = min(max_output, out)

        reqs.append(Request(req_id=i, arrival_s=t, input_len=inp, output_len=out))

    return reqs


def load_trace_csv(path: str) -> List[Request]:
    """Load a trace from CSV.

    Required columns: input_len, output_len
    Optional column: arrival_s (if absent, arrival_s=0 for all)
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        fieldset = set(h.strip() for h in reader.fieldnames)
        if "input_len" not in fieldset or "output_len" not in fieldset:
            raise ValueError("CSV must have columns: input_len, output_len")

        reqs: List[Request] = []
        for idx, row in enumerate(reader):
            inp = int(float(row["input_len"]))
            out = int(float(row["output_len"]))
            arr = float(row.get("arrival_s", 0.0) or 0.0)
            reqs.append(Request(req_id=idx, arrival_s=arr, input_len=inp, output_len=out))

    return reqs


def save_trace_csv(reqs: Iterable[Request], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["req_id", "arrival_s", "input_len", "output_len"])
        for r in reqs:
            writer.writerow([r.req_id, f"{r.arrival_s:.6f}", r.input_len, r.output_len])
