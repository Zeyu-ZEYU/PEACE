from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Request:
    """A single inference request in the trace."""

    req_id: int
    arrival_s: float
    input_len: int
    output_len: int


@dataclass(frozen=True)
class HyperParams:
    """A compact config for CSA + pipelining.

    This simulator does not model per-layer/per-head details. Instead it keeps
    per-level drop ratios and per-level micro-partitions, which are enough to
    (a) record that a "configuring" step happened and (b) let a cost model
    optionally adjust compute/communication estimates.
    """

    num_levels: int
    drop_ratio_by_level: Dict[int, float]
    micro_partitions_by_level: Dict[int, int]
    policy: str = "heuristic"
    accuracy_degradation_constraint: float = 0.0
    ttft_constraint_s: Optional[float] = None
    tbt_constraint_s: Optional[float] = None


@dataclass(frozen=True)
class ProfileCurves:
    """Piecewise-linear curves for a metric, keyed by metric name."""

    # metric -> list of (x, y) points. x is typically input_len.
    curves: Dict[str, list[tuple[float, float]]]


@dataclass
class RequestResult:
    req: Request
    method: str
    model: str
    ttft_s: float
    tbt_s: float
    response_time_s: float
    prefill_start_s: float
    prefill_end_s: float
    decode_start_s: float
    decode_end_s: float
    prefill_gpus: int
    decode_gpus: int
    hyperparams: Optional[HyperParams] = None

