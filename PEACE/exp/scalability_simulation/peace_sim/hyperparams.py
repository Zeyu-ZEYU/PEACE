from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Optional

from .types import HyperParams


@dataclass(frozen=True)
class ConfiguringRequest:
    input_len: int
    accuracy_degradation_constraint: float = 0.0
    ttft_constraint_s: Optional[float] = None
    tbt_constraint_s: Optional[float] = None


class HyperParamConfigurator:
    """Interface for hyper-parameter configuring.

    In the paper, this is solved by regression models trained offline.
    For this simulator, we provide a heuristic policy and an option to
    load a pickled regressor.
    """

    def configure(self, req: ConfiguringRequest) -> HyperParams:
        raise NotImplementedError


class HeuristicConfigurator(HyperParamConfigurator):
    """A simple, transparent policy.

    - For longer prompts, use more CSA levels.
    - If accuracy degradation constraint is 0, disable token dropping.
    - Otherwise, increase drop ratios with distance (higher levels).
    - Choose micro-partitions to roughly increase overlap on longer prompts.

    This is *not* meant to replicate the paper's trained models; it is meant
    to be a sane default that keeps the simulator runnable.
    """

    def configure(self, req: ConfiguringRequest) -> HyperParams:
        s = req.input_len

        if s <= 100_000:
            nv = 2
        elif s <= 400_000:
            nv = 3
        else:
            nv = 4

        drop_by_level = {1: 0.0}
        if req.accuracy_degradation_constraint <= 0.0:
            for v in range(2, nv + 1):
                drop_by_level[v] = 0.0
        else:
            # Map a small allowed degradation to a conservative drop ratio.
            # E.g., 1% -> 0.10, 3% -> 0.25 (capped)
            base = min(0.35, 0.05 + 0.07 * (req.accuracy_degradation_constraint * 100.0))
            keep = 1.0 - base
            for v in range(2, nv + 1):
                # Further levels keep fewer tokens (heuristic).
                keep_v = max(0.05, keep / (2 ** (v - 2)))
                drop_by_level[v] = min(0.95, 1.0 - keep_v)

        # Micro-partitions (pipeline granularity). Level 1 usually higher.
        micro_by_level = {}
        if s <= 64_000:
            micro_by_level[1] = 4
        elif s <= 256_000:
            micro_by_level[1] = 8
        else:
            micro_by_level[1] = 12

        for v in range(2, nv + 1):
            micro_by_level[v] = max(1, micro_by_level[1] // (2 ** (v - 1)))

        return HyperParams(
            num_levels=nv,
            drop_ratio_by_level=drop_by_level,
            micro_partitions_by_level=micro_by_level,
            policy="heuristic",
            accuracy_degradation_constraint=req.accuracy_degradation_constraint,
            ttft_constraint_s=req.ttft_constraint_s,
            tbt_constraint_s=req.tbt_constraint_s,
        )


class PickleRegressorConfigurator(HyperParamConfigurator):
    """Load a pickled regressor with a scikit-learn like `.predict()` API.

    Expected output format of `predict()`:
      [nv, drop_level2, micro_level1]

    The simulator will expand it into per-level configs.

    Notes:
    - No dependency on scikit-learn is required here; we only need the pickle.
    - You are responsible for providing a compatible object.
    """

    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as f:
            self.model: Any = pickle.load(f)

    def configure(self, req: ConfiguringRequest) -> HyperParams:
        x = [
            req.accuracy_degradation_constraint,
            float(req.ttft_constraint_s or -1.0),
            float(req.tbt_constraint_s or -1.0),
            float(req.input_len),
        ]
        pred = self.model.predict([x])[0]
        nv = int(round(pred[0]))
        nv = max(1, min(6, nv))
        drop2 = float(pred[1])
        drop2 = max(0.0, min(0.95, drop2))
        m1 = int(round(pred[2]))
        m1 = max(1, min(32, m1))

        drop_by_level = {1: 0.0}
        keep = 1.0 - drop2
        for v in range(2, nv + 1):
            keep_v = max(0.05, keep / (2 ** (v - 2)))
            drop_by_level[v] = min(0.95, 1.0 - keep_v)

        micro_by_level = {1: m1}
        for v in range(2, nv + 1):
            micro_by_level[v] = max(1, m1 // (2 ** (v - 1)))

        return HyperParams(
            num_levels=nv,
            drop_ratio_by_level=drop_by_level,
            micro_partitions_by_level=micro_by_level,
            policy="pickle_regressor",
            accuracy_degradation_constraint=req.accuracy_degradation_constraint,
            ttft_constraint_s=req.ttft_constraint_s,
            tbt_constraint_s=req.tbt_constraint_s,
        )
