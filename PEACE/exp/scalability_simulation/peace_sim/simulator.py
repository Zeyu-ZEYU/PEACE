from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .cluster import ClusterState, ClusterTopology
from .hyperparams import ConfiguringRequest, HeuristicConfigurator, HyperParamConfigurator, PickleRegressorConfigurator
from .profiles import ProfileDB
from .selector import ModelSpec, choose_sp_architecture, gpus_for_decode, gpus_for_prefill
from .types import HyperParams, Request, RequestResult


@dataclass(frozen=True)
class SimulationConfig:
    method: str  # 'sarathi' or 'peace'
    model: ModelSpec
    dataset: str

    # Cluster
    num_gpus_total: int = 10_000
    gpus_per_server: int = 8
    servers_per_rack: int = 32

    # PEACE-specific: split total GPUs into prefill/decode pools
    peace_prefill_gpu_fraction: float = 0.2

    # Baseline-specific: fixed GPUs per request (TP*PP)
    sarathi_gpus_per_request: int = 8

    # Hyper-parameter configuring
    configuring_policy: str = "heuristic"  # 'heuristic' or 'pickle_regressor'
    pickle_regressor_path: Optional[str] = None
    accuracy_degradation_constraint: float = 0.0
    ttft_constraint_s: Optional[float] = None
    tbt_constraint_s: Optional[float] = None

    # Optional KV migration modeling
    kv_migration_metric: str = "kv_migration_s"  # if present in profiles, used directly
    kv_migration_default_s: float = 0.0


def _make_configurator(cfg: SimulationConfig) -> Optional[HyperParamConfigurator]:
    if cfg.method != "peace":
        return None

    if cfg.configuring_policy == "heuristic":
        return HeuristicConfigurator()
    if cfg.configuring_policy == "pickle_regressor":
        if not cfg.pickle_regressor_path:
            raise ValueError("pickle_regressor_path is required for configuring_policy=pickle_regressor")
        return PickleRegressorConfigurator(cfg.pickle_regressor_path)
    raise ValueError(f"Unknown configuring_policy: {cfg.configuring_policy}")


def _estimate(profile: ProfileDB, method: str, model: str, dataset: str, metric: str, x: float, default: Optional[float] = None) -> float:
    try:
        return profile.estimate(method, model, dataset, metric, x)
    except KeyError:
        if default is None:
            raise
        return default


def run_simulation(trace: List[Request], profile: ProfileDB, cfg: SimulationConfig) -> List[RequestResult]:
    """Run a trace-driven simulation.

    This is a lightweight scheduler:
    - Sarathi: one GPU pool
    - PEACE: split GPU pool into prefill and decode pools

    Durations are taken from `profile` via piecewise interpolation.
    """

    method = cfg.method
    if method not in {"sarathi", "peace"}:
        raise ValueError("cfg.method must be 'sarathi' or 'peace'")

    topo = ClusterTopology(
        num_gpus_total=cfg.num_gpus_total,
        gpus_per_server=cfg.gpus_per_server,
        servers_per_rack=cfg.servers_per_rack,
    )

    configurator = _make_configurator(cfg)

    if method == "sarathi":
        pool = ClusterState(topo)
        prefill_pool = pool
        decode_pool = pool
    else:
        # Split GPUs into two clusters by slicing global GPU indices.
        prefill_gpus = max(1, int(round(cfg.num_gpus_total * cfg.peace_prefill_gpu_fraction)))
        decode_gpus = max(1, cfg.num_gpus_total - prefill_gpus)

        prefill_pool = ClusterState(
            ClusterTopology(
                num_gpus_total=prefill_gpus,
                gpus_per_server=cfg.gpus_per_server,
                servers_per_rack=cfg.servers_per_rack,
            )
        )
        decode_pool = ClusterState(
            ClusterTopology(
                num_gpus_total=decode_gpus,
                gpus_per_server=cfg.gpus_per_server,
                servers_per_rack=cfg.servers_per_rack,
            )
        )

    results: List[RequestResult] = []

    for req in trace:
        # Decide GPU usage.
        if method == "sarathi":
            prefill_gpus = cfg.sarathi_gpus_per_request
            decode_gpus = cfg.sarathi_gpus_per_request
            arch = None
            hyper: Optional[HyperParams] = None
        else:
            arch = choose_sp_architecture(req.input_len, cfg.model)
            prefill_gpus = gpus_for_prefill(req.input_len, arch, cfg.model)
            decode_gpus = gpus_for_decode(req.input_len, arch, cfg.model)

            # Hyper-parameter configuring.
            cr = ConfiguringRequest(
                input_len=req.input_len,
                accuracy_degradation_constraint=cfg.accuracy_degradation_constraint,
                ttft_constraint_s=cfg.ttft_constraint_s,
                tbt_constraint_s=cfg.tbt_constraint_s,
            )
            hyper = configurator.configure(cr) if configurator else None

        # Profile-driven metrics.
        ttft_s = _estimate(profile, method, cfg.model.name, cfg.dataset, "ttft_s", req.input_len)
        tbt_s = _estimate(profile, method, cfg.model.name, cfg.dataset, "tbt_s", req.input_len)
        response_s = _estimate(profile, method, cfg.model.name, cfg.dataset, "response_time_s", req.input_len)

        # Occupancy approximations.
        prefill_dur_s = max(0.0, ttft_s)
        decode_dur_s = max(0.0, response_s - ttft_s)

        # Prefill scheduling.
        prefill_start, prefill_end, _ = prefill_pool.allocate(
            num_gpus=prefill_gpus,
            earliest_s=req.arrival_s,
            duration_s=prefill_dur_s,
        )

        # KV migration (optional).
        mig_s = _estimate(
            profile,
            method,
            cfg.model.name,
            cfg.dataset,
            cfg.kv_migration_metric,
            req.input_len,
            default=cfg.kv_migration_default_s,
        )

        # Decode scheduling.
        decode_start, decode_end, _ = decode_pool.allocate(
            num_gpus=decode_gpus,
            earliest_s=prefill_end + mig_s,
            duration_s=decode_dur_s,
        )

        results.append(
            RequestResult(
                req=req,
                method=method,
                model=cfg.model.name,
                ttft_s=ttft_s,
                tbt_s=tbt_s,
                response_time_s=response_s,
                prefill_start_s=prefill_start,
                prefill_end_s=prefill_end,
                decode_start_s=decode_start,
                decode_end_s=decode_end,
                prefill_gpus=prefill_gpus,
                decode_gpus=decode_gpus,
                hyperparams=hyper,
            )
        )

    return results
