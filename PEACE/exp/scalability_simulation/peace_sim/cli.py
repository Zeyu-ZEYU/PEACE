from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Optional

from .metrics import save_results_json, save_summary_json, summarize
from .profiles import load_profile_db
from .selector import ModelSpec
from .simulator import SimulationConfig, run_simulation
from .trace import DATASETS, generate_synthetic_trace, load_trace_csv, save_trace_csv


def _default_model_spec(name: str, tp_size: int) -> ModelSpec:
    # Hidden sizes are approximate; only used for the SP-arch threshold.
    hidden_sizes = {
        "glm-4": 4096,
        "internlm2.5": 4096,
    }
    if name not in hidden_sizes:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(hidden_sizes.keys())}")
    return ModelSpec(name=name, hidden_size=hidden_sizes[name], tp_size=tp_size)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PEACE scalability-test style simulator")

    p.add_argument("--method", choices=["sarathi", "peace"], default="peace")
    p.add_argument("--model", choices=["glm-4", "internlm2.5"], default="glm-4")
    p.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="needlebench")

    p.add_argument("--profile", default=os.path.join("profiles", "sample_profile.json"), help="Path to profile JSON")

    # Trace
    p.add_argument("--trace-csv", default=None, help="CSV trace with columns input_len,output_len,(arrival_s)")
    p.add_argument("--generate-trace", action="store_true", help="Generate a synthetic trace")
    p.add_argument("--n", type=int, default=None, help="Number of requests (overrides dataset default)")
    p.add_argument("--arrival-rate", type=float, default=1.0, help="Poisson arrival rate (req/s) for synthetic trace")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-trace", default=None, help="If set, save the used trace to this CSV path")

    # Cluster
    p.add_argument("--num-gpus", type=int, default=10_000)
    p.add_argument("--gpus-per-server", type=int, default=8)
    p.add_argument("--servers-per-rack", type=int, default=32)

    # PEACE knobs
    p.add_argument("--peace-prefill-fraction", type=float, default=0.2)
    p.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size used when choosing SP arch")

    # Sarathi knob
    p.add_argument("--sarathi-gpus-per-request", type=int, default=8)

    # Hyper-parameter configuring
    p.add_argument("--accuracy-degradation", type=float, default=0.0, help="Accuracy degradation constraint C_Ad (e.g., 0.01)")
    p.add_argument("--configuring-policy", choices=["heuristic", "pickle_regressor"], default="heuristic")
    p.add_argument("--pickle-regressor", default=None, help="Path to pickled regressor (if policy=pickle_regressor)")

    # Outputs
    p.add_argument("--outdir", default="out", help="Directory to write summary/results")
    p.add_argument("--save-results", action="store_true", help="Save request-level results to JSON")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    profile = load_profile_db(args.profile)

    # Trace
    if args.trace_csv:
        trace = load_trace_csv(args.trace_csv)
    else:
        # default is to generate if no trace is provided
        if not args.generate_trace:
            args.generate_trace = True
        trace = generate_synthetic_trace(
            dataset=args.dataset,
            n=args.n,
            arrival_rate_rps=args.arrival_rate,
            seed=args.seed,
        )

    if args.save_trace:
        save_trace_csv(trace, args.save_trace)

    model_spec = _default_model_spec(args.model, tp_size=args.tp_size)

    cfg = SimulationConfig(
        method=args.method,
        model=model_spec,
        dataset=args.dataset,
        num_gpus_total=args.num_gpus,
        gpus_per_server=args.gpus_per_server,
        servers_per_rack=args.servers_per_rack,
        peace_prefill_gpu_fraction=args.peace_prefill_fraction,
        sarathi_gpus_per_request=args.sarathi_gpus_per_request,
        configuring_policy=args.configuring_policy,
        pickle_regressor_path=args.pickle_regressor,
        accuracy_degradation_constraint=args.accuracy_degradation,
    )

    results = run_simulation(trace=trace, profile=profile, cfg=cfg)
    summary = summarize(results)

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.json")
    save_summary_json(summary, summary_path)

    if args.save_results:
        save_results_json(results, os.path.join(args.outdir, "results.json"))

    # Pretty print to stdout
    print("=== Simulation summary ===")
    print(f"method={cfg.method}  model={cfg.model.name}  dataset={cfg.dataset}")
    for k in [
        "n_requests",
        "avg_ttft_s",
        "avg_tbt_s",
        "avg_response_time_s",
        "prefill_throughput_tps",
        "decode_throughput_tps",
        "trace_makespan_s",
    ]:
        if k in summary:
            print(f"{k}: {summary[k]:.4f}")

    print(f"\nWrote summary: {summary_path}")
    if args.save_results:
        print(f"Wrote request-level results: {os.path.join(args.outdir,'results.json')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
