"""PEACE scalability-test simulator (trace-driven).

This package provides:
- Synthetic trace generator approximating NeedleBench / BookCorpus stats.
- A cluster/topology model for 10k-GPU-scale experiments.
- GPU selection + a lightweight contention model.
- Profile-driven latency/throughput estimation.

It is designed so you can swap in *your own* profiled curves and (optionally)
your regression models for hyper-parameter configuring.
"""

__all__ = [
    "cli",
    "cluster",
    "profiles",
    "simulator",
    "trace",
]
