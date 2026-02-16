# PEACE Scalability Simulation

This repository provides a lightweight Python-based simulator that reproduces 
the scalability experiments described in the PEACE paper.

The simulator does NOT execute real LLM inference. Instead, it uses:
- Offline profiling curves (prefill/decode time vs. input length)
- Synthetic request traces
- A cluster-level scheduling abstraction

This design enables large-scale experiments (e.g., 10K GPUs) without requiring
actual GPU hardware.

---

## Directory Structure

peace_scalability_sim/
│
├── run_simulation.py          # Main entry point
├── simulator/
│   ├── cluster.py             # Cluster abstraction
│   ├── request.py             # Request model
│   ├── scheduler.py           # Scheduling logic (PEACE-style separation)
│   └── simulator.py           # Event-driven simulator
│
├── profiles/
│   └── sample_profile.json    # Example performance profile
│
└── README.md

---

## Requirements

- Python 3.8+
- No external dependencies required (standard library only)

---

## Quick Start

Run with default parameters:

    python run_simulation.py

Run with custom configuration:

    python run_simulation.py \
        --num_gpus 10000 \
        --arrival_rate 5.0 \
        --simulation_time 1000 \
        --profile profiles/sample_profile.json

---

## Simulation Model

Each request consists of:
- Input length (tokens)
- Output length (tokens)

The profile file defines:
- Prefill time as a function of input length
- Decode time per token

The simulator performs event-driven scheduling over a cluster of GPUs
and reports metrics including:

- Throughput (requests/sec)
- Average latency
- Tail latency (P95, P99)

---

## Profile Format

The profile JSON file follows this format:

{
  "prefill_time_per_token_ms": 0.5,
  "decode_time_per_token_ms": 0.2
}

You can replace this with a more detailed curve-based profile if needed.

---

## Extending the Simulator

You may extend the simulator to:

- Implement more advanced scheduling strategies
- Introduce heterogeneous GPU types
- Load real request traces
- Integrate more accurate latency curves

---

## Example Output

The simulator prints summary statistics at the end:

Throughput: 1200.5 req/s
Average latency: 45.3 ms
P95 latency: 80.1 ms
P99 latency: 120.4 ms

---

## Notes

This simulator is intended for research reproduction and scalability analysis.
It is not designed to replace real-world benchmarking on actual hardware.

Please adapt the simulator according to your PEACE experimental configuration.
