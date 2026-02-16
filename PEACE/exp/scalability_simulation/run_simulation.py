#!/usr/bin/env python3
"""Convenience wrapper for `peace_sim.cli`.

Run:
    python run_simulation.py --help
"""

from peace_sim.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
