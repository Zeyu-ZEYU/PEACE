"""PEACE paper defaults for experiment scripts.

This module centralizes a small set of constants that are repeatedly used by
the experiment harnesses in `PEACE/exp/`.

The values here are derived from the IPDPS'26 camera-ready paper:
`paper_ipdps26_camera_ready.tex`.

These defaults are *not* hard requirements; they simply match the paper’s
evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


DEFAULT_SHORT_THRESHOLD: int = 4096
"""Prompt tokens < 4096 are considered short; >= 4096 are considered long."""


DEFAULT_LONG_RANGE: Tuple[int, int] = (100_000, 500_000)
"""Paper resampling range for long inputs (in tokens)."""


@dataclass(frozen=True)
class PaperModelConfig:
    """Model configuration used in the paper evaluation."""

    key: str
    display_name: str
    tensor_parallel_size: int
    short_decode_replicas: int


PAPER_MODELS: Dict[str, PaperModelConfig] = {
    "mistral7b": PaperModelConfig(
        key="mistral7b",
        display_name="Mistral-v0.3 7B",
        tensor_parallel_size=1,
        short_decode_replicas=4,
    ),
    "phi3_14b": PaperModelConfig(
        key="phi3_14b",
        display_name="Phi-3 14B",
        tensor_parallel_size=1,
        short_decode_replicas=4,
    ),
    "yi34b": PaperModelConfig(
        key="yi34b",
        display_name="Yi 34B",
        tensor_parallel_size=4,
        short_decode_replicas=1,
    ),
    "llama31_70b": PaperModelConfig(
        key="llama31_70b",
        display_name="Llama-3.1 70B",
        tensor_parallel_size=4,
        short_decode_replicas=1,
    ),
}


def get_paper_model(key: str) -> PaperModelConfig:
    """Return a paper model config by key.

    Args:
        key: One of: mistral7b, phi3_14b, yi34b, llama31_70b.
    """
    k = key.strip().lower()
    if k not in PAPER_MODELS:
        valid = ", ".join(sorted(PAPER_MODELS.keys()))
        raise KeyError(f"Unknown paper model key: {key}. Valid keys: {valid}")
    return PAPER_MODELS[k]
