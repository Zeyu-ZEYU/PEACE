"""
Variant definitions for the IPDPS'26 PEACE ablation study.

The ablation study in paper_ipdps26_camera_ready.tex defines:

- PEACE/PE  : PEACE without Preemption
- PEACE/Dis : PEACE without short-request decode Disaggregation
- PEACE/CoL : PEACE without Prefill-Decode CoLocation concurrency
- PEACE/FSP : PEACE without Fast SP for long-prefill (uses ring attention)

This module standardizes environment variables used by the experiment harness.
Your serving stack (e.g., a PEACE-patched vLLM) is expected to read these flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Variant:
    """Ablation variant descriptor."""
    name: str
    description: str
    env: Dict[str, str]


VARIANTS: Dict[str, Variant] = {
    "peace": Variant(
        name="PEACE",
        description="Full PEACE (all components enabled).",
        env={},
    ),
    "pe": Variant(
        name="PEACE/PE",
        description="PEACE without Preemption (short-prefill cannot preempt long-prefill).",
        env={"PEACE_DISABLE_PREEMPTION": "1"},
    ),
    "dis": Variant(
        name="PEACE/Dis",
        description="PEACE without short-request decode disaggregation (short prefill+decode on same GPUs).",
        env={"PEACE_DISABLE_DISAGGREGATION": "1"},
    ),
    "col": Variant(
        name="PEACE/CoL",
        description="PEACE without prefill–decode colocation concurrency (short-prefill preempts long-decode).",
        env={"PEACE_DISABLE_COLOCATION": "1"},
    ),
    "fsp": Variant(
        name="PEACE/FSP",
        description="PEACE without Fast SP for long-prefill (fallback to ring attention).",
        env={"PEACE_DISABLE_FAST_SP": "1"},
    ),
}


def get_variant(key: str) -> Variant:
    """Return a Variant by key (case-insensitive)."""
    k = key.strip().lower()
    if k not in VARIANTS:
        valid = ", ".join(sorted(VARIANTS.keys()))
        raise ValueError(f"Unknown variant: {key!r}. Valid: {valid}")
    return VARIANTS[k]
