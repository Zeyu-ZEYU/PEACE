from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


SPArch = Literal["sp-ntp", "sp-tp"]


@dataclass(frozen=True)
class ModelSpec:
    """Minimal model spec needed by the simulator.

    hidden_size is the Transformer hidden dim (d in the appendix).
    tp_size is the tensor-parallel size (N in the appendix).
    """

    name: str
    hidden_size: int
    tp_size: int = 1


def choose_sp_architecture(input_len: int, model: ModelSpec) -> SPArch:
    """Choose between SP-NTP and SP-TP.

    Based on Appendix "Considerations for Selecting SP Architectures".

    For deployments without TP, we always choose SP-NTP.
    For deployments with TP (N>2), the appendix derives a threshold:
        SP-NTP better if L_in > 2*d*N/(N-2)

    The main paper also notes that SP-NTP is preferable when input length is >100K.
    """
    if model.tp_size <= 1:
        return "sp-ntp"
    n = model.tp_size
    if n <= 2:
        # Appendix says SP-NTP always better for N=2.
        return "sp-ntp"

    d = model.hidden_size
    threshold = int(math.ceil(2.0 * d * n / (n - 2)))
    if input_len >= max(100_000, threshold):
        return "sp-ntp"
    return "sp-tp"


def spt_length_tokens(arch: SPArch, model: ModelSpec) -> int:
    """Return the SPT length used to estimate GPU count.

    The paper reports that in their experiments:
    - SPT length can be 64K without TP
    - SPT length can be 128K for TP=8

    We expose a simple heuristic that matches this behavior.
    """
    if arch == "sp-tp" and model.tp_size >= 8:
        return 128_000
    # Default
    return 64_000


def gpus_for_prefill(input_len: int, arch: SPArch, model: ModelSpec) -> int:
    """Estimate number of GPUs for prefill."""
    spt = spt_length_tokens(arch, model)
    return max(1, int(math.ceil(input_len / spt)))


def gpus_for_decode(input_len: int, arch: SPArch, model: ModelSpec) -> int:
    """Estimate number of GPUs for decode.

    In the paper, decode uses sharded KV cache across multiple GPUs.
    For this simulator we keep it simple and reuse the prefill GPU count.
    """
    return gpus_for_prefill(input_len, arch, model)
