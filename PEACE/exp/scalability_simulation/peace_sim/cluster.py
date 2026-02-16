from __future__ import annotations

from dataclasses import dataclass
from heapq import nsmallest
from typing import List, Tuple


@dataclass(frozen=True)
class GPUId:
    rack: int
    server: int
    gpu: int

    def __str__(self) -> str:
        return f"{self.rack}-{self.server:03d}-{self.gpu}"


@dataclass(frozen=True)
class ClusterTopology:
    """A simple hierarchical topology.

    - GPUs are grouped into servers, servers into racks.
    - We expose a coarse bandwidth model by distance.

    Bandwidths are *effective* one-way bandwidth values in GB/s.
    """

    num_gpus_total: int
    gpus_per_server: int = 8
    servers_per_rack: int = 32

    # Effective bandwidths (GB/s)
    bw_intra_server_gbps: float = 300.0
    bw_intra_rack_gbps: float = 50.0
    bw_inter_rack_gbps: float = 25.0

    def num_servers(self) -> int:
        return (self.num_gpus_total + self.gpus_per_server - 1) // self.gpus_per_server

    def num_racks(self) -> int:
        return (self.num_servers() + self.servers_per_rack - 1) // self.servers_per_rack

    def gpu_id(self, global_gpu_index: int) -> GPUId:
        server_idx = global_gpu_index // self.gpus_per_server
        gpu_idx = global_gpu_index % self.gpus_per_server
        rack_idx = server_idx // self.servers_per_rack
        server_in_rack = server_idx % self.servers_per_rack
        return GPUId(rack=rack_idx, server=server_in_rack, gpu=gpu_idx)

    def bandwidth_gbps(self, a: GPUId, b: GPUId) -> float:
        if a.rack == b.rack and a.server == b.server:
            return self.bw_intra_server_gbps
        if a.rack == b.rack:
            return self.bw_intra_rack_gbps
        return self.bw_inter_rack_gbps


class ClusterState:
    """Mutable cluster state for contention simulation.

    Each GPU has an "available_at" timestamp.

    Allocation policy (default) tries to keep an allocation within as few servers
    as possible, and uses the earliest-available GPUs.
    """

    def __init__(self, topo: ClusterTopology):
        self.topo = topo
        # server -> per-gpu available_at times
        self._servers: List[List[float]] = [
            [0.0 for _ in range(topo.gpus_per_server)] for _ in range(topo.num_servers())
        ]

    def reset(self) -> None:
        for s in self._servers:
            for i in range(len(s)):
                s[i] = 0.0

    def _best_single_server_allocation(self, k: int, earliest_s: float) -> Tuple[int, float, List[int]]:
        """Return (server_idx, start_s, gpu_local_ids) for the best single-server fit."""
        best_server = -1
        best_start = float("inf")
        best_gpus: List[int] = []

        for server_idx, avail in enumerate(self._servers):
            # pick k GPUs with smallest available_at
            gpu_ids = sorted(range(len(avail)), key=lambda i: avail[i])[:k]
            start_s = max(earliest_s, max(avail[i] for i in gpu_ids))
            if start_s < best_start:
                best_server = server_idx
                best_start = start_s
                best_gpus = gpu_ids

        return best_server, best_start, best_gpus

    def allocate(self, num_gpus: int, earliest_s: float, duration_s: float) -> Tuple[float, float, List[GPUId]]:
        """Allocate GPUs and reserve them for [start, end].

        Returns (start_s, end_s, allocated_gpu_ids).

        Note: This is a lightweight contention model; it does not model
        communication overlap between jobs.
        """
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")

        gpus: List[GPUId] = []

        if num_gpus <= self.topo.gpus_per_server:
            server_idx, start_s, local_gpu_ids = self._best_single_server_allocation(num_gpus, earliest_s)
            end_s = start_s + duration_s
            for i in local_gpu_ids:
                self._servers[server_idx][i] = end_s
                gpus.append(self.topo.gpu_id(server_idx * self.topo.gpus_per_server + i))
            return start_s, end_s, gpus

        # Need multiple servers.
        servers_needed = (num_gpus + self.topo.gpus_per_server - 1) // self.topo.gpus_per_server
        # Rank servers by their earliest "all GPUs free" time (max avail) to approximate "most available".
        server_scores = []
        for server_idx, avail in enumerate(self._servers):
            server_scores.append((max(avail), server_idx))
        server_scores.sort(key=lambda x: x[0])
        chosen = [idx for _, idx in server_scores[:servers_needed]]

        # Within chosen servers, take GPUs with smallest available_at times.
        candidates: List[Tuple[float, int, int]] = []  # (avail_time, server_idx, gpu_local)
        for server_idx in chosen:
            for gpu_local, t in enumerate(self._servers[server_idx]):
                candidates.append((t, server_idx, gpu_local))
        candidates.sort(key=lambda x: x[0])
        picked = candidates[:num_gpus]

        start_s = max(earliest_s, max(t for t, _, _ in picked))
        end_s = start_s + duration_s
        for _, server_idx, gpu_local in picked:
            self._servers[server_idx][gpu_local] = end_s
            gpus.append(self.topo.gpu_id(server_idx * self.topo.gpus_per_server + gpu_local))
        return start_s, end_s, gpus

    def makespan(self) -> float:
        """Return the latest GPU availability time."""
        return max(max(avail) for avail in self._servers) if self._servers else 0.0

    def busy_time_sum(self) -> float:
        """A rough measure of how much GPU time was consumed.

        This sums per-GPU busy time assuming all started at 0.
        """
        return sum(sum(avail) for avail in self._servers)
