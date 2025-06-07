import heapq
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.pyplot as plt


@dataclass(order=True)
class Request:
    arrival_time: float
    input_tokens: int
    output_tokens: int
    id: int = field(compare=False)
    request_type: str = field(default="short", compare=False)  # 'short' or 'long'
    start_time: float = field(default=None, compare=False)
    finish_time: float = field(default=None, compare=False)

    def service_time(self, prefill_speed: float, decode_speed: float) -> float:
        prefill = self.input_tokens / prefill_speed
        decode = self.output_tokens / decode_speed
        return prefill + decode


class ReservationBasedSimulator:
    def __init__(
        self,
        trace: List[Request],
        model_speed: dict,
        total_gpus: int = 16,
        short_fraction: float = 0.875,  # 87.5% GPUs for short requests
    ):
        self.prefill_speed = model_speed["prefill"]
        self.decode_speed = model_speed["decode"]
        self.trace = sorted(trace, key=lambda r: r.arrival_time)

        self.total_gpus = total_gpus
        self.short_pool_size = int(total_gpus * short_fraction)
        self.long_pool_size = total_gpus - self.short_pool_size

        self.short_pool_heap = []  # [(end_time, req), ...]
        self.long_pool_heap = []

        self.short_queue = []
        self.long_queue = []

        self.time = 0.0
        self.finished_requests = []

    def simulate(self):
        idx = 0
        n = len(self.trace)

        while idx < n or self.short_queue or self.long_queue or self.short_pool_heap or self.long_pool_heap:
            # Insert new arrivals
            while idx < n and self.trace[idx].arrival_time <= self.time:
                r = self.trace[idx]
                if r.request_type == "short":
                    self.short_queue.append(r)
                else:
                    self.long_queue.append(r)
                idx += 1

            # Release completed jobs
            while self.short_pool_heap and self.short_pool_heap[0][0] <= self.time:
                heapq.heappop(self.short_pool_heap)
            while self.long_pool_heap and self.long_pool_heap[0][0] <= self.time:
                heapq.heappop(self.long_pool_heap)

            # Try scheduling short requests
            while len(self.short_pool_heap) < self.short_pool_size and self.short_queue:
                req = self.short_queue.pop(0)
                req.start_time = self.time
                service = req.service_time(self.prefill_speed, self.decode_speed)
                req.finish_time = self.time + service
                heapq.heappush(self.short_pool_heap, (req.finish_time, req))
                self.finished_requests.append(req)

            # Try scheduling long requests
            while len(self.long_pool_heap) < self.long_pool_size and self.long_queue:
                req = self.long_queue.pop(0)
                req.start_time = self.time
                service = req.service_time(self.prefill_speed, self.decode_speed)
                req.finish_time = self.time + service
                heapq.heappush(self.long_pool_heap, (req.finish_time, req))
                self.finished_requests.append(req)

            # Advance simulation time
            next_arrival = self.trace[idx].arrival_time if idx < n else float("inf")
            next_finish_short = self.short_pool_heap[0][0] if self.short_pool_heap else float("inf")
            next_finish_long = self.long_pool_heap[0][0] if self.long_pool_heap else float("inf")
            self.time = min(next_arrival, next_finish_short, next_finish_long)

        return self.finished_requests

    def evaluate(self):
        delays = [r.start_time - r.arrival_time for r in self.finished_requests]
        jcts = [r.finish_time - r.arrival_time for r in self.finished_requests]
        makespan = max(r.finish_time for r in self.finished_requests)
        throughput = len(self.finished_requests) / makespan

        short_jct = [r.finish_time - r.arrival_time for r in self.finished_requests if r.request_type == "short"]
        long_jct = [r.finish_time - r.arrival_time for r in self.finished_requests if r.request_type == "long"]

        return {
            "avg_queueing_delay": np.mean(delays),
            "p99_queueing_delay": np.percentile(delays, 99),
            "avg_jct": np.mean(jcts),
            "p99_jct": np.percentile(jcts, 99),
            "throughput": throughput,
            "avg_short_jct": np.mean(short_jct),
            "avg_long_jct": np.mean(long_jct),
        }


def generate_trace(n: int = 1000, seed: int = 42) -> List[Request]:
    random.seed(seed)
    np.random.seed(seed)
    trace = []
    arrival_time = 0.0

    for i in range(n):
        inter_arrival = np.random.exponential(scale=0.4)
        arrival_time += inter_arrival

        if random.random() < 0.95:
            input_tokens = random.randint(100, 2000)
            req_type = "short"
        else:
            input_tokens = random.randint(8000, 16000)
            req_type = "long"

        output_tokens = random.randint(50, 1000)
        trace.append(Request(arrival_time=arrival_time, input_tokens=input_tokens,
                             output_tokens=output_tokens, id=i, request_type=req_type))
    return trace


def plot_metrics(metrics: dict):
    keys = ['avg_queueing_delay', 'p99_queueing_delay', 'avg_jct', 'p99_jct', 'throughput', 'avg_short_jct', 'avg_long_jct']
    values = [metrics[k] for k in keys]

    plt.figure(figsize=(12, 6))
    plt.bar(keys, values)
    plt.xticks(rotation=30)
    plt.ylabel("Seconds / Requests per sec")
    plt.title("Reservation-Based Scheduling Metrics")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟 Llama-3.1 70B 单GPU预估速度
    model_speed = {
        "prefill": 1200,
        "decode": 800
    }

    # 生成Trace
    trace = generate_trace(n=1000)

    # 运行Reservation-Based Scheduler
    simulator = ReservationBasedSimulator(trace, model_speed, total_gpus=16, short_fraction=0.875)
    simulator.simulate()
    metrics = simulator.evaluate()

    # 输出指标
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    plot_metrics(metrics)
