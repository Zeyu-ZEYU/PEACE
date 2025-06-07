import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple


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


class PriorityBasedScheduler:
    def __init__(
        self,
        trace: List[Request],
        model_speed: dict,
        num_gpus: int = 8,
    ):
        self.prefill_speed = model_speed["prefill"]
        self.decode_speed = model_speed["decode"]
        self.trace = sorted(trace, key=lambda r: r.arrival_time)
        self.num_gpus = num_gpus

        self.short_queue: List[Request] = []
        self.long_queue: List[Request] = []
        self.infer_heap: List[Tuple[float, Request]] = []

        self.time = 0.0
        self.finished_requests: List[Request] = []

    def simulate(self):
        idx = 0
        n = len(self.trace)

        while idx < n or self.short_queue or self.long_queue or self.infer_heap:
            # Add new arrivals
            while idx < n and self.trace[idx].arrival_time <= self.time:
                req = self.trace[idx]
                if req.request_type == "short":
                    self.short_queue.append(req)
                else:
                    self.long_queue.append(req)
                idx += 1

            # Release completed jobs
            while self.infer_heap and self.infer_heap[0][0] <= self.time:
                heapq.heappop(self.infer_heap)

            # Schedule short requests first
            while len(self.infer_heap) < self.num_gpus and self.short_queue:
                req = self.short_queue.pop(0)
                req.start_time = self.time
                latency = req.service_time(self.prefill_speed, self.decode_speed)
                req.finish_time = self.time + latency
                heapq.heappush(self.infer_heap, (req.finish_time, req))
                self.finished_requests.append(req)

            # Schedule long requests if no short ones
            while len(self.infer_heap) < self.num_gpus and not self.short_queue and self.long_queue:
                req = self.long_queue.pop(0)
                req.start_time = self.time
                latency = req.service_time(self.prefill_speed, self.decode_speed)
                req.finish_time = self.time + latency
                heapq.heappush(self.infer_heap, (req.finish_time, req))
                self.finished_requests.append(req)

            # Advance time
            next_event_time = float("inf")
            if idx < n:
                next_event_time = self.trace[idx].arrival_time
            if self.infer_heap:
                next_event_time = min(next_event_time, self.infer_heap[0][0])
            self.time = next_event_time

        return self.finished_requests

    def evaluate(self):
        queueing_delays = [r.start_time - r.arrival_time for r in self.finished_requests]
        jcts = [r.finish_time - r.arrival_time for r in self.finished_requests]
        makespan = max(r.finish_time for r in self.finished_requests)
        throughput = len(self.finished_requests) / makespan

        short_jcts = [r.finish_time - r.arrival_time for r in self.finished_requests if r.request_type == "short"]
        long_jcts = [r.finish_time - r.arrival_time for r in self.finished_requests if r.request_type == "long"]

        return {
            "avg_queueing_delay": np.mean(queueing_delays),
            "p99_queueing_delay": np.percentile(queueing_delays, 99),
            "avg_jct": np.mean(jcts),
            "p99_jct": np.percentile(jcts, 99),
            "throughput": throughput,
            "avg_short_jct": np.mean(short_jcts),
            "avg_long_jct": np.mean(long_jcts),
        }


def generate_trace(n: int = 1000, seed: int = 42) -> List[Request]:
    random.seed(seed)
    np.random.seed(seed)
    trace = []
    arrival_time = 0.0

    for i in range(n):
        inter_arrival = np.random.exponential(scale=0.5)
        arrival_time += inter_arrival

        if random.random() < 0.9:
            input_tokens = random.randint(100, 2000)
            request_type = "short"
        else:
            input_tokens = random.randint(8192, 16000)
            request_type = "long"

        output_tokens = random.randint(50, 1000)
        trace.append(Request(arrival_time, input_tokens, output_tokens, i, request_type))

    return trace


def plot_metrics(metrics: dict):
    keys = ['avg_queueing_delay', 'p99_queueing_delay', 'avg_jct', 'p99_jct', 'throughput', 'avg_short_jct', 'avg_long_jct']
    values = [metrics[k] for k in keys]

    plt.figure(figsize=(12, 6))
    plt.bar(keys, values)
    plt.xticks(rotation=30)
    plt.ylabel("Seconds / Req/sec")
    plt.title("Priority-Based Scheduling Metrics")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 模拟Llama-3.1 70B在A100集群下的吞吐能力
    model_speed = {
        "prefill": 1200,  # tokens/sec
        "decode": 800     # tokens/sec
    }

    trace = generate_trace(n=1000)
    scheduler = PriorityBasedScheduler(trace, model_speed, num_gpus=8)
    scheduler.simulate()
    metrics = scheduler.evaluate()

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    plot_metrics(metrics)
