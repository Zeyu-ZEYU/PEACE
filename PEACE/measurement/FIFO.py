import heapq
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt


@dataclass(order=True)
class Request:
    arrival_time: float
    input_tokens: int
    output_tokens: int
    id: int = field(compare=False)
    start_time: float = field(default=None, compare=False)
    finish_time: float = field(default=None, compare=False)

    def service_time(self, prefill_speed: float, decode_speed: float) -> float:
        """
        Estimate service time = prefill + decode latency
        Speeds are in tokens/sec
        """
        prefill = self.input_tokens / prefill_speed
        decode = self.output_tokens / decode_speed
        return prefill + decode


class FIFOInferSimulator:
    def __init__(self, model_speed: dict, trace: List[Request], num_gpus: int = 8):
        """
        model_speed: dict with {'prefill': tokens/sec, 'decode': tokens/sec}
        trace: List of Request objects
        """
        self.prefill_speed = model_speed['prefill']
        self.decode_speed = model_speed['decode']
        self.trace = sorted(trace, key=lambda r: r.arrival_time)
        self.num_gpus = num_gpus
        self.queue = []
        self.now = 0.0
        self.infer_heap = []

    def simulate(self):
        """
        Simulate FIFO scheduling.
        """
        idx = 0
        n = len(self.trace)
        finished_requests = []

        while idx < n or self.infer_heap or self.queue:
            # Load requests that arrive at current time
            while idx < n and self.trace[idx].arrival_time <= self.now:
                self.queue.append(self.trace[idx])
                idx += 1

            # Free up completed GPUs
            while self.infer_heap and self.infer_heap[0][0] <= self.now:
                heapq.heappop(self.infer_heap)

            # Schedule requests if GPUs are available
            while len(self.infer_heap) < self.num_gpus and self.queue:
                req = self.queue.pop(0)
                req.start_time = self.now
                service = req.service_time(self.prefill_speed, self.decode_speed)
                req.finish_time = self.now + service
                heapq.heappush(self.infer_heap, (req.finish_time, req))
                finished_requests.append(req)

            # Advance time
            if self.infer_heap:
                self.now = self.infer_heap[0][0]
            elif idx < n:
                self.now = self.trace[idx].arrival_time

        return finished_requests

    def evaluate(self, finished_requests: List[Request]):
        queueing_delays = [r.start_time - r.arrival_time for r in finished_requests]
        jcts = [r.finish_time - r.arrival_time for r in finished_requests]
        makespan = max(r.finish_time for r in finished_requests)
        throughput = len(finished_requests) / makespan

        return {
            'avg_queueing_delay': np.mean(queueing_delays),
            'p99_queueing_delay': np.percentile(queueing_delays, 99),
            'avg_jct': np.mean(jcts),
            'p99_jct': np.percentile(jcts, 99),
            'throughput': throughput
        }


def generate_synthetic_trace(n: int = 1000, seed: int = 42) -> List[Request]:
    """
    Generate a synthetic trace resembling Azure 2024 LLM inference pattern:
    - Most requests are short (input <= 2K tokens)
    - Few long-tail requests (input up to 8K)
    """
    random.seed(seed)
    np.random.seed(seed)
    trace = []
    for i in range(n):
        arrival_time = np.random.exponential(scale=0.5) + (trace[-1].arrival_time if trace else 0)
        if random.random() < 0.8:
            input_tokens = random.randint(100, 2000)
        else:
            input_tokens = random.randint(2000, 8192)
        output_tokens = random.randint(50, 1000)
        trace.append(Request(arrival_time=arrival_time, input_tokens=input_tokens, output_tokens=output_tokens, id=i))
    return trace


def plot_metrics(metrics):
    names = ['avg_queueing_delay', 'p99_queueing_delay', 'avg_jct', 'p99_jct', 'throughput']
    values = [metrics[k] for k in names]
    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.title("FIFO Scheduling Performance")
    plt.xticks(rotation=30)
    plt.ylabel("Time (s) or Requests/sec")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Llama 3.1 70B assumed performance on A100: ~1000 tokens/s
    model_speed = {
        'prefill': 1200,  # tokens/sec per GPU
        'decode': 800
    }

    # Generate synthetic trace
    trace = generate_synthetic_trace(n=1000)

    # Simulate FIFO
    simulator = FIFOInferSimulator(model_speed=model_speed, trace=trace, num_gpus=8)
    finished = simulator.simulate()
    metrics = simulator.evaluate(finished)

    # Output & plot results
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    plot_metrics(metrics)
