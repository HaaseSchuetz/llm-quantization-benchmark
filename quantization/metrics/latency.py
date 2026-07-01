import time
from typing import Dict, List, Tuple
import torch
import numpy as np

class LatencyBenchmark:
    """Benchmark latency of quantized models."""

    @staticmethod
    def benchmark_latency(
        model,
        tokenizer,
        prompts: List[str],
        num_runs: int = 10,
        max_new_tokens: int = 100,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """Benchmark average latency for inference."""
        # Tokenize prompts
        inputs = [tokenizer(prompt, return_tensors="pt").to("cuda") for prompt in prompts]

        # Warmup
        for _ in range(warmup_runs):
            for inp in inputs:
                _ = model.generate(**inp, max_new_tokens=10)

        # Measure
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            for inp in inputs:
                _ = model.generate(**inp, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()  # Wait for GPU to finish
            latencies.append(time.time() - start_time)

        # Calculate stats
        avg_latency = np.mean(latencies) * 1000  # Convert to ms
        std_latency = np.std(latencies) * 1000
        min_latency = np.min(latencies) * 1000
        max_latency = np.max(latencies) * 1000

        # Per-prompt latency
        per_prompt_latency = avg_latency / len(prompts)

        return {
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "per_prompt_latency_ms": per_prompt_latency,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens
        }

    @staticmethod
    def benchmark_token_generation_speed(
        model,
        tokenizer,
        prompt: str = "Hello, world!",
        num_tokens: int = 100,
        num_runs: int = 5
    ) -> float:
        """Benchmark tokens generated per second."""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Warmup
        for _ in range(2):
            _ = model.generate(**inputs, max_new_tokens=10)

        # Measure
        start_time = time.time()
        for _ in range(num_runs):
            outputs = model.generate(**inputs, max_new_tokens=num_tokens)
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        total_tokens = num_tokens * num_runs
        tokens_per_sec = total_tokens / total_time

        return tokens_per_sec
