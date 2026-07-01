import time
from typing import Dict, List
import torch
import numpy as np

class ThroughputBenchmark:
    """Benchmark throughput (tokens/sec) of quantized models."""

    @staticmethod
    def benchmark_throughput(
        model,
        tokenizer,
        prompts: List[str],
        max_new_tokens: List[int] = [10, 50, 100],
        num_runs: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark throughput at different token lengths."""
        results = {}

        for tokens in max_new_tokens:
            # Tokenize prompts
            inputs = [tokenizer(prompt, return_tensors="pt").to("cuda") for prompt in prompts]

            # Warmup
            for _ in range(2):
                for inp in inputs:
                    _ = model.generate(**inp, max_new_tokens=10)

            # Measure
            start_time = time.time()
            for _ in range(num_runs):
                for inp in inputs:
                    _ = model.generate(**inp, max_new_tokens=tokens)
            torch.cuda.synchronize()
            total_time = time.time() - start_time

            total_tokens = tokens * len(prompts) * num_runs
            tokens_per_sec = total_tokens / total_time

            results[f"tokens_{tokens}"] = {
                "throughput_tokens_per_sec": tokens_per_sec,
                "total_time_sec": total_time,
                "total_tokens": total_tokens
            }

        return results
