import torch
import psutil
import GPUtil
from typing import Dict, Optional

class MemoryBenchmark:
    """Benchmark memory usage of quantized models."""

    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, float]:
        """Get GPU memory usage in MB."""
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"gpu_used_mb": 0.0, "gpu_total_mb": 0.0}
        gpu = gpus[0]
        return {
            "gpu_used_mb": gpu.memoryUsed,
            "gpu_total_mb": gpu.memoryTotal,
            "gpu_utilization": gpu.memoryUtil * 100
        }

    @staticmethod
    def get_cpu_memory_usage() -> float:
        """Get CPU memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def get_model_memory(model) -> Dict[str, float]:
        """Get model memory usage."""
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_mb = (param_size + buffer_size) / (1024 * 1024)
        return {
            "model_size_mb": total_mb,
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024)
        }

    @staticmethod
    def benchmark_memory(model, tokenizer, input_text: str = "Hello, world!") -> Dict[str, float]:
        """Benchmark memory usage during inference."""
        # Warmup
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        _ = model.generate(**inputs, max_new_tokens=10)

        # Measure
        gpu_before = MemoryBenchmark.get_gpu_memory_usage()
        cpu_before = MemoryBenchmark.get_cpu_memory_usage()

        # Run inference
        outputs = model.generate(**inputs, max_new_tokens=100)

        gpu_after = MemoryBenchmark.get_gpu_memory_usage()
        cpu_after = MemoryBenchmark.get_cpu_memory_usage()

        # Calculate deltas
        gpu_used = gpu_after["gpu_used_mb"] - gpu_before["gpu_used_mb"]
        cpu_used = cpu_after - cpu_before

        # Model size
        model_memory = MemoryBenchmark.get_model_memory(model)

        return {
            **model_memory,
            "gpu_used_mb": gpu_used,
            "cpu_used_mb": cpu_used,
            "peak_gpu_mb": gpu_after["gpu_used_mb"]
        }
