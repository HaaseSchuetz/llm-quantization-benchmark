import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
from quantization.methods import get_quantizer
from quantization.metrics.memory import MemoryBenchmark
from quantization.metrics.latency import LatencyBenchmark
from quantization.metrics.throughput import ThroughputBenchmark
from evaluation.evaluator import LLEvaluator
from config.logging_config import setup_logging
import torch

logger = logging.getLogger(__name__)

def benchmark_quantization(
    model_name: str,
    quantization_methods: List[str],
    benchmarks: List[str],
    save_dir: str = "results",
    limit: int = None,
    prompts: List[str] = None
):
    """Benchmark a model across multiple quantization methods."""
    # Load configs
    with open("config/models.json") as f:
        models_config = json.load(f)
    with open("config/quantization.json") as f:
        quantization_configs = json.load(f)
    with open("config/benchmarks.json") as f:
        benchmarks_config = json.load(f)

    if prompts is None:
        prompts = [
            "Explain the concept of quantization in machine learning.",
            "Write a Python function to reverse a string.",
            "What are the benefits of using LLMs in production?",
            "Summarize the key ideas of parameter-efficient fine-tuning."
        ]

    # Initialize evaluator and results
    evaluator = LLEvaluator()
    all_results = []

    # Create save directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "raw").mkdir(parents=True, exist_ok=True)
    (save_path / "reports").mkdir(parents=True, exist_ok=True)

    # Benchmark each quantization method
    for q_method in quantization_methods:
        logger.info(f"Benchmarking {model_name} with {q_method}...")

        # Get quantizer
        q_config = quantization_configs[q_method]
        quantizer = get_quantizer(q_config)

        # Quantize model
        model, tokenizer = quantizer.quantize(model_name)

        # Benchmark efficiency metrics
        logger.info("Running efficiency benchmarks...")
        efficiency_results = {
            "memory": MemoryBenchmark.benchmark_memory(model, tokenizer, prompts[0]),
            "latency": LatencyBenchmark.benchmark_latency(model, tokenizer, prompts[:2]),
            "throughput": ThroughputBenchmark.benchmark_throughput(model, tokenizer, prompts[:2]),
            "token_speed": LatencyBenchmark.benchmark_token_generation_speed(model, tokenizer)
        }

        # Benchmark accuracy
        logger.info("Running accuracy benchmarks...")
        accuracy_results = []
        for benchmark in benchmarks:
            try:
                eval_result = evaluator.evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    quantization=q_method,
                    benchmark_name=benchmark,
                    limit=limit,
                    save_raw=True
                )
                accuracy_results.append(eval_result)
                logger.info(f"✅ {benchmark}: {eval_result.metrics}")
            except Exception as e:
                logger.error(f"❌ {benchmark}: {str(e)}")

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Store results
        result = {
            "model": model_name,
            "quantization": q_method,
            "efficiency": efficiency_results,
            "accuracy": [vars(r) for r in accuracy_results],
            "config": q_config
        }
        all_results.append(result)

    # Save results
    results_path = save_path / "raw" / f"{model_name}_quantization_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"All benchmarks complete! Results saved to: {results_path}")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM quantization methods.")
    parser.add_argument("--model", type=str, required=True, help="Model name (from config/models.json)")
    parser.add_argument("--quantizations", nargs="+", required=True, help="Quantization methods (from config/quantization.json)")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu", "truthfulqa"], help="Benchmarks to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per benchmark")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path (optional)")
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)

    benchmark_quantization(
        model_name=args.model,
        quantization_methods=args.quantizations,
        benchmarks=args.benchmarks,
        save_dir=args.save_dir,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
