# LLM Quantization Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HaaseSchuetz/llm-quantization-benchmark/blob/main/notebooks/demo.ipynb)

**A comprehensive benchmarking suite for LLM quantization methods, comparing performance, memory, latency, and throughput.**

---

## **Why This Project?**
Quantization is **critical** for deploying LLMs in production, but it **degrades accuracy**. This suite helps you:
- **Compare quantization methods** (INT8, INT4, GPTQ, AWQ, etc.).
- **Measure trade-offs** between **accuracy, memory, latency, and throughput**.
- **Find the best method** for your use case (e.g., edge devices vs. cloud).
- **Reproduce results** with standardized benchmarks.


---

## **Features**
   Feature               | Description                                  |
 |-----------------------|----------------------------------------------|
 | **Multi-Method**      | Supports **FP16, INT8, INT4, GPTQ, AWQ, Optimum**. |
 | **Efficiency Metrics**| Measures **memory, latency, throughput, tokens/sec**. |
 | **Accuracy Metrics** | Evaluates on **MMLU, TruthfulQA, Perplexity**. |
 | **Visualizations**    | Generates **plots** for trade-off analysis.   |
 | **Modular**           | Easy to **add new methods/benchmarks**.      |
 | **Reproducible**      | Full config management and logging.         |

---
---
## **Supported Quantization Methods**
 | Method       | Library          | Bits | Description                                  |
 |--------------|------------------|------|----------------------------------------------|
 | **FP16**     | Transformers     | 16   | Baseline (no quantization).                  |
 | **INT8**     | bitsandbytes     | 8    | 8-bit quantization.                          |
 | **INT4**     | bitsandbytes     | 4    | 4-bit quantization (NF4).                    |
 | **GPTQ**     | auto-gptq        | 4    | GPTQ: Accurate 4-bit quantization.            |
 | **AWQ**      | auto-awq         | 4    | Activation-aware quantization.               |
 | **Optimum**  | Hugging Face     | 8    | ONNX-based 8-bit quantization.                |

---
---
## **Supported Benchmarks**
 | Benchmark       | Description                                  | Metric       |
 |-----------------|----------------------------------------------|--------------|
 | **MMLU**        | Massive Multitask Language Understanding     | Accuracy     |
 | **TruthfulQA**  | Measures truthfulness of answers             | Accuracy     |
 | **Perplexity**  | Language modeling quality                    | Perplexity   |

---
---
## Setup
### 1. Clone the Repo
```bash
git clone https://github.com/HaaseSchuetz/llm-quantization-benchmark.git
cd llm-quantization-benchmark
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. CUDA
For GPTQ/AWQ, ensure you have CUDA 12.1+:
```bash
nvcc --version  # Check CUDA version
```

## Usage
Benchmarking
```bash
# Benchmark Mistral-7B with INT4 and INT8
python scripts/benchmark.py \
    --model mistral-7b \
    --quantizations int4 int8 \
    --benchmarks mmlu truthfulqa \
    --limit 100  # Use 100 examples per benchmark (remove for full)
```

Compare quantization methods
```bash
# Compare all methods for Mistral-7B
python scripts/benchmark.py \
    --model mistral-7b \
    --quantizations fp16 int8 int4 gptq awq \
    --benchmarks mmlu

# Generate comparison report and plots
python scripts/compare.py --results-dir results/raw --plot
```

Generate report
```bash
python scripts/compare.py --format markdown --filename mistral_quantization
```

## Example results
| Model | Quantization | Bits | Model Size (MB) | GPU Memory Used (MB) | Avg Latency (ms) | Tokens/sec |
| --- | --- | --- | --- | --- | --- | --- |
| mistral-7b | FP16 (Baseline) | 16 | 28200.0 | 56400 | 120.5 | 245.3 |
| mistral-7b | INT8 | 8 | 14100.0 | 28200 | 85.2 | 348.7 |
| mistral-7b | INT4 | 4 | 7050.0 | 14100 | 62.1 | 478.2 |
| mistral-7b | GPTQ (4-bit) | 4 | 7050.0 | 14100 | 58.3 | 510.4 |
| mistral-7b | AWQ (4-bit) | 4 | 7050.0 | 14100 | 55.7 | 535.1 |

| Model | Quantization | Bits | Benchmark | Metric | Value |
| --- | --- | --- | --- | --- | --- |
| mistral-7b | FP16 (Baseline) | 16 | MMLU | accuracy | 0.623 |
| mistral-7b | INT8 | 8 | MMLU | accuracy | 0.618 |
| mistral-7b | INT4 | 4 | MMLU | accuracy | 0.592 |
| mistral-7b | GPTQ (4-bit) | 4 | MMLU | accuracy | 0.605 |
| mistral-7b | AWQ (4-bit) | 4 | MMLU | accuracy | 0.611 |

> Results show ~2–5% accuracy drop with 4-bit quantization, but ~2x speedup and 4x memory savings.

## Methods
| Method | Pros | Cons | Best For |
| --- | --- | --- | --- |
| FP16 | No accuracy loss, simple | High memory usage | Development, high-end GPUs |
| INT8 | 2x memory savings, minimal accuracy loss | Slower than INT4 | Cloud deployment |
| INT4 | 4x memory savings, faster inference | ~2–5% accuracy drop | Edge devices |
| GPTQ | High accuracy for 4-bit | Requires calibration | Production with accuracy focus |
| AWQ | Better accuracy than GPTQ | Slower than GPTQ | High-accuracy 4-bit |
| Optimum | ONNX-based, hardware-optimized | Limited to 8-bit | CPU deployment |

## Metrics
| Metric | Description | Importance |
| --- | --- | --- |
| Accuracy | % correct answers on benchmarks | Primary metric for capability |
| Perplexity | Model uncertainty | Language modeling quality |
| Memory | GPU/CPU memory usage | Deployment feasibility |
| Latency | Time per inference request | User experience |
| Throughput | Tokens/sec | Batch processing efficiency |
