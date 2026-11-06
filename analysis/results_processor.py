"""Process and analyze benchmark results."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def load_results(results_dir: str = "results/raw") -> List[Dict]:
    """Load all benchmark results from the results directory."""
    results = []
    raw_dir = Path(results_dir)
    for file in raw_dir.glob("*_quantization_benchmark.json"):
        with open(file) as f:
            data = json.load(f)
        results.extend(data)
    return results


def extract_efficiency_data(results: List[Dict], include_model: bool = False) -> List[Dict]:
    """Extract efficiency metrics from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        include_model: Whether to include model name in output
        
    Returns:
        List of dictionaries with efficiency metrics
    """
    efficiency_data = []
    
    for result in results:
        model = result.get("model", "")
        quantization = result.get("quantization", "")
        q_bits = result["config"]["bits"]
        q_name = result["config"]["name"]
        
        # Efficiency metrics
        efficiency = result["efficiency"]
        entry = {
            "Quantization": q_name,
            "Bits": q_bits,
            "Model Size (MB)": efficiency["memory"]["model_size_mb"],
            "GPU Memory Used (MB)": efficiency["memory"]["gpu_used_mb"],
            "Avg Latency (ms)": efficiency["latency"]["avg_latency_ms"],
            "Tokens/sec": efficiency["token_speed"]
        }
        
        if include_model:
            entry["Model"] = model
            
        # Include GPU Memory for plotting
        entry["GPU Memory (MB)"] = efficiency["memory"]["gpu_used_mb"]
        entry["Latency (ms)"] = efficiency["latency"]["avg_latency_ms"]
        
        efficiency_data.append(entry)
    
    return efficiency_data


def extract_accuracy_data(results: List[Dict], include_model: bool = False) -> List[Dict]:
    """Extract accuracy metrics from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        include_model: Whether to include model name in output
        
    Returns:
        List of dictionaries with accuracy metrics
    """
    accuracy_data = []
    
    for result in results:
        model = result.get("model", "")
        quantization = result.get("quantization", "")
        q_bits = result["config"]["bits"]
        q_name = result["config"]["name"]
        
        # Accuracy metrics
        for acc_result in result["accuracy"]:
            benchmark = acc_result["benchmark"]
            # Handle both dict and direct value
            metrics = acc_result.get("metrics", {})
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    entry = {
                        "Quantization": q_name,
                        "Bits": q_bits,
                        "Benchmark": benchmark,
                        "Metric": metric,
                        "Value": value
                    }
                    if include_model:
                        entry["Model"] = model
                    accuracy_data.append(entry)
            else:
                entry = {
                    "Quantization": q_name,
                    "Bits": q_bits,
                    "Benchmark": benchmark,
                    "Metric": "accuracy",
                    "Value": metrics
                }
                if include_model:
                    entry["Model"] = model
                accuracy_data.append(entry)
    
    return accuracy_data


def create_dataframes(results: List[Dict], include_model: bool = False) -> Dict[str, pd.DataFrame]:
    """Create DataFrames from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        include_model: Whether to include model name in output
        
    Returns:
        Dictionary with 'efficiency' and 'accuracy' DataFrames
    """
    efficiency_data = extract_efficiency_data(results, include_model)
    accuracy_data = extract_accuracy_data(results, include_model)
    
    return {
        "efficiency": pd.DataFrame(efficiency_data),
        "accuracy": pd.DataFrame(accuracy_data)
    }


def merge_dataframes(eff_df: pd.DataFrame, acc_df: pd.DataFrame, 
                     on_columns: List[str] = None) -> pd.DataFrame:
    """Merge efficiency and accuracy DataFrames.
    
    Args:
        eff_df: Efficiency DataFrame
        acc_df: Accuracy DataFrame
        on_columns: Columns to merge on (defaults to Quantization and Bits)
        
    Returns:
        Merged DataFrame
    """
    if on_columns is None:
        on_columns = ["Quantization", "Bits"]
        # Check if Model column exists
        if "Model" in eff_df.columns and "Model" in acc_df.columns:
            on_columns = ["Model", "Quantization", "Bits"]
    
    return pd.merge(eff_df, acc_df, on=on_columns)


def generate_comparison_report(
    results: List[Dict],
    output_format: str = "markdown",
    filename: str = "quantization_comparison",
    include_model: bool = True
) -> Union[str, Dict[str, str]]:
    """Generate a comparison report from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_format: Format for the report ('markdown', 'csv', 'json')
        filename: Base filename for the report
        include_model: Whether to include model name in report
        
    Returns:
        Report as string or dict (for csv/json with multiple parts)
    """
    # Create dataframes
    dfs = create_dataframes(results, include_model)
    eff_df = dfs["efficiency"]
    acc_df = dfs["accuracy"]
    
    # Generate reports
    if output_format == "markdown":
        # Efficiency report
        eff_md = eff_df.to_markdown(index=False)
        
        # Accuracy report
        if include_model:
            acc_md = acc_df.pivot_table(
                index=["Model", "Quantization", "Bits"],
                columns=["Benchmark", "Metric"],
                values="Value"
            ).to_markdown()
        else:
            acc_md = acc_df.pivot_table(
                index=["Quantization", "Bits"],
                columns=["Benchmark", "Metric"],
                values="Value"
            ).to_markdown()
        
        report = f"""# LLM Quantization Comparison Report

## Efficiency Metrics
{eff_md}

## Accuracy Metrics
{acc_md}

---
*Generated by [llm-quantization-benchmark](https://github.com/HaaseSchuetz/llm-quantization-benchmark)*
"""
        return report
    
    elif output_format == "csv":
        return {
            "efficiency": eff_df.to_csv(index=False),
            "accuracy": acc_df.to_csv(index=False)
        }
    
    elif output_format == "json":
        return json.dumps({
            "efficiency": dfs["efficiency"].to_dict(orient="records"),
            "accuracy": dfs["accuracy"].to_dict(orient="records")
        }, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {output_format}")


def save_report(
    report: Union[str, Dict[str, str]],
    filename: str,
    output_format: str,
    save_dir: str = "results/reports"
) -> Path:
    """Save a report to file.
    
    Args:
        report: Report content (string or dict for csv/json)
        filename: Base filename for the report
        output_format: Format of the report
        save_dir: Directory to save the report
        
    Returns:
        Path to the saved report file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ext = {
        "markdown": "md",
        "csv": "csv",
        "json": "json"
    }[output_format]
    
    if output_format == "markdown":
        filepath = save_path / f"{filename}.{ext}"
        with open(filepath, "w") as f:
            f.write(report)
    else:
        # For CSV/JSON, save efficiency and accuracy separately
        if isinstance(report, dict):
            for key, content in report.items():
                filepath = save_path / f"{filename}_{key}.{ext}"
                with open(filepath, "w") as f:
                    f.write(content)
        else:
            filepath = save_path / f"{filename}.{ext}"
            with open(filepath, "w") as f:
                f.write(report)
    
    logger.info(f"Report saved to: {save_path}")
    return filepath


def plot_results(
    results: List[Dict],
    save_dir: str = "results/reports",
    include_model: bool = True
) -> None:
    """Generate plots from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        save_dir: Directory to save plots
        include_model: Whether to include model name in plots
    """
    # Create dataframes
    dfs = create_dataframes(results, include_model)
    eff_df = dfs["efficiency"]
    acc_df = dfs["accuracy"]
    
    # Merge data for plotting
    if include_model:
        on_cols = ["Model", "Quantization", "Bits"]
    else:
        on_cols = ["Quantization", "Bits"]
    
    # Check which columns exist
    available_cols = [c for c in on_cols if c in eff_df.columns and c in acc_df.columns]
    merged_df = pd.merge(eff_df, acc_df, on=available_cols)
    
    # Ensure we have the right column names for plotting
    x_cols = {
        "size": "Model Size (MB)",
        "latency": "Latency (ms)",
        "throughput": "Tokens/sec"
    }
    
    y_col = "Value"  # Accuracy value
    
    # Plot 1: Model Size vs. Accuracy
    plt.figure(figsize=(12, 6))
    if "Model Size (MB)" in merged_df.columns:
        sns.scatterplot(
            data=merged_df,
            x="Model Size (MB)",
            y=y_col,
            hue="Bits",
            style="Quantization",
            size="Bits",
            sizes=(100, 200, 300),
            alpha=0.7
        )
        plt.title("Model Size vs. Accuracy (Lower Size = Better, Higher Accuracy = Better)")
        plt.ylabel("Accuracy")
    else:
        logger.warning("Model Size (MB) column not found in merged dataframe")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "size_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Latency vs. Accuracy
    plt.figure(figsize=(12, 6))
    if "Latency (ms)" in merged_df.columns:
        sns.scatterplot(
            data=merged_df,
            x="Latency (ms)",
            y=y_col,
            hue="Bits",
            style="Quantization",
            size="Bits",
            sizes=(100, 200, 300),
            alpha=0.7
        )
        plt.title("Latency vs. Accuracy (Lower Latency = Better, Higher Accuracy = Better)")
        plt.ylabel("Accuracy")
    else:
        logger.warning("Latency (ms) column not found in merged dataframe")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "latency_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Throughput vs. Accuracy
    plt.figure(figsize=(12, 6))
    if "Tokens/sec" in merged_df.columns:
        sns.scatterplot(
            data=merged_df,
            x="Tokens/sec",
            y=y_col,
            hue="Bits",
            style="Quantization",
            size="Bits",
            sizes=(100, 200, 300),
            alpha=0.7
        )
        plt.title("Throughput vs. Accuracy (Higher Throughput = Better, Higher Accuracy = Better)")
        plt.ylabel("Accuracy")
    else:
        logger.warning("Tokens/sec column not found in merged dataframe")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "throughput_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Plots saved to: {save_dir}")
