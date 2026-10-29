import argparse
import logging
from pathlib import Path
from typing import Dict, List
from analysis.results_processor import (
    load_results,
    generate_comparison_report,
    save_report,
    plot_results
)
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare quantization methods.")
    parser.add_argument("--results-dir", type=str, default="results/raw", help="Directory with benchmark results")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "csv", "json"])
    parser.add_argument("--filename", type=str, default="quantization_comparison")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path (optional)")
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)

    # Load results
    results = load_results(args.results_dir)

    # Generate report
    report = generate_comparison_report(results, args.format, args.filename)
    save_report(report, args.filename, args.format)

    # Generate plots
    if args.plot:
        plot_results(results)

    logger.info("Comparison Report:")
    if args.format == "markdown":
        print(report)

if __name__ == "__main__":
    main()
