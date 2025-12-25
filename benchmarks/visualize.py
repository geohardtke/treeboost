#!/usr/bin/env python3
"""
Generate professional benchmark visualizations for TreeBoost vs competitors.

Usage:
    python benchmarks/visualize.py results/benchmark_20241201_120000.json
    python benchmarks/visualize.py results/benchmark_*.json --combine
    python benchmarks/visualize.py --latest
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Professional color palette
COLORS = {
    "treeboost": "#3498db",   # Blue
    "xgboost": "#e74c3c",     # Red
    "lightgbm": "#2ecc71",    # Green
    "catboost": "#9b59b6",    # Purple
}

MARKERS = {
    "treeboost": "o",
    "xgboost": "s",
    "lightgbm": "^",
    "catboost": "D",
}

# Display names
DISPLAY_NAMES = {
    "treeboost": "TreeBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def load_results(filepath: Path) -> dict:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def setup_style():
    """Set up matplotlib style for professional plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "normal",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_training_time(results: list[dict], output_dir: Path):
    """Generate training time comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group data by library and dataset
    libraries = sorted(set(r["library"] for r in results))
    datasets = sorted(set(r["dataset_name"] for r in results),
                      key=lambda x: ["tiny", "small", "medium", "large", "xlarge"].index(x)
                      if x in ["tiny", "small", "medium", "large", "xlarge"] else 999)

    x = np.arange(len(datasets))
    width = 0.8 / len(libraries)

    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        times = []
        stds = []
        for ds in datasets:
            ds_result = next((r for r in lib_results if r["dataset_name"] == ds), None)
            if ds_result:
                times.append(ds_result["train_time_ms"])
                stds.append(ds_result["train_time_std"])
            else:
                times.append(0)
                stds.append(0)

        offset = (i - len(libraries) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            times,
            width,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
            yerr=stds,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Training Time (ms)")
    ax.set_title("Training Time Comparison")
    ax.set_xticks(x)

    # Create readable x-tick labels with sample counts
    xtick_labels = []
    for ds in datasets:
        ds_result = next((r for r in results if r["dataset_name"] == ds), None)
        if ds_result:
            n = ds_result["n_samples"]
            if n >= 1_000_000:
                label = f"{ds}\n({n/1_000_000:.1f}M)"
            elif n >= 1_000:
                label = f"{ds}\n({n/1_000:.0f}K)"
            else:
                label = f"{ds}\n({n})"
        else:
            label = ds
        xtick_labels.append(label)

    ax.set_xticklabels(xtick_labels)
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "training_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_time(results: list[dict], output_dir: Path):
    """Generate prediction time comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    libraries = sorted(set(r["library"] for r in results))
    datasets = sorted(set(r["dataset_name"] for r in results),
                      key=lambda x: ["tiny", "small", "medium", "large", "xlarge"].index(x)
                      if x in ["tiny", "small", "medium", "large", "xlarge"] else 999)

    x = np.arange(len(datasets))
    width = 0.8 / len(libraries)

    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        times = []
        stds = []
        for ds in datasets:
            ds_result = next((r for r in lib_results if r["dataset_name"] == ds), None)
            if ds_result:
                times.append(ds_result["predict_time_ms"])
                stds.append(ds_result["predict_time_std"])
            else:
                times.append(0)
                stds.append(0)

        offset = (i - len(libraries) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            times,
            width,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
            yerr=stds,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Prediction Time (ms)")
    ax.set_title("Prediction Time Comparison (Test Set)")
    ax.set_xticks(x)

    # Create readable x-tick labels
    xtick_labels = []
    for ds in datasets:
        ds_result = next((r for r in results if r["dataset_name"] == ds), None)
        if ds_result:
            n = ds_result["n_samples"]
            test_n = int(n * 0.2)  # 20% test set
            if test_n >= 1_000_000:
                label = f"{ds}\n({test_n/1_000_000:.1f}M)"
            elif test_n >= 1_000:
                label = f"{ds}\n({test_n/1_000:.0f}K)"
            else:
                label = f"{ds}\n({test_n})"
        else:
            label = ds
        xtick_labels.append(label)

    ax.set_xticklabels(xtick_labels)
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_path = output_dir / "prediction_time.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_speedup(results: list[dict], output_dir: Path):
    """Generate speedup comparison chart (relative to TreeBoost)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    libraries = sorted(set(r["library"] for r in results))
    datasets = sorted(set(r["dataset_name"] for r in results),
                      key=lambda x: ["tiny", "small", "medium", "large", "xlarge"].index(x)
                      if x in ["tiny", "small", "medium", "large", "xlarge"] else 999)

    # Check if treeboost results exist
    treeboost_results = {r["dataset_name"]: r for r in results if r["library"] == "treeboost"}
    if not treeboost_results:
        print("Warning: No TreeBoost results found for speedup comparison")
        return

    other_libs = [lib for lib in libraries if lib != "treeboost"]

    x = np.arange(len(datasets))
    width = 0.8 / len(other_libs) if other_libs else 0.8

    # Training speedup
    for i, lib in enumerate(other_libs):
        lib_results = [r for r in results if r["library"] == lib]
        speedups = []
        for ds in datasets:
            ds_result = next((r for r in lib_results if r["dataset_name"] == ds), None)
            tb_result = treeboost_results.get(ds)
            if ds_result and tb_result and tb_result["train_time_ms"] > 0:
                speedup = ds_result["train_time_ms"] / tb_result["train_time_ms"]
                speedups.append(speedup)
            else:
                speedups.append(0)

        offset = (i - len(other_libs) / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            speedups,
            width,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
        )

    ax1.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="TreeBoost baseline")
    ax1.set_xlabel("Dataset Size")
    ax1.set_ylabel("Relative Time (higher = slower)")
    ax1.set_title("Training Time Relative to TreeBoost")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(loc="upper left")

    # Add "TreeBoost faster" / "TreeBoost slower" annotations
    ylim = ax1.get_ylim()
    ax1.fill_between([-0.5, len(datasets) - 0.5], 0, 1, alpha=0.1, color="green")
    ax1.fill_between([-0.5, len(datasets) - 0.5], 1, ylim[1], alpha=0.1, color="red")
    ax1.text(len(datasets) - 0.6, 0.5, "TreeBoost faster", ha="right", va="center", fontsize=9, color="green", alpha=0.8)
    ax1.text(len(datasets) - 0.6, ylim[1] * 0.8, "TreeBoost slower", ha="right", va="center", fontsize=9, color="red", alpha=0.8)

    # Prediction speedup
    for i, lib in enumerate(other_libs):
        lib_results = [r for r in results if r["library"] == lib]
        speedups = []
        for ds in datasets:
            ds_result = next((r for r in lib_results if r["dataset_name"] == ds), None)
            tb_result = treeboost_results.get(ds)
            if ds_result and tb_result and tb_result["predict_time_ms"] > 0:
                speedup = ds_result["predict_time_ms"] / tb_result["predict_time_ms"]
                speedups.append(speedup)
            else:
                speedups.append(0)

        offset = (i - len(other_libs) / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            speedups,
            width,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
        )

    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5, label="TreeBoost baseline")
    ax2.set_xlabel("Dataset Size")
    ax2.set_ylabel("Relative Time (higher = slower)")
    ax2.set_title("Prediction Time Relative to TreeBoost")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(loc="upper left")

    # Add annotations
    ylim2 = ax2.get_ylim()
    ax2.fill_between([-0.5, len(datasets) - 0.5], 0, 1, alpha=0.1, color="green")
    ax2.fill_between([-0.5, len(datasets) - 0.5], 1, ylim2[1], alpha=0.1, color="red")
    ax2.text(len(datasets) - 0.6, 0.5, "TreeBoost faster", ha="right", va="center", fontsize=9, color="green", alpha=0.8)
    ax2.text(len(datasets) - 0.6, ylim2[1] * 0.8, "TreeBoost slower", ha="right", va="center", fontsize=9, color="red", alpha=0.8)

    plt.tight_layout()
    output_path = output_dir / "speedup.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy(results: list[dict], output_dir: Path):
    """Generate accuracy (R²) comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    libraries = sorted(set(r["library"] for r in results))
    datasets = sorted(set(r["dataset_name"] for r in results),
                      key=lambda x: ["tiny", "small", "medium", "large", "xlarge"].index(x)
                      if x in ["tiny", "small", "medium", "large", "xlarge"] else 999)

    x = np.arange(len(datasets))
    width = 0.8 / len(libraries)

    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        r2_scores = []
        for ds in datasets:
            ds_result = next((r for r in lib_results if r["dataset_name"] == ds), None)
            if ds_result:
                r2_scores.append(ds_result["r2"])
            else:
                r2_scores.append(0)

        offset = (i - len(libraries) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            r2_scores,
            width,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
        )

    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("R² Score")
    ax.set_title("Model Accuracy Comparison (R² Score)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_path = output_dir / "accuracy.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_scaling(results: list[dict], output_dir: Path):
    """Generate scaling (time vs dataset size) chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    libraries = sorted(set(r["library"] for r in results))

    # Sort datasets by sample count
    dataset_sizes = {}
    for r in results:
        dataset_sizes[r["dataset_name"]] = r["n_samples"]

    datasets = sorted(dataset_sizes.keys(), key=lambda x: dataset_sizes[x])

    # Training time scaling
    for lib in libraries:
        lib_results = sorted(
            [r for r in results if r["library"] == lib],
            key=lambda r: r["n_samples"]
        )
        if not lib_results:
            continue

        samples = [r["n_samples"] for r in lib_results]
        train_times = [r["train_time_ms"] for r in lib_results]

        ax1.plot(
            samples,
            train_times,
            marker=MARKERS.get(lib, "o"),
            markersize=8,
            linewidth=2,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
        )

    ax1.set_xlabel("Number of Samples")
    ax1.set_ylabel("Training Time (ms)")
    ax1.set_title("Training Time Scaling")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left")

    # Prediction time scaling
    for lib in libraries:
        lib_results = sorted(
            [r for r in results if r["library"] == lib],
            key=lambda r: r["n_samples"]
        )
        if not lib_results:
            continue

        samples = [int(r["n_samples"] * 0.2) for r in lib_results]  # Test set size
        predict_times = [r["predict_time_ms"] for r in lib_results]

        ax2.plot(
            samples,
            predict_times,
            marker=MARKERS.get(lib, "o"),
            markersize=8,
            linewidth=2,
            label=DISPLAY_NAMES.get(lib, lib),
            color=COLORS.get(lib, "#888888"),
        )

    ax2.set_xlabel("Number of Test Samples")
    ax2.set_ylabel("Prediction Time (ms)")
    ax2.set_title("Prediction Time Scaling")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    output_path = output_dir / "scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary(results: list[dict], output_dir: Path):
    """Generate a comprehensive summary chart."""
    fig = plt.figure(figsize=(16, 12))

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    libraries = sorted(set(r["library"] for r in results))
    datasets = sorted(set(r["dataset_name"] for r in results),
                      key=lambda x: ["tiny", "small", "medium", "large", "xlarge"].index(x)
                      if x in ["tiny", "small", "medium", "large", "xlarge"] else 999)

    x = np.arange(len(datasets))
    width = 0.8 / len(libraries)

    # 1. Training Time (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        times = [next((r["train_time_ms"] for r in lib_results if r["dataset_name"] == ds), 0) for ds in datasets]
        offset = (i - len(libraries) / 2 + 0.5) * width
        ax1.bar(x + offset, times, width, label=DISPLAY_NAMES.get(lib, lib), color=COLORS.get(lib, "#888888"))

    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Training Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Prediction Time (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        times = [next((r["predict_time_ms"] for r in lib_results if r["dataset_name"] == ds), 0) for ds in datasets]
        offset = (i - len(libraries) / 2 + 0.5) * width
        ax2.bar(x + offset, times, width, label=DISPLAY_NAMES.get(lib, lib), color=COLORS.get(lib, "#888888"))

    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Prediction Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(loc="upper left", fontsize=8)

    # 3. Scaling (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    for lib in libraries:
        lib_results = sorted([r for r in results if r["library"] == lib], key=lambda r: r["n_samples"])
        if lib_results:
            samples = [r["n_samples"] for r in lib_results]
            times = [r["train_time_ms"] for r in lib_results]
            ax3.plot(samples, times, marker=MARKERS.get(lib, "o"), markersize=6, linewidth=2,
                     label=DISPLAY_NAMES.get(lib, lib), color=COLORS.get(lib, "#888888"))

    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Training Time (ms)")
    ax3.set_title("Training Time Scaling")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.legend(loc="upper left", fontsize=8)

    # 4. Accuracy (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    for i, lib in enumerate(libraries):
        lib_results = [r for r in results if r["library"] == lib]
        r2_scores = [next((r["r2"] for r in lib_results if r["dataset_name"] == ds), 0) for ds in datasets]
        offset = (i - len(libraries) / 2 + 0.5) * width
        ax4.bar(x + offset, r2_scores, width, label=DISPLAY_NAMES.get(lib, lib), color=COLORS.get(lib, "#888888"))

    ax4.set_xlabel("Dataset")
    ax4.set_ylabel("R² Score")
    ax4.set_title("Model Accuracy")
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc="lower right", fontsize=8)

    # Main title
    fig.suptitle("GBDT Benchmark: TreeBoost vs XGBoost vs LightGBM vs CatBoost", fontsize=16, fontweight="bold", y=0.98)

    output_path = output_dir / "summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """Find the most recent benchmark results file."""
    json_files = list(results_dir.glob("benchmark_*.json"))
    if not json_files:
        return None
    return max(json_files, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Path to benchmark JSON file (or use --latest)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recent benchmark results"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for charts (default: same as input)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip generating summary chart"
    )
    args = parser.parse_args()

    setup_style()

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    # Find input file
    if args.latest:
        input_path = find_latest_results(results_dir)
        if not input_path:
            print("Error: No benchmark results found in results/")
            return 1
        print(f"Using latest results: {input_path}")
    elif args.input:
        input_path = Path(args.input)
    else:
        print("Error: Specify input file or use --latest")
        return 1

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    # Load results
    data = load_results(input_path)
    results = data.get("results", [])

    if not results:
        print("Error: No results found in file")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")

    # Generate all charts
    plot_training_time(results, output_dir)
    plot_prediction_time(results, output_dir)
    plot_speedup(results, output_dir)
    plot_accuracy(results, output_dir)
    plot_scaling(results, output_dir)

    if not args.no_summary:
        plot_summary(results, output_dir)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
