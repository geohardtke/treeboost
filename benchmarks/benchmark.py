#!/usr/bin/env python3
"""
Benchmark TreeBoost against XGBoost, LightGBM, and CatBoost.

Compares training time, prediction time, and accuracy across various dataset sizes.
All libraries are forced to use CPU-only for fair comparison.

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --iterations 5
    python benchmarks/benchmark.py --sizes small medium large
    python benchmarks/benchmark.py --skip catboost  # Skip slow libraries
    python benchmarks/benchmark.py --output results/my_benchmark.json

Requirements:
    pip install xgboost lightgbm catboost numpy scikit-learn
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Check available libraries
LIBRARIES = {}

try:
    from treeboost import GBDTConfig, GBDTModel
    LIBRARIES["treeboost"] = True
except ImportError:
    LIBRARIES["treeboost"] = False
    print("Warning: treeboost not installed. Run: pip install -e .")

try:
    import xgboost as xgb
    LIBRARIES["xgboost"] = True
except ImportError:
    LIBRARIES["xgboost"] = False
    print("Warning: xgboost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIBRARIES["lightgbm"] = True
except ImportError:
    LIBRARIES["lightgbm"] = False
    print("Warning: lightgbm not installed. Run: pip install lightgbm")

try:
    from catboost import CatBoostRegressor, Pool
    LIBRARIES["catboost"] = True
except ImportError:
    LIBRARIES["catboost"] = False
    print("Warning: catboost not installed. Run: pip install catboost")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    library: str
    dataset_name: str
    n_samples: int
    n_features: int
    train_time_ms: float
    train_time_std: float
    predict_time_ms: float
    predict_time_std: float
    mse: float
    r2: float
    iterations: int


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    platform: str
    python_version: str
    cpu_count: int
    timestamp: str


def get_system_info() -> SystemInfo:
    """Collect system information."""
    return SystemInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        timestamp=datetime.now().isoformat(),
    )


def generate_regression_data(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression data.

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create target with non-linear relationships
    y = (
        X[:, 0] * 2 +
        X[:, 1] ** 2 +
        np.sin(X[:, 2] * 2) +
        X[:, 3] * X[:, 4] +
        noise * np.random.randn(n_samples)
    ).astype(np.float32)

    # Train/test split (80/20)
    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Compute MSE and R2 score."""
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return float(mse), float(r2)


def benchmark_function(
    func: Callable,
    iterations: int = 3,
    warmup: int = 1,
) -> tuple[float, float]:
    """Benchmark a function and return mean and std in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    gc.collect()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_ms, std_ms


# =============================================================================
# Library-specific training and prediction functions
# =============================================================================

def train_treeboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train TreeBoost model (CPU, with all optimizations enabled)."""
    config = GBDTConfig()
    config.num_rounds = n_rounds
    config.max_depth = max_depth
    config.learning_rate = learning_rate
    config.max_leaves = 31
    # Ensure optimizations are enabled (should be default, but be explicit)
    config.parallel_prediction = True
    config.column_reordering = True
    config.packed_dataset = True
    return GBDTModel.train(X_train, y_train, config)


def predict_treeboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with TreeBoost model."""
    return model.predict(X)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train XGBoost model (CPU only)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "verbosity": 0,
        "nthread": -1,
        "device": "cpu",  # Force CPU
    }
    return xgb.train(params, dtrain, num_boost_round=n_rounds)


def predict_xgboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with XGBoost model."""
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train LightGBM model (CPU only)."""
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "verbosity": -1,
        "num_threads": -1,
        "device": "cpu",  # Force CPU
    }
    return lgb.train(params, train_data, num_boost_round=n_rounds)


def predict_lightgbm(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with LightGBM model."""
    return model.predict(X)


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train CatBoost model (CPU only)."""
    model = CatBoostRegressor(
        iterations=n_rounds,
        depth=max_depth,
        learning_rate=learning_rate,
        verbose=False,
        thread_count=-1,
        task_type="CPU",  # Force CPU
    )
    model.fit(X_train, y_train)
    return model


def predict_catboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with CatBoost model."""
    return model.predict(X)


# Library configurations
LIBRARY_CONFIGS = {
    "treeboost": {
        "train": train_treeboost,
        "predict": predict_treeboost,
        "color": "#3498db",  # Blue
        "marker": "o",
    },
    "xgboost": {
        "train": train_xgboost,
        "predict": predict_xgboost,
        "color": "#e74c3c",  # Red
        "marker": "s",
    },
    "lightgbm": {
        "train": train_lightgbm,
        "predict": predict_lightgbm,
        "color": "#2ecc71",  # Green
        "marker": "^",
    },
    "catboost": {
        "train": train_catboost,
        "predict": predict_catboost,
        "color": "#9b59b6",  # Purple
        "marker": "D",
    },
}


# Dataset configurations
DATASET_CONFIGS = {
    "tiny": {"n_samples": 1_000, "n_features": 10},
    "small": {"n_samples": 10_000, "n_features": 20},
    "medium": {"n_samples": 100_000, "n_features": 50},
    "large": {"n_samples": 500_000, "n_features": 100},
    "xlarge": {"n_samples": 1_000_000, "n_features": 100},
}


def run_benchmark(
    library: str,
    dataset_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
    iterations: int,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single library and dataset."""
    if not LIBRARIES.get(library, False):
        return None

    config = LIBRARY_CONFIGS[library]
    train_fn = config["train"]
    predict_fn = config["predict"]

    try:
        # Benchmark training
        model = None
        def do_train():
            nonlocal model
            model = train_fn(X_train, y_train, n_rounds, max_depth, learning_rate)

        train_mean, train_std = benchmark_function(do_train, iterations=iterations)

        # Benchmark prediction
        def do_predict():
            return predict_fn(model, X_test)

        predict_mean, predict_std = benchmark_function(do_predict, iterations=iterations)

        # Compute accuracy metrics
        y_pred = predict_fn(model, X_test)
        mse, r2 = compute_metrics(y_test, y_pred)

        return BenchmarkResult(
            library=library,
            dataset_name=dataset_name,
            n_samples=len(X_train) + len(X_test),
            n_features=X_train.shape[1],
            train_time_ms=train_mean,
            train_time_std=train_std,
            predict_time_ms=predict_mean,
            predict_time_std=predict_std,
            mse=mse,
            r2=r2,
            iterations=iterations,
        )
    except Exception as e:
        print(f"  Error benchmarking {library}: {e}")
        return None


def run_all_benchmarks(
    datasets: list[str],
    libraries: list[str],
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
    iterations: int,
) -> list[BenchmarkResult]:
    """Run benchmarks across all datasets and libraries."""
    results = []

    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {dataset_name}")
            continue

        config = DATASET_CONFIGS[dataset_name]
        n_samples = config["n_samples"]
        n_features = config["n_features"]

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name} ({n_samples:,} samples, {n_features} features)")
        print("=" * 70)

        # Generate data
        print("Generating data...")
        X_train, X_test, y_train, y_test = generate_regression_data(
            n_samples, n_features
        )
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        for lib in libraries:
            if not LIBRARIES.get(lib, False):
                print(f"\n  {lib}: SKIPPED (not installed)")
                continue

            print(f"\n  {lib}:")
            result = run_benchmark(
                lib,
                dataset_name,
                X_train, X_test,
                y_train, y_test,
                n_rounds,
                max_depth,
                learning_rate,
                iterations,
            )

            if result:
                print(f"    Train: {result.train_time_ms:>8.2f} ms (±{result.train_time_std:.2f})")
                print(f"    Predict: {result.predict_time_ms:>6.2f} ms (±{result.predict_time_std:.2f})")
                print(f"    MSE: {result.mse:.6f}, R²: {result.r2:.4f}")
                results.append(result)

        # Clean up
        del X_train, X_test, y_train, y_test
        gc.collect()

    return results


def save_results(results: list[BenchmarkResult], output_path: Path, system_info: SystemInfo):
    """Save benchmark results to JSON file."""
    data = {
        "metadata": {
            "system": asdict(system_info),
            "timestamp": datetime.now().isoformat(),
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: list[BenchmarkResult]):
    """Print a summary table of results."""
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    # Group by dataset
    datasets = sorted(set(r.dataset_name for r in results))
    libraries = sorted(set(r.library for r in results))

    # Training time comparison
    print("\n--- Training Time (ms) ---")
    print(f"{'Dataset':<12}", end="")
    for lib in libraries:
        print(f"{lib:>14}", end="")
    print()
    print("-" * (12 + 14 * len(libraries)))

    for ds in datasets:
        print(f"{ds:<12}", end="")
        ds_results = {r.library: r for r in results if r.dataset_name == ds}
        for lib in libraries:
            if lib in ds_results:
                print(f"{ds_results[lib].train_time_ms:>14.2f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

    # Prediction time comparison
    print("\n--- Prediction Time (ms) ---")
    print(f"{'Dataset':<12}", end="")
    for lib in libraries:
        print(f"{lib:>14}", end="")
    print()
    print("-" * (12 + 14 * len(libraries)))

    for ds in datasets:
        print(f"{ds:<12}", end="")
        ds_results = {r.library: r for r in results if r.dataset_name == ds}
        for lib in libraries:
            if lib in ds_results:
                print(f"{ds_results[lib].predict_time_ms:>14.2f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

    # R² comparison
    print("\n--- R² Score ---")
    print(f"{'Dataset':<12}", end="")
    for lib in libraries:
        print(f"{lib:>14}", end="")
    print()
    print("-" * (12 + 14 * len(libraries)))

    for ds in datasets:
        print(f"{ds:<12}", end="")
        ds_results = {r.library: r for r in results if r.dataset_name == ds}
        for lib in libraries:
            if lib in ds_results:
                print(f"{ds_results[lib].r2:>14.4f}", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

    # Speedup vs TreeBoost
    print("\n--- Speedup vs TreeBoost (Training) ---")
    print(f"{'Dataset':<12}", end="")
    for lib in libraries:
        if lib != "treeboost":
            print(f"{lib:>14}", end="")
    print()
    print("-" * (12 + 14 * (len(libraries) - 1)))

    for ds in datasets:
        print(f"{ds:<12}", end="")
        ds_results = {r.library: r for r in results if r.dataset_name == ds}
        treeboost_time = ds_results.get("treeboost", None)
        if treeboost_time:
            for lib in libraries:
                if lib != "treeboost" and lib in ds_results:
                    speedup = ds_results[lib].train_time_ms / treeboost_time.train_time_ms
                    marker = "faster" if speedup > 1 else "slower"
                    print(f"{speedup:>10.2f}x {marker[:3]:>3}", end="")
                elif lib != "treeboost":
                    print(f"{'N/A':>14}", end="")
        else:
            for lib in libraries:
                if lib != "treeboost":
                    print(f"{'N/A':>14}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TreeBoost against other GBDT libraries"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset sizes to benchmark (default: small medium large)"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=list(LIBRARY_CONFIGS.keys()),
        help="Libraries to skip"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=list(LIBRARY_CONFIGS.keys()),
        help="Only benchmark these libraries"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of boosting rounds (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: 6)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    args = parser.parse_args()

    # Determine which libraries to benchmark
    if args.only:
        libraries = args.only
    else:
        libraries = [lib for lib in LIBRARY_CONFIGS.keys() if lib not in args.skip]

    # Check at least one library is available
    available = [lib for lib in libraries if LIBRARIES.get(lib, False)]
    if not available:
        print("Error: No libraries available for benchmarking.")
        print("Install with: pip install treeboost xgboost lightgbm catboost")
        return 1

    print("=" * 70)
    print("GBDT BENCHMARK: TreeBoost vs XGBoost vs LightGBM vs CatBoost")
    print("=" * 70)

    system_info = get_system_info()
    print(f"\nPlatform: {system_info.platform}")
    print(f"Python: {system_info.python_version}")
    print(f"CPU cores: {system_info.cpu_count}")
    print(f"Device: CPU (all libraries forced to CPU-only)")

    print(f"\nLibraries: {', '.join(available)}")
    print(f"Datasets: {', '.join(args.sizes)}")
    print(f"Settings: {args.rounds} rounds, max_depth={args.max_depth}, lr={args.learning_rate}")
    print(f"Iterations: {args.iterations}")

    # Run benchmarks
    results = run_all_benchmarks(
        datasets=args.sizes,
        libraries=libraries,
        n_rounds=args.rounds,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save and results:
        script_dir = Path(__file__).parent
        results_dir = script_dir / "results"
        results_dir.mkdir(exist_ok=True)

        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"benchmark_{timestamp}.json"

        save_results(results, output_path, system_info)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
