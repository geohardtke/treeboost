#!/usr/bin/env python3
"""
Benchmark TreeBoost against XGBoost, LightGBM, and CatBoost.

Compares training time, prediction time, and accuracy across various dataset sizes.
Supports both cross-library comparisons and CPU vs GPU comparisons.

Usage:
    # Cross-library CPU comparison (default)
    python benchmarks/benchmark.py

    # Cross-library GPU comparison (all libraries using GPU)
    python benchmarks/benchmark.py --mode cross-library-gpu

    # Single library CPU vs GPU comparison
    python benchmarks/benchmark.py --mode treeboost-gpu
    python benchmarks/benchmark.py --mode xgboost-gpu

    # All libraries CPU vs GPU comparison
    python benchmarks/benchmark.py --mode all-gpu

    # Other options
    python benchmarks/benchmark.py --iterations 5
    python benchmarks/benchmark.py --sizes small medium large
    python benchmarks/benchmark.py --skip catboost
    python benchmarks/benchmark.py --output results/my_benchmark.json

Requirements:
    pip install xgboost lightgbm catboost numpy scikit-learn
"""

import argparse
import contextlib
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr output (for hiding GPU compilation warnings)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

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
    device: str = "cpu"  # "cpu" or "gpu"


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
# TreeBoost training and prediction functions
# =============================================================================

def train_treeboost_cpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train TreeBoost model (CPU backend)."""
    config = GBDTConfig()
    config.num_rounds = n_rounds
    config.max_depth = max_depth
    config.learning_rate = learning_rate
    config.max_leaves = 31
    config.parallel_prediction = True
    config.column_reordering = True
    config.packed_dataset = True
    config.backend = "cpu"
    return GBDTModel.train(X_train, y_train, config)


def train_treeboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train TreeBoost model (GPU backend, auto-selects CUDA > WGPU)."""
    config = GBDTConfig()
    config.num_rounds = n_rounds
    config.max_depth = max_depth
    config.learning_rate = learning_rate
    config.max_leaves = 31
    config.parallel_prediction = True
    config.column_reordering = True
    config.packed_dataset = True
    config.backend = "gpu"  # Auto-select best GPU (CUDA > WGPU)
    return GBDTModel.train(X_train, y_train, config)


def predict_treeboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with TreeBoost model."""
    return model.predict(X)


# =============================================================================
# XGBoost training and prediction functions
# =============================================================================

def train_xgboost_cpu(
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


def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train XGBoost model (GPU via CUDA)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "verbosity": 0,
        "device": "cuda",  # Force GPU
        "tree_method": "hist",  # GPU histogram method
    }
    with suppress_stderr():
        return xgb.train(params, dtrain, num_boost_round=n_rounds)


def predict_xgboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with XGBoost model."""
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)


# =============================================================================
# LightGBM training and prediction functions
# =============================================================================

def train_lightgbm_cpu(
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


def train_lightgbm_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train LightGBM model (GPU via CUDA)."""
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "regression",
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "verbosity": -1,
        "device": "gpu",  # Force GPU
    }
    with suppress_stderr():
        return lgb.train(params, train_data, num_boost_round=n_rounds)


def predict_lightgbm(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with LightGBM model."""
    return model.predict(X)


# =============================================================================
# CatBoost training and prediction functions
# =============================================================================

def train_catboost_cpu(
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


def train_catboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
) -> Any:
    """Train CatBoost model (GPU via CUDA)."""
    model = CatBoostRegressor(
        iterations=n_rounds,
        depth=max_depth,
        learning_rate=learning_rate,
        verbose=False,
        task_type="GPU",  # Force GPU
    )
    with suppress_stderr():
        model.fit(X_train, y_train)
    return model


def predict_catboost(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict with CatBoost model."""
    return model.predict(X)


# Library configurations for cross-library comparison (CPU only)
LIBRARY_CONFIGS = {
    "treeboost": {
        "train": train_treeboost_cpu,
        "predict": predict_treeboost,
        "color": "#3498db",  # Blue
        "marker": "o",
    },
    "xgboost": {
        "train": train_xgboost_cpu,
        "predict": predict_xgboost,
        "color": "#e74c3c",  # Red
        "marker": "s",
    },
    "lightgbm": {
        "train": train_lightgbm_cpu,
        "predict": predict_lightgbm,
        "color": "#2ecc71",  # Green
        "marker": "^",
    },
    "catboost": {
        "train": train_catboost_cpu,
        "predict": predict_catboost,
        "color": "#9b59b6",  # Purple
        "marker": "D",
    },
}

# Library configurations for cross-library GPU comparison
LIBRARY_CONFIGS_GPU = {
    "treeboost": {
        "train": train_treeboost_gpu,
        "predict": predict_treeboost,
        "color": "#3498db",  # Blue
        "marker": "o",
    },
    "xgboost": {
        "train": train_xgboost_gpu,
        "predict": predict_xgboost,
        "color": "#e74c3c",  # Red
        "marker": "s",
    },
    "lightgbm": {
        "train": train_lightgbm_gpu,
        "predict": predict_lightgbm,
        "color": "#2ecc71",  # Green
        "marker": "^",
    },
    "catboost": {
        "train": train_catboost_gpu,
        "predict": predict_catboost,
        "color": "#9b59b6",  # Purple
        "marker": "D",
    },
}

# GPU comparison configurations
GPU_CONFIGS = {
    "treeboost": {
        "cpu_train": train_treeboost_cpu,
        "gpu_train": train_treeboost_gpu,
        "predict": predict_treeboost,
    },
    "xgboost": {
        "cpu_train": train_xgboost_cpu,
        "gpu_train": train_xgboost_gpu,
        "predict": predict_xgboost,
    },
    "lightgbm": {
        "cpu_train": train_lightgbm_cpu,
        "gpu_train": train_lightgbm_gpu,
        "predict": predict_lightgbm,
    },
    "catboost": {
        "cpu_train": train_catboost_cpu,
        "gpu_train": train_catboost_gpu,
        "predict": predict_catboost,
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
    device: str = "cpu",
    use_gpu_config: bool = False,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a single library and dataset."""
    if not LIBRARIES.get(library, False):
        return None

    if use_gpu_config:
        # Cross-library GPU comparison mode
        config = LIBRARY_CONFIGS_GPU[library]
        train_fn = config["train"]
        predict_fn = config["predict"]
        device = "gpu"
    elif device == "cpu":
        config = LIBRARY_CONFIGS[library]
        train_fn = config["train"]
        predict_fn = config["predict"]
    else:
        config = GPU_CONFIGS[library]
        train_fn = config["gpu_train"]
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
            device=device,
        )
    except Exception as e:
        print(f"  Error benchmarking {library} ({device}): {e}")
        return None


def run_all_benchmarks(
    datasets: list[str],
    libraries: list[str],
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
    iterations: int,
    use_gpu: bool = False,
) -> list[BenchmarkResult]:
    """Run benchmarks across all datasets and libraries."""
    results = []
    device_label = "GPU" if use_gpu else "CPU"

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

            print(f"\n  {lib} ({device_label}):")
            result = run_benchmark(
                lib,
                dataset_name,
                X_train, X_test,
                y_train, y_test,
                n_rounds,
                max_depth,
                learning_rate,
                iterations,
                use_gpu_config=use_gpu,
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


def run_gpu_comparison(
    datasets: list[str],
    libraries: list[str],
    n_rounds: int,
    max_depth: int,
    learning_rate: float,
    iterations: int,
) -> list[BenchmarkResult]:
    """Run CPU vs GPU benchmarks for specified libraries."""
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

            # CPU benchmark
            print(f"    CPU:")
            cpu_result = run_benchmark(
                lib,
                dataset_name,
                X_train, X_test,
                y_train, y_test,
                n_rounds,
                max_depth,
                learning_rate,
                iterations,
                device="cpu",
            )

            if cpu_result:
                print(f"      Train: {cpu_result.train_time_ms:>8.2f} ms (±{cpu_result.train_time_std:.2f})")
                print(f"      Predict: {cpu_result.predict_time_ms:>6.2f} ms (±{cpu_result.predict_time_std:.2f})")
                results.append(cpu_result)

            # GPU benchmark
            print(f"    GPU:")
            gpu_result = run_benchmark(
                lib,
                dataset_name,
                X_train, X_test,
                y_train, y_test,
                n_rounds,
                max_depth,
                learning_rate,
                iterations,
                device="gpu",
            )

            if gpu_result:
                print(f"      Train: {gpu_result.train_time_ms:>8.2f} ms (±{gpu_result.train_time_std:.2f})")
                print(f"      Predict: {gpu_result.predict_time_ms:>6.2f} ms (±{gpu_result.predict_time_std:.2f})")
                results.append(gpu_result)

            # Show speedup if both succeeded
            if cpu_result and gpu_result:
                speedup = cpu_result.train_time_ms / gpu_result.train_time_ms
                print(f"    GPU Speedup: {speedup:.2f}x")

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


def print_gpu_summary(results: list[BenchmarkResult]):
    """Print a GPU vs CPU summary table."""
    if not results:
        print("\nNo results to summarize.")
        return

    print("\n" + "=" * 90)
    print("GPU vs CPU COMPARISON")
    print("=" * 90)

    # Group by dataset and library
    datasets = sorted(set(r.dataset_name for r in results))
    libraries = sorted(set(r.library for r in results))

    print("\n--- Training Time (ms) and GPU Speedup ---")
    print(f"{'Dataset':<12}{'Library':<12}{'CPU (ms)':>12}{'GPU (ms)':>12}{'Speedup':>12}{'Winner':>10}")
    print("-" * 70)

    for ds in datasets:
        for lib in libraries:
            cpu_result = next((r for r in results if r.dataset_name == ds and r.library == lib and r.device == "cpu"), None)
            gpu_result = next((r for r in results if r.dataset_name == ds and r.library == lib and r.device == "gpu"), None)

            if cpu_result or gpu_result:
                cpu_time = f"{cpu_result.train_time_ms:.2f}" if cpu_result else "N/A"
                gpu_time = f"{gpu_result.train_time_ms:.2f}" if gpu_result else "N/A"

                if cpu_result and gpu_result:
                    speedup = cpu_result.train_time_ms / gpu_result.train_time_ms
                    winner = "GPU" if speedup > 1.0 else "CPU"
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "N/A"
                    winner = "N/A"

                print(f"{ds:<12}{lib:<12}{cpu_time:>12}{gpu_time:>12}{speedup_str:>12}{winner:>10}")


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
    parser.add_argument(
        "--mode",
        type=str,
        default="cross-library",
        choices=["cross-library", "cross-library-gpu", "treeboost-gpu", "xgboost-gpu", "lightgbm-gpu", "catboost-gpu", "all-gpu"],
        help="Benchmark mode: cross-library (CPU, default), cross-library-gpu (all libs GPU), treeboost-gpu (CPU vs GPU), all-gpu (all libs CPU vs GPU)"
    )
    args = parser.parse_args()

    # Determine which libraries to benchmark based on mode
    if args.mode == "cross-library":
        if args.only:
            libraries = args.only
        else:
            libraries = [lib for lib in LIBRARY_CONFIGS.keys() if lib not in args.skip]
        is_gpu_mode = False
        is_cross_gpu = False
    elif args.mode == "cross-library-gpu":
        if args.only:
            libraries = args.only
        else:
            libraries = [lib for lib in LIBRARY_CONFIGS_GPU.keys() if lib not in args.skip]
        is_gpu_mode = False
        is_cross_gpu = True
    elif args.mode == "all-gpu":
        libraries = [lib for lib in GPU_CONFIGS.keys() if lib not in args.skip]
        is_gpu_mode = True
        is_cross_gpu = False
    else:
        # Single library GPU comparison (e.g., treeboost-gpu, xgboost-gpu)
        lib_name = args.mode.replace("-gpu", "")
        libraries = [lib_name]
        is_gpu_mode = True
        is_cross_gpu = False

    # Check at least one library is available
    available = [lib for lib in libraries if LIBRARIES.get(lib, False)]
    if not available:
        print("Error: No libraries available for benchmarking.")
        print("Install with: pip install treeboost xgboost lightgbm catboost")
        return 1

    print("=" * 70)
    if is_gpu_mode:
        print(f"GBDT BENCHMARK: CPU vs GPU Comparison")
    elif is_cross_gpu:
        print("GBDT BENCHMARK: TreeBoost vs XGBoost vs LightGBM vs CatBoost (GPU)")
    else:
        print("GBDT BENCHMARK: TreeBoost vs XGBoost vs LightGBM vs CatBoost (CPU)")
    print("=" * 70)

    system_info = get_system_info()
    print(f"\nPlatform: {system_info.platform}")
    print(f"Python: {system_info.python_version}")
    print(f"CPU cores: {system_info.cpu_count}")

    if is_gpu_mode:
        print(f"Mode: CPU vs GPU comparison")
    elif is_cross_gpu:
        print(f"Device: GPU (all libraries using GPU)")
    else:
        print(f"Device: CPU (all libraries using CPU)")

    print(f"\nLibraries: {', '.join(available)}")
    print(f"Datasets: {', '.join(args.sizes)}")
    print(f"Settings: {args.rounds} rounds, max_depth={args.max_depth}, lr={args.learning_rate}")
    print(f"Iterations: {args.iterations}")

    # Run benchmarks
    if is_gpu_mode:
        results = run_gpu_comparison(
            datasets=args.sizes,
            libraries=libraries,
            n_rounds=args.rounds,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
        )
        print_gpu_summary(results)
    elif is_cross_gpu:
        results = run_all_benchmarks(
            datasets=args.sizes,
            libraries=libraries,
            n_rounds=args.rounds,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            use_gpu=True,
        )
        print_summary(results)
    else:
        results = run_all_benchmarks(
            datasets=args.sizes,
            libraries=libraries,
            n_rounds=args.rounds,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            use_gpu=False,
        )
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
            mode_suffix = f"_{args.mode}" if args.mode != "cross-library" else ""
            output_path = results_dir / f"benchmark{mode_suffix}_{timestamp}.json"

        save_results(results, output_path, system_info)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
