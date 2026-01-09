# TreeBoost

[![Crates.io](https://img.shields.io/crates/v/treeboost.svg)](https://crates.io/crates/treeboost) [![Docs](https://img.shields.io/docsrs/treeboost)](https://docs.rs/treeboost) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

![TreeBoost](images/treeboost.jpeg)

> **Practical tabular ML for messy, real-world data. Fast baselines first, deep control when you need it.**

TreeBoost is a Rust-first library for tabular machine learning that starts simple and scales to expert use. It is built for the reality of production datasets: time series, missing values, drift, mixed feature types, and noisy labels. You get a clean path from “just give me a working model” to full control over training, backends, and constraints.

## At a Glance

- AutoML and AutoTuner for fast, explainable baselines
- Hybrid Linear+Tree mode for trend extrapolation + non-linear interactions
- Built-in preprocessing that serializes with your model
- GPU acceleration (WebGPU, CUDA) plus AVX-512/SVE2 CPU backends
- Zero-copy serialization and incremental TRB updates for production pipelines

## Why TreeBoost

Most libraries are tuned for leaderboard-style modeling. TreeBoost is built for shipping models:

- **Fast baseline in one call** for beginners and teams under time pressure.
- **White-box AutoML** that explains why it chose a mode and lets you iterate.
- **Upgradeable control** without rewriting your data pipeline.
- **Deployment-friendly**: zero-copy serialization, fast inference, and a CLI for batch jobs.

## Three API Levels (Start Simple, Go Deep)

- **AutoModel** — One call trains a solid baseline and produces a model you can ship. Export a `config.json` when you want to improve it later.
- **UniversalModel** — Choose the learning mode (PureTree, LinearThenTree, RandomForest) and tune it without leaving a high-level API.
- **GBDTModel** — Lowest-level API for maximum control, backend selection, and benchmarking.

You can move between these levels without changing your dataset format.

📖 **See [docs/API.md](docs/API.md) for complete API documentation with examples.**

## Recommended Workflow

1. **AutoModel** for a strong baseline and a training report.
2. **Inspect the report and config** to understand the model choice.
3. **Refine with UniversalModel or GBDTModel** for extra accuracy, constraints, or incremental updates.

## Quick Start (AutoModel)

```rust
use treeboost::{auto_train, AutoModel};

let model = auto_train(&df, "target")?;
println!("{}", model.summary());

model.save("model.rkyv")?;
model.save_config("config.json")?;
```

This gives you a deployable model plus a config you can tweak later.

## Expected Outputs (After Training)

After training, you typically save:

- `model.rkyv` for fast inference and deployment
- `config.json` for reproducible retraining or fine-tuning
- `model.trb` if you want incremental updates later

```rust
let model = auto_train(&df, "target")?;
model.save("model.rkyv")?;
model.save_config("config.json")?;
model.save_trb("model.trb", "initial training")?;
```

**Example `config.json` (abridged):**

```json
{
  "mode": "LinearThenTree",
  "num_rounds": 120,
  "learning_rate": 0.1,
  "subsample": 0.9,
  "validation_ratio": 0.1,
  "early_stopping_rounds": 20,
  "linear_rounds": 10,
  "tree_config": {
    "max_depth": 6,
    "max_leaves": 31,
    "lambda": 1.0,
    "min_samples_leaf": 20,
    "colsample": 0.8
  },
  "linear_config": {
    "lambda": 1.0,
    "l1_ratio": 0.0,
    "shrinkage_factor": 0.3
  }
}
```

The actual file includes the full set of tree/linear fields so you can tweak every detail.

## Quick Start (UniversalModel)

```rust
use treeboost::{UniversalConfig, UniversalModel, BoostingMode};
use treeboost::dataset::DatasetLoader;
use treeboost::loss::MseLoss;

let loader = DatasetLoader::new(255);
let dataset = loader.load_parquet("data.parquet", "target", None)?;

let config = UniversalConfig::new()
    .with_mode(BoostingMode::LinearThenTree)
    .with_num_rounds(100)
    .with_linear_rounds(10)
    .with_learning_rate(0.1);

let model = UniversalModel::train(&dataset, config, &MseLoss)?;
let predictions = model.predict(&dataset);
```

**Quick mode selection:**

| Your Data                                  | Use This Mode                  |
| ------------------------------------------ | ------------------------------ |
| General tabular, categoricals              | `BoostingMode::PureTree`       |
| Time-series, trending, needs extrapolation | `BoostingMode::LinearThenTree` |
| Noisy data, want robustness                | `BoostingMode::RandomForest`   |

## Quick Start (GBDTModel)

```rust
use treeboost::{GBDTConfig, GBDTModel};

let config = GBDTConfig::new()
    .with_num_rounds(200)
    .with_max_depth(6)
    .with_learning_rate(0.05);

let model = GBDTModel::train(&features, num_features, &targets, config, None)?;
```

## Python (GBDTModel)

```python
import numpy as np
from treeboost import GBDTConfig, GBDTModel

X = np.random.randn(10000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(10000) * 0.1).astype(np.float32)

config = GBDTConfig()
config.num_rounds = 100
config.max_depth = 6
config.learning_rate = 0.1

model = GBDTModel.train(X, y, config)
```

## What You Get

- **AutoML mode selection** that evaluates probes and explains its choice.
- **Hybrid Linear+Tree architecture** for trend extrapolation and interactions.
- **Built-in preprocessing**: encoders, scalers, and imputers that serialize with the model.
- **Linear Trees** for piecewise-linear data with far fewer trees.
- **Conformal prediction** for uncertainty intervals.
- **Incremental learning** via TRB format with drift detection.

## Backends (Automatic by Default)

TreeBoost auto-selects the fastest backend. You can override it if needed.

```rust
use treeboost::{GBDTConfig, GBDTModel};
use treeboost::backend::BackendType;

let config = GBDTConfig::new()
    .with_backend(BackendType::Scalar);

let model = GBDTModel::train(&features, num_features, &targets, config, None)?;
```

Supported backends: Scalar, AVX-512, SVE2, WGPU, CUDA.

## CLI

```bash
# Train a model
treeboost train --data data.csv --target price --output model.rkyv

# Predict
treeboost predict --model model.rkyv --data test.csv --output predictions.json
```

Run `treeboost --help` for full options.

## Installation

```bash
cargo add treeboost
```

```bash
# Python bindings (requires Rust toolchain + maturin)
pip install treeboost
```

Feature flags: `gpu`, `cuda`, `mmap`, `python`.

## Project Links

- **API Reference**: [docs/API.md](docs/API.md) - Complete API documentation with examples
- Docs: https://docs.rs/treeboost
- Crate: https://crates.io/crates/treeboost
- GitHub: https://github.com/ml-rust/treeboost

## License

Apache License 2.0
