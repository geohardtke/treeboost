# TreeBoost

A high-performance, production-ready Gradient Boosted Decision Tree (GBDT) engine written in pure Rust. Designed for tabular data with robust handling of dirty/noisy data, TreeBoost combines classical GBDT with modern robustness techniques and performance optimizations.

## Features

### Core Capabilities

- **Histogram-based training**: Memory-efficient u8 binning (up to 256 bins per feature) for large datasets
- **Shannon Entropy regularization**: Drift-resilient splitting objectives with configurable regularization weight
- **Pseudo-Huber loss**: Robust regression with outlier resistance (differentiable approximation to Huber loss)
- **Mean Squared Error (MSE) loss**: Standard regression objective
- **Split Conformal Prediction**: Distribution-free prediction intervals with finite-sample coverage guarantees
- **Zero-copy serialization**: Fast model loading via rkyv with memory mapping support

### Robustness Features

- **Ordered Target Encoding**: High-cardinality categorical features without label leakage
- **Count-Min Sketch filtering**: Automatic rare category handling for dirty data
- **M-Estimate smoothing**: Bayesian regularization for categorical encoding
- **Best-First (Leaf-wise) tree growth**: Optimizes splits greedily for maximum gain reduction
- **Early stopping**: Validation-based stopping to prevent overfitting

### Performance Optimizations

- **Feature-parallel histogram construction**: Rayon-powered work-stealing scheduler
- **Column reordering by importance**: Cache-optimized prediction paths
- **4-bit packed datasets**: Reduced memory footprint for low-cardinality features
- **Parallel predictions**: SIMD-friendly batch inference
- **Histogram Subtraction Trick**: O(n) split finding without full histogram recomputation

### Advanced Constraints

- **Monotonic constraints**: Enforce increasing/decreasing relationships per feature
- **Interaction constraints**: Control which features can interact in splits
- **Configurable regularization**: L2 leaf weight regularization (lambda) and entropy-based split penalties

## Installation

### From PyPI (Python)

```bash
pip install treeboost
```

### From Source (Rust)

```bash
cd TreeBoost
cargo build --release
```

### Building Python Bindings from Source

```bash
pip install maturin
cd TreeBoost
maturin develop --release
```

## Quick Start

### Python

```python
import numpy as np
from treeboost import GBDTConfig, GBDTModel

# Generate or load your data
X = np.random.randn(10000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + 0.1 * np.random.randn(10000)).astype(np.float32)

# Create and configure model
config = GBDTConfig()
config.num_rounds = 100          # Number of boosting iterations
config.max_depth = 6             # Maximum tree depth
config.learning_rate = 0.1       # Shrinkage rate
config.max_leaves = 31           # Maximum leaves per tree
config.entropy_weight = 0.1      # Shannon entropy regularization

# Train model
model = GBDTModel.train(X, y, config)

# Make predictions
predictions = model.predict(X)

# Get prediction intervals (if conformal prediction was enabled)
if model.conformal_quantile() is not None:
    predictions, lower, upper = model.predict_with_intervals(X)
    print(f"Predictions: {predictions[:5]}")
    print(f"90% Intervals: [{lower[:5]}, {upper[:5]}]")

# Feature importances
importances = model.feature_importances()

# Save and load
model.save("my_model.rkyv")
loaded_model = GBDTModel.load("my_model.rkyv")
```

### Rust

```rust
use treeboost::{GBDTConfig, GBDTModel};
use treeboost::dataset::DatasetLoader;

// Load data from Parquet or CSV
let loader = DatasetLoader::new(255);  // 255 bins per feature
let dataset = loader.load_parquet("data.parquet", "target_column", None)?;

// Configure training
let config = GBDTConfig::new()
    .with_num_rounds(100)
    .with_max_depth(6)
    .with_learning_rate(0.1)
    .with_pseudo_huber_loss(1.0)      // Delta for pseudo-Huber loss
    .with_entropy_weight(0.1)          // Shannon entropy regularization
    .with_subsample(0.8)               // Row subsampling (80%)
    .with_colsample(0.8);              // Column subsampling (80%)

// Train model
let model = GBDTModel::train(&dataset, config)?;

// Make predictions
let predictions = model.predict(&dataset);

// Save model
treeboost::serialize::save_model(&model, "model.rkyv")?;
```

## CLI Usage

TreeBoost includes a command-line interface for common tasks.

### Training

```bash
treeboost train \
  --data data.csv \
  --target price \
  --output model.rkyv \
  --rounds 100 \
  --max-depth 6 \
  --learning-rate 0.1 \
  --max-leaves 31 \
  --entropy-weight 0.1 \
  --min-samples-leaf 20 \
  --loss mse
```

**Training Options:**

| Option                 | Default  | Description                                                                                            |
| ---------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `--data`               | required | Input data file (CSV or Parquet)                                                                       |
| `--target`             | required | Target column name                                                                                     |
| `--output`             | required | Output model path (`.rkyv`)                                                                            |
| `--rounds`             | 100      | Number of boosting rounds                                                                              |
| `--max-depth`          | 6        | Maximum tree depth                                                                                     |
| `--max-leaves`         | 31       | Maximum leaves per tree                                                                                |
| `--learning-rate`      | 0.1      | Shrinkage rate (0.0-1.0)                                                                               |
| `--min-samples-leaf`   | 20       | Minimum samples per leaf                                                                               |
| `--lambda`             | 1.0      | L2 leaf regularization                                                                                 |
| `--entropy-weight`     | 0.0      | Shannon entropy regularization weight                                                                  |
| `--subsample`          | 1.0      | Row subsampling ratio (0.0-1.0)                                                                        |
| `--colsample`          | 1.0      | Column subsampling ratio (0.0-1.0)                                                                     |
| `--loss`               | mse      | Loss function: `mse` or `huber`                                                                        |
| `--huber-delta`        | 1.0      | Delta for pseudo-Huber loss                                                                            |
| `--num-bins`           | 255      | Feature discretization bins (1-255)                                                                    |
| `--early-stopping`     | 0        | Validation rounds before stopping (0 to disable)                                                       |
| `--validation-ratio`   | 0.1      | Validation set ratio for early stopping                                                                |
| `--conformal`          | —        | Calibration ratio for conformal prediction (e.g., `0.2` for 20%)                                       |
| `--conformal-quantile` | 0.9      | Desired coverage level (e.g., 0.9 for 90%)                                                             |
| `--monotonic`          | —        | Monotonic constraints (comma-separated: `+1`=increasing, `-1`=decreasing, `0`=none)                    |
| `--interactions`       | —        | Interaction groups (semicolon-separated: `0,1,2;3,4` means groups can interact within but not between) |
| `--features`           | —        | Feature columns to use (comma-separated; all if omitted)                                               |
| `--no-parallel`        | —        | Disable parallel prediction                                                                            |
| `--no-reorder`         | —        | Disable column reordering optimization                                                                 |
| `--no-pack`            | —        | Disable 4-bit packing optimization                                                                     |
| `--no-optimizations`   | —        | Disable all performance optimizations                                                                  |

**Example with Constraints:**

```bash
treeboost train \
  --data housing.csv \
  --target price \
  --output model.rkyv \
  --rounds 150 \
  --max-depth 7 \
  --loss huber \
  --huber-delta 1.0 \
  --entropy-weight 0.05 \
  --monotonic "+1,0,-1,0,+1" \
  --early-stopping 10 \
  --validation-ratio 0.15 \
  --conformal 0.2 \
  --conformal-quantile 0.9
```

### Prediction

```bash
treeboost predict \
  --model model.rkyv \
  --data test_data.csv \
  --output predictions.json
```

**Prediction Options:**

| Option        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| `--model`     | Path to trained model (`.rkyv`)                                    |
| `--data`      | Input data file (CSV or Parquet)                                   |
| `--output`    | Output predictions file (JSON)                                     |
| `--target`    | Target column name (optional, for evaluation metrics)              |
| `--intervals` | Include prediction intervals (if conformal prediction was enabled) |

**Output Format (without intervals):**

```json
[
  {"row": 0, "prediction": 42.5},
  {"row": 1, "prediction": 39.2},
  ...
]
```

**Output Format (with intervals):**

```json
[
  {"row": 0, "prediction": 42.5, "lower": 38.2, "upper": 46.8},
  {"row": 1, "prediction": 39.2, "lower": 35.1, "upper": 43.3},
  ...
]
```

### Model Inspection

```bash
treeboost info \
  --model model.rkyv \
  --importances \
  --num-features 20
```

**Output includes:**

- Model statistics (number of trees, base prediction)
- Training configuration (all hyperparameters)
- Feature importances (sum of split gains per feature)
- Conformal quantile (if applicable)

## Configuration Reference

### Core Training Parameters

```python
config = GBDTConfig()

# Ensemble
config.num_rounds = 100              # Boosting iterations
config.learning_rate = 0.1           # Shrinkage (lower = slower, more stable)

# Tree structure
config.max_depth = 6                 # Max depth (deeper = more complex)
config.max_leaves = 31               # Max leaves (2^depth - 1 is maximum)
config.min_samples_leaf = 20         # Leaf size threshold
config.min_hessian_leaf = 1.0        # Hessian sum threshold per leaf

# Regularization
config.lambda = 1.0                  # L2 leaf weight penalty
config.entropy_weight = 0.0          # Shannon entropy penalty (0 = disabled)
config.min_gain = 0.0                # Minimum gain to accept split

# Subsampling (Stochastic Gradient Boosting)
config.subsample = 1.0               # Row subsampling (1.0 = no subsampling)
config.colsample = 1.0               # Column subsampling (1.0 = all columns)

# Loss function
config.loss_type = "mse"             # "mse" or "huber"

# Conformal prediction
config.calibration_ratio = 0.0       # Split ratio for calibration (0 = disabled)
config.conformal_quantile = 0.9      # Coverage level (e.g., 0.9 = 90%)

# Early stopping
config.early_stopping_rounds = 0     # Rounds without improvement (0 = disabled)
config.validation_ratio = 0.1        # Validation set ratio

# Performance tuning
config.parallel_prediction = True    # Use Rayon for batch inference
config.column_reordering = True      # Reorder columns by importance
config.packed_dataset = True         # Use 4-bit packing for low-cardinality features

# Constraints
config.monotonic_constraints = [...]  # Per-feature monotonicity
config.interaction_groups = [...]     # Feature interaction groups
```

### Loss Functions

**MSE Loss** (default)

- Standard mean squared error: `L = (y - ŷ)²`
- Appropriate for normally distributed targets
- Sensitive to outliers

```python
config.loss_type = "mse"
```

**Pseudo-Huber Loss**

- Differentiable approximation to Huber loss: smooth around zero, linear in tails
- Robust to outliers with configurable transition point
- Delta controls the transition from quadratic to linear

```python
config.loss_type = "huber"
config.huber_delta = 1.0  # Adjust based on target scale
```

### Entropy Regularization

Shannon Entropy weight controls drift-resilience:

```python
config.entropy_weight = 0.1  # Balance between gain and entropy
```

Higher values prioritize balanced, stable splits at the cost of fitting capacity. Recommended for time-series or noisy data.

### Conformal Prediction

Enable distribution-free prediction intervals:

```python
config.calibration_ratio = 0.2       # Reserve 20% for calibration
config.conformal_quantile = 0.9      # 90% coverage guarantee
model = GBDTModel.train(X, y, config)

# Use calibrated model
predictions, lower, upper = model.predict_with_intervals(X_test)
```

The coverage is guaranteed in finite samples by the conformal prediction algorithm.

## Data Format Support

### Rust API

- **Parquet** files via Polars
- **CSV** files via Polars
- **In-memory** NumPy arrays (via Python bindings)

```rust
let loader = DatasetLoader::new(255);

// Load Parquet with specific target and feature columns
let dataset = loader.load_parquet(
    "data.parquet",
    "target_column",
    Some(&["feature1", "feature2", "feature3"])
)?;

// Load CSV
let dataset = loader.load_csv("data.csv", "target_column", None)?;
```

### Python API

- **NumPy arrays** (float32 or float64)
- **Pandas DataFrames** (via conversion to NumPy)

```python
import numpy as np
from treeboost import GBDTModel

X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randn(1000).astype(np.float32)

model = GBDTModel.train(X, y, config)
```

## Performance Characteristics

### Memory Usage

- Histogram-based training: O(num_features × num_bins) per tree
- u8 binning reduces feature size: 1 byte per bin instead of 4-8 bytes for raw floats
- 4-bit packing further reduces memory for features with <16 distinct values

### Time Complexity

- Feature-parallel histogram construction: O(num_rows × num_features) per round
- Split finding: O(num_features × num_bins) via histogram subtraction
- Total training: O(num_rounds × num_features × num_rows)

### Scalability

- Tested on datasets with millions of rows and hundreds of features
- Multi-threaded histogram construction scales near-linearly with core count
- Parallel prediction utilizes all cores for batch scoring

## Advanced Usage

### Feature Ordering Strategies

```python
config.column_reordering = True
config.reordering_strategy = "ByImportance"  # Default
```

### Monotonic Constraints

Enforce feature relationships (e.g., increasing price with square footage):

```python
from treeboost import MonotonicConstraint

config.monotonic_constraints = [
    MonotonicConstraint.Increasing,   # Feature 0: increasing
    MonotonicConstraint.None,         # Feature 1: no constraint
    MonotonicConstraint.Decreasing,   # Feature 2: decreasing
]
```

### Interaction Constraints

Restrict which features can interact:

```python
config.interaction_groups = [
    [0, 1, 2],      # Features 0-2 can interact with each other
    [3, 4],         # Features 3-4 can interact with each other
    # Features in different groups cannot interact
]
```

## Benchmarks

TreeBoost is optimized for both speed and memory efficiency on tabular data:

```bash
cd TreeBoost
cargo bench --bench competitors
```

Benchmarks compare against:

- **forust-ml**: Rust GBDT implementation
- **gbdt**: Lightweight Rust GBDT
- Standard implementations in other languages

## Project Structure

```
TreeBoost/
├── src/
│   ├── lib.rs                      # Library root and error types
│   ├── main.rs                     # CLI interface
│   ├── booster/                    # GBDT model and training
│   │   ├── config.rs               # Training configuration
│   │   ├── gbdt.rs                 # Model training and inference
│   │   └── mod.rs
│   ├── dataset/                    # Data loading and binning
│   │   ├── loader.rs               # Parquet/CSV loading via Polars
│   │   ├── binned.rs               # Columnar u8 storage
│   │   ├── binner.rs               # T-Digest quantile binning
│   │   ├── packed.rs               # 4-bit packing optimization
│   │   ├── reorder.rs              # Column reordering by importance
│   │   └── mod.rs
│   ├── tree/                       # Tree structures and growth
│   │   ├── tree.rs                 # Tree representation
│   │   ├── node.rs                 # Tree nodes
│   │   ├── split.rs                # Split finding with entropy regularization
│   │   ├── grow.rs                 # Best-First tree growth
│   │   └── mod.rs
│   ├── histogram/                  # Histogram construction
│   │   ├── builder.rs              # Feature-parallel builder
│   │   ├── entry.rs                # Histogram bin entries
│   │   └── mod.rs
│   ├── loss/                       # Loss functions
│   │   ├── traits.rs               # LossFunction trait
│   │   ├── mse.rs                  # Mean Squared Error
│   │   ├── huber.rs                # Pseudo-Huber Loss
│   │   └── mod.rs
│   ├── encoding/                   # Categorical encoding
│   │   ├── target.rs               # Ordered Target Encoding
│   │   ├── cms.rs                  # Count-Min Sketch
│   │   └── mod.rs
│   ├── inference/                  # Prediction
│   │   ├── predict.rs              # Prediction engine
│   │   ├── conformal.rs            # Conformal prediction intervals
│   │   └── mod.rs
│   ├── serialize/                  # Model persistence
│   │   ├── rkyv_io.rs              # Zero-copy serialization
│   │   └── mod.rs
│   └── python/                     # Python bindings (PyO3)
│       ├── bindings.rs             # Python API
│       └── mod.rs
├── Cargo.toml
├── pyproject.toml
├── samples/                        # Development and test data
├── benches/                        # Criterion benchmarks
├── tests/                          # Integration tests
└── README.md
```

## Dependencies

### Rust

- **polars**: Columnar data manipulation and I/O
- **rayon**: Data parallelism with work-stealing scheduler
- **rkyv**: Zero-copy serialization
- **tdigest**: Streaming quantile estimation
- **faer**: Pure-Rust linear algebra
- **bytemuck**: Safe transmutation for SIMD
- **rustc-hash**: Fast hashing for lookup tables
- **thiserror**: Ergonomic error handling
- **clap**: CLI argument parsing (for binary only)

### Python (Optional)

- **pyo3**: Python bindings
- **numpy**: Array interface

## Troubleshooting

### Python Import Errors

If you encounter import errors when using the Python package:

```bash
pip install -e .
```

Or use maturin to develop with your local Rust:

```bash
pip install maturin
maturin develop --release
```

### CUDA/GPU Support

TreeBoost currently targets CPU execution. GPU support is not available in the public release.

### Memory Issues on Large Datasets

If you encounter out-of-memory errors:

1. Increase `--num-bins` to coarser granularity (reduces feature cardinality)
2. Use `--colsample < 1.0` to subsample columns per tree
3. Use `--subsample < 1.0` to subsample rows per tree
4. Enable `--no-pack` if packing is causing issues (though this uses more memory)

### Slow Training

If training is slower than expected:

1. Check that parallel optimizations are enabled (default):
   ```bash
   treeboost info --model model.rkyv  # Shows enabled optimizations
   ```
2. Verify you're using `--release` builds in development
3. Use `cargo bench` to profile specific components

## Contributing

TreeBoost is part of the ml-rust organization. Contributions are welcome. Please ensure:

- Tests pass: `cargo test`
- Code is formatted: `cargo fmt`
- No clippy warnings: `cargo clippy --all-targets`
- Benchmarks don't regress: `cargo bench`

## License

TreeBoost is licensed under the Apache License 2.0. See `LICENSE` file for details.

## References

- Friedman, J. H. (2001). "Greedy function approximation: A gradient boosting machine."
- Ke, G., et al. (2017). "LightGBM: A fast, distributed, high-performance gradient boosting framework." NIPS.
- Barber, D., et al. (2022). "Conformal prediction under covariate shift." NeurIPS.
- Breiman, L., et al. (1984). "Classification and Regression Trees."

## Citation

If you use TreeBoost in your research, please cite:

```bibtex
@software{treeboost2024,
  title={TreeBoost: High-performance Gradient Boosted Decision Trees in Rust},
  author={Farhan},
  year={2024},
  url={https://github.com/ml-rust/treeboost}
}
```
