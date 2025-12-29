# TreeBoost API Reference

Complete API documentation for TreeBoost's Rust and Python interfaces.

## Table of Contents

1. [Core Training](#core-training) - GBDTConfig, GBDTModel
2. [AutoTuner](#autotuner) - Hyperparameter optimization
3. [Dataset](#dataset) - Data loading and preparation
4. [Loss Functions](#loss-functions) - Training objectives
5. [Inference](#inference) - Prediction and uncertainty
6. [Encoding](#encoding) - Categorical feature handling
7. [Serialization](#serialization) - Model persistence
8. [Backend Selection](#backend-selection) - Hardware acceleration

---

## Core Training

### GBDTConfig

Configuration object for gradient boosted decision tree training. Uses a builder pattern for chaining configurations.

**Rust Creation:**
```rust
use treeboost::GBDTConfig;

let config = GBDTConfig::new()
    .with_num_rounds(100)
    .with_max_depth(6)
    .with_learning_rate(0.1)
    .with_lambda(1.0)
    .with_entropy_weight(0.1)
    .with_subsample(0.8)
    .with_colsample(0.8)
    .with_seed(42);
```

**Python Creation:**
```python
from treeboost import GBDTConfig

config = GBDTConfig()
config.num_rounds = 100
config.max_depth = 6
config.learning_rate = 0.1
config.lambda_reg = 1.0
config.entropy_weight = 0.1
config.subsample = 0.8
config.colsample = 0.8
config.seed = 42
```

**Core Hyperparameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | int | 100 | Number of boosting iterations (trees) |
| `max_depth` | int | 6 | Maximum tree depth |
| `learning_rate` | float | 0.1 | Shrinkage per boosting round (0.0-1.0) |
| `max_leaves` | int | 31 | Maximum leaves per tree |
| `lambda` | float | 1.0 | L2 leaf regularization |
| `min_split_gain` | float | 0.0 | Minimum gain to split a node |
| `min_leaf_weight` | float | 0.0 | Minimum sum of weights in leaf |

**Advanced Hyperparameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entropy_weight` | float | 0.0 | Shannon entropy penalty (0.0-1.0) for drift prevention |
| `subsample` | float | 1.0 | Row sampling ratio per round (0.0-1.0) |
| `colsample` | float | 1.0 | Feature sampling ratio per tree (0.0-1.0) |
| `colsample_bylevel` | float | 1.0 | Feature sampling ratio per level (0.0-1.0) |
| `seed` | Option<u64> | None | Random seed for reproducibility |

**Loss Function Selection:**

```rust
// Regression losses (default: MSE)
config.with_mse_loss()                          // Mean Squared Error
config.with_pseudo_huber_loss(delta: 1.0)      // Robust to outliers

// Classification losses
config.with_binary_logloss()                    // Binary classification
config.with_multiclass_logloss(num_classes: 10)// Multi-class classification
```

**Conformal Prediction (Uncertainty):**

```rust
config
    .with_calibration_ratio(0.2)    // Reserve 20% of data for calibration
    .with_conformal_quantile(0.9)   // 90% prediction intervals (0.5-0.99)
```

**Constraints:**

```rust
use treeboost::tree::MonotonicConstraint;

// Monotonic constraints enforce monotonicity on features
config.set_monotonic_constraints(vec![
    MonotonicConstraint::Increasing,   // Feature 0 must increase
    MonotonicConstraint::None,         // Feature 1 unrestricted
    MonotonicConstraint::Decreasing,   // Feature 2 must decrease
]);

// Interaction groups restrict feature interactions
config.set_interaction_groups(vec![
    vec![0, 1, 2],  // Features 0,1,2 can interact with each other
    vec![3, 4],     // Features 3,4 in separate interaction group
    vec![5],        // Feature 5 cannot interact with others
]);
```

**Early Stopping:**

```rust
config.with_early_stopping(patience: 10, val_ratio: 0.2);
// Stops if validation loss doesn't improve for 10 rounds
// Uses 20% of data for validation
```

**Backend Selection:**

```rust
use treeboost::backend::BackendType;

config.with_backend(BackendType::Auto);    // (Default) Auto-select best backend
config.with_backend(BackendType::Cuda);    // NVIDIA CUDA GPU
config.with_backend(BackendType::Wgpu);    // WebGPU (all GPUs)
config.with_backend(BackendType::Avx512);  // AVX-512 CPU (x86-64 only)
config.with_backend(BackendType::Sve2);    // SVE2 CPU (ARM only)
config.with_backend(BackendType::Scalar);  // Scalar fallback (any CPU)
```

---

### GBDTModel

Trained gradient boosted decision tree model. All predictions use the CPU (no GPU overhead for inference).

**Training from BinnedDataset (Recommended):**

**Rust:**
```rust
use treeboost::{GBDTModel, GBDTConfig};
use treeboost::dataset::DatasetLoader;

let loader = DatasetLoader::new(255);  // 255 bins
let dataset = loader.load_parquet("train.parquet", "target", None)?;
let model = GBDTModel::train_binned(&dataset, config)?;
```

**Python:**
```python
from treeboost import GBDTModel, DatasetLoader

loader = DatasetLoader(num_bins=255)
dataset = loader.load_parquet("train.parquet", target="target")
model = GBDTModel.train_binned(dataset, config)
```

**Training from Raw Arrays:**

**Rust:**
```rust
let model = GBDTModel::train(
    &features,     // &[f32] in column-major layout
    num_features,  // usize
    &targets,      // &[f32]
    config,        // GBDTConfig
    None           // Optional feature names
)?;
```

**Python:**
```python
import numpy as np

X = np.random.randn(10000, 50).astype(np.float32)
y = np.random.randn(10000).astype(np.float32)

model = GBDTModel.train(X, y, config)
```

**Train and Save in One Step:**

**Rust:**
```rust
use treeboost::ModelFormat;

let model = GBDTModel::train_with_output(
    &dataset,
    config.clone(),
    "output_dir",
    &[ModelFormat::Rkyv, ModelFormat::Bincode]
)?;
// Saves to: output_dir/model.rkyv, output_dir/model.bin, output_dir/config.json
```

**Python:**
```python
model = GBDTModel.train(X, y, config, output_dir="models/my_model")
# Saves to: models/my_model/model.rkyv and models/my_model/config.json
```

**Making Predictions:**

**Rust:**
```rust
// Point predictions only
let predictions: Vec<f32> = model.predict(&dataset);

// With confidence intervals (if conformal enabled)
let (predictions, lower, upper) = model.predict_with_intervals(&dataset);

// Feature importance (gain-based)
let importances = model.feature_importance();

// Model metadata
let num_trees = model.num_trees();
```

**Python:**
```python
# Point predictions
predictions = model.predict(X)

# With intervals
predictions, lower, upper = model.predict_with_intervals(X)

# Feature importance
importances = model.feature_importance()

# Metadata
num_trees = model.num_trees()
```

**Serialization:**

**Rust:**
```rust
use treeboost::serialize::{save_model, load_model, save_model_bincode, load_model_bincode};
use treeboost::ModelFormat;

// rkyv format (zero-copy, fastest loading)
save_model(&model, "model.rkyv")?;
let model = load_model("model.rkyv")?;

// bincode format (compact, smaller files)
save_model_bincode(&model, "model.bin")?;
let model = load_model_bincode("model.bin")?;

// Save with config.json for reproducibility
model.save_to_directory(
    "output_dir",
    &config,
    &[ModelFormat::Rkyv, ModelFormat::Bincode]
)?;
// Creates: output_dir/model.rkyv, output_dir/model.bin, output_dir/config.json
```

**Python:**
```python
# Save single format
model.save("model.rkyv", format="rkyv")
model.save("model.bin", format="bincode")

# Load model
model = GBDTModel.load("model.rkyv", format="rkyv")

# Save with config
model.save_to_directory("output_dir", config, formats=["rkyv", "bincode"])
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `num_trees()` | usize | Number of trees in ensemble |
| `predict(dataset)` | Vec<f32> | Point predictions |
| `predict_with_intervals(dataset)` | (Vec<f32>, Vec<f32>, Vec<f32>) | Predictions with lower/upper bounds |
| `feature_importance()` | Vec<f32> | Feature importance scores (gain-based) |
| `save_to_directory(dir, config, formats)` | Result<()> | Save model + config.json |

---

## AutoTuner

Automatic hyperparameter optimization using efficient search strategies.

### TunerConfig

Configuration for the AutoTuner.

**Rust:**
```rust
use treeboost::tuner::{TunerConfig, GridStrategy, EvalStrategy};

let config = TunerConfig::new()
    .with_iterations(3)                                      // Number of refinement iterations
    .with_grid_strategy(GridStrategy::LatinHypercube { n_samples: 50 })
    .with_eval_strategy(EvalStrategy::holdout(0.2))          // 20% validation split
    .with_verbose(true)                                      // Print progress
    .with_output_dir("results/")                             // Save trial logs
    .with_save_model_formats(vec![ModelFormat::Rkyv]);       // Save best models
```

**Python:**
```python
from treeboost import TunerConfig, GridStrategy, EvalStrategy

config = (
    TunerConfig.quick()  # Preset: 2 iterations, simple space
    .with_grid_strategy(GridStrategy.lhs(50))
    .with_eval_strategy(EvalStrategy.holdout(0.2))
    .with_verbose(True)
)
# Other presets: TunerConfig.thorough(), TunerConfig.custom()
```

### GridStrategy

Hyperparameter space sampling strategy.

```rust
use treeboost::tuner::GridStrategy;

// Cartesian product grid (exhaustive)
GridStrategy::Cartesian { points_per_dim: 3 }
// Tries 3^n configurations for n parameters

// Latin Hypercube Sampling (efficient exploration)
GridStrategy::LatinHypercube { n_samples: 50 }
// Evenly samples 50 points in parameter space

// Random search
GridStrategy::Random { n_samples: 50 }
// Randomly samples 50 points
```

**Python:**
```python
from treeboost import GridStrategy

GridStrategy.cartesian(3)      # 3 points per dimension
GridStrategy.lhs(50)           # 50 LHS samples
GridStrategy.random(50)        # 50 random samples
```

### EvalStrategy

Evaluation strategy for measuring hyperparameter quality.

```rust
use treeboost::tuner::EvalStrategy;

// Holdout split
EvalStrategy::holdout(0.2)                // 20% validation set
    .with_folds(5)                        // Add 5-fold CV on validation set

// Conformal prediction
EvalStrategy::conformal(0.1, 0.9)         // 10% calibration, 90% coverage
```

**Python:**
```python
from treeboost import EvalStrategy

EvalStrategy.holdout(0.2)                 # 20% holdout
EvalStrategy.holdout(0.2).with_folds(5)   # 5-fold CV
EvalStrategy.conformal(0.1, 0.9)          # Conformal intervals
```

### ParameterSpace

Defines the hyperparameter search space.

**Rust:**
```rust
use treeboost::tuner::{ParameterSpace, ParamBounds};

let space = ParameterSpace::new()
    .with_param("max_depth", ParamBounds::discrete(3, 10), 6.0)
    .with_param("learning_rate", ParamBounds::log_continuous(0.005, 0.5), 0.05)
    .with_param("lambda", ParamBounds::continuous(0.0, 5.0), 1.0);

// Or use presets
let space = ParameterSpace::default_regression();    // Best for regression
let space = ParameterSpace::default_classification(); // Best for classification
let space = ParameterSpace::minimal();               // Quick tuning: depth + LR only
```

**Python:**
```python
from treeboost import ParameterSpace, ParamBounds

space = ParameterSpace()
space.add_discrete("max_depth", [3, 4, 5, 6, 7, 8, 9, 10])
space.add_continuous("learning_rate", 0.005, 0.5, scale="log")
space.add_continuous("lambda", 0.0, 5.0)

# Or presets
space = ParameterSpace.default_regression()
space = ParameterSpace.default_classification()
```

### AutoTuner Usage

**Rust:**
```rust
use treeboost::{AutoTuner, GBDTConfig};

let base_config = GBDTConfig::new().with_num_rounds(100).with_seed(42);

let mut tuner = AutoTuner::new(base_config)
    .with_config(tuner_config)
    .with_space(param_space)
    .with_seed(42)
    .with_callback(|trial, current, total| {
        println!("Trial {}/{}: val_loss={:.6}", current, total, trial.val_metric);
    });

let (best_config, history) = tuner.tune(&dataset)?;

// Access results
println!("Best validation loss: {:.6}", history.best().unwrap().val_metric);
println!("Total trials: {}", history.len());
println!("Best parameters: {:?}", history.best().unwrap().params);

// Train final model
let final_model = GBDTModel::train_binned(&dataset, best_config)?;
```

**Python:**
```python
from treeboost import AutoTuner

tuner = AutoTuner(base_config)
tuner.config = tuner_config
tuner.space = param_space

best_config, history = tuner.tune(X, y)

# Access results
print(f"Best validation loss: {history.best().val_metric:.6f}")
print(f"Total trials: {history.len()}")
for trial in history.top_n(5):
    print(f"  Trial {trial.trial_id}: loss={trial.val_metric:.6f}")

# Train final model
final_model = GBDTModel.train(X, y, best_config)
```

### SearchHistory

Results from hyperparameter search.

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `len()` | int | Total number of trials |
| `best()` | TrialResult | Best trial by validation metric |
| `top_n(n)` | Vec<TrialResult> | Top n trials |
| `trials()` | Vec<TrialResult> | All trials |

### TrialResult

Result of a single hyperparameter trial.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | int | Unique trial identifier |
| `iteration` | int | Which iteration (1-3 typically) |
| `val_metric` | float | Validation loss/metric |
| `train_metric` | float | Training loss |
| `f1_score` | Option<float> | F1 score (if classification) |
| `roc_auc` | Option<float> | ROC-AUC score (if classification) |
| `num_trees` | int | Number of trees in this trial |
| `params` | Dict[str, float] | Hyperparameters used |

---

## Dataset

Data loading and preparation utilities.

### DatasetLoader

Loads data from various formats and performs binning.

**Rust:**
```rust
use treeboost::dataset::DatasetLoader;

let loader = DatasetLoader::new(255);  // 255 bins

// From Parquet
let dataset = loader.load_parquet("train.parquet", "target", None)?;

// From CSV
let dataset = loader.load_csv(
    "train.csv",
    "target",
    Some(vec!["feature1", "feature2"])  // Optional feature selection
)?;

// From raw arrays
let dataset = loader.from_arrays(
    &features,  // &[f32]
    &targets,   // &[f32]
    Some(vec!["f0", "f1", "f2"])  // Feature names
)?;
```

**Python:**
```python
from treeboost import DatasetLoader

loader = DatasetLoader(num_bins=255)

dataset = loader.load_parquet("train.parquet", target="target")
dataset = loader.load_csv("train.csv", target="target")

import numpy as np
X = np.random.randn(10000, 50).astype(np.float32)
y = np.random.randn(10000).astype(np.float32)
dataset = loader.from_numpy(X, y, feature_names=[f"f{i}" for i in range(50)])
```

### BinnedDataset

Quantized dataset with feature binning.

**Properties:**

```rust
dataset.num_rows()                    // usize
dataset.num_features()                // usize
dataset.feature_info()                // Vec<FeatureInfo>
dataset.targets()                     // &[f32]
dataset.feature_names()               // Vec<&str>
```

---

## Loss Functions

Training objectives for regression and classification.

**Rust:**
```rust
use treeboost::loss::{MseLoss, PseudoHuberLoss, BinaryLogLoss};

let mse = MseLoss::new();
let loss_value = mse.loss(target, prediction);
let grad = mse.gradient(target, prediction);
let hess = mse.hessian(target, prediction);
let (grad, hess) = mse.gradient_hessian(target, prediction);

let huber = PseudoHuberLoss::new(delta: 1.0);
// Same methods as MSE

let logloss = BinaryLogLoss::new();
let prob = logloss.to_probability(raw_logit);
let class = logloss.to_class(prob, threshold: 0.5);
```

**Supported Loss Functions:**

- `MseLoss` - Mean Squared Error (regression)
- `PseudoHuberLoss` - Huber loss (robust regression, outlier-resistant)
- `BinaryLogLoss` - Logistic loss (binary classification)
- `MultiClassLogLoss` - Softmax loss (multi-class classification)

---

## Inference

Prediction and uncertainty quantification.

### Prediction with Intervals

**Rust:**
```rust
use treeboost::inference::Prediction;

// Without intervals
let pred = Prediction::point(0.5);
assert_eq!(pred.point, 0.5);

// With conformal intervals
let pred = Prediction::with_interval(0.5, 0.3, 0.7);
assert_eq!(pred.point, 0.5);
assert_eq!(pred.lower, 0.3);
assert_eq!(pred.upper, 0.7);
assert_eq!(pred.interval_width, 0.4);
```

### ConformalPredictor

Uncertainty quantification using split conformal prediction.

**Rust:**
```rust
use treeboost::inference::ConformalPredictor;

// From residuals (training - predictions on calibration set)
let predictor = ConformalPredictor::from_residuals(residuals, coverage: 0.9)?;

// From quantile directly
let predictor = ConformalPredictor::from_quantile(quantile: 0.15, coverage: 0.9)?;

// Single prediction
let pred = predictor.predict(point_pred);

// Batch predictions
let preds = predictor.predict_batch(&point_predictions);

// Measurement
let actual_coverage = predictor.empirical_coverage(&y_true, &predictions);
```

---

## Encoding

High-cardinality categorical feature handling.

### OrderedTargetEncoder

Stateful encoding of categorical features without target leakage.

**Rust:**
```rust
use treeboost::encoding::OrderedTargetEncoder;

let mut encoder = OrderedTargetEncoder::new(smoothing: 10.0);

// Training (streaming encoding)
for (category, target) in training_pairs {
    let encoded_val = encoder.encode_and_update(&category, target);
}

// Inference (fixed statistics)
let encoded = encoder.encode_inference("category_value");

// Get serializable mapping
let mapping = encoder.get_encoding_map();
let encoded_batch = mapping.encode_batch(&categories);
```

### CategoryFilter

Rare category filtering using Count-Min Sketch.

**Rust:**
```rust
use treeboost::encoding::CategoryFilter;

let filter = CategoryFilter::default_for_high_cardinality();
// Or: CategoryFilter::new(eps: 0.001, confidence: 0.99, min_count: 5)

// Two-pass workflow
filter.count_batch(&categories);      // First pass: count
filter.finalize();                     // Mark frequent ones
let is_frequent = filter.is_frequent("category");
let filtered = filter.filter_batch(&categories);  // "rare" → "unknown"
```

---

## Serialization

Model persistence and format handling.

### ModelFormat

Enumeration of supported serialization formats.

```rust
use treeboost::ModelFormat;

ModelFormat::Rkyv      // Zero-copy (fastest loading)
ModelFormat::Bincode   // Compact binary format
```

### Save/Load Functions

**Rust:**
```rust
use treeboost::serialize::{save_model, load_model, save_model_bincode, load_model_bincode};

// rkyv (recommended for production)
save_model(&model, "model.rkyv")?;
let model = load_model("model.rkyv")?;

// bincode (compact files)
save_model_bincode(&model, "model.bin")?;
let model = load_model_bincode("model.bin")?;
```

**Python:**
```python
from treeboost import GBDTModel

model.save("model.rkyv", format="rkyv")
model = GBDTModel.load("model.rkyv", format="rkyv")
```

---

## Backend Selection

Hardware acceleration and device configuration.

### BackendType

Enumeration of available compute backends.

```rust
use treeboost::backend::BackendType;

BackendType::Auto      // Auto-select best available (default)
BackendType::Cuda      // NVIDIA CUDA GPU
BackendType::Wgpu      // WebGPU (portable, all GPUs)
BackendType::Avx512    // AVX-512 CPU (x86-64 only)
BackendType::Sve2      // SVE2 CPU (ARM only)
BackendType::Scalar    // Scalar fallback (any CPU, safest)
```

### Backend Performance

| Backend | Hardware | Training Speed | Inference | Best For |
|---------|----------|---|---|---|
| CUDA | NVIDIA GPU | 10-50x faster | CPU | Large datasets on NVIDIA |
| WGPU | Any GPU | 5-20x faster | CPU | Portability, any GPU |
| AVX-512 | x86-64 CPU | 3-5x faster | CPU | CPU-only with modern CPUs |
| SVE2 | ARM CPU | 2-3x faster | CPU | ARM servers/clusters |
| Scalar | Any CPU | 1x (baseline) | CPU | Maximum compatibility |

**Inference Note:** All backends use optimized CPU inference (Rayon parallelism). GPU acceleration is for training only, so deployment doesn't require expensive GPU VMs.

---

## Complete Example

**Rust:**
```rust
use treeboost::{GBDTConfig, GBDTModel, AutoTuner, TunerConfig, GridStrategy, EvalStrategy, ParameterSpace, ModelFormat};
use treeboost::dataset::DatasetLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load and prepare data
    let loader = DatasetLoader::new(255);
    let dataset = loader.load_parquet("train.parquet", "target", None)?;

    // 2. Configure base model
    let base_config = GBDTConfig::new()
        .with_num_rounds(100)
        .with_seed(42);

    // 3. Tune hyperparameters
    let tuner_config = TunerConfig::new()
        .with_iterations(3)
        .with_grid_strategy(GridStrategy::LatinHypercube { n_samples: 50 })
        .with_eval_strategy(EvalStrategy::holdout(0.2).with_folds(5))
        .with_verbose(true);

    let mut tuner = AutoTuner::new(base_config)
        .with_config(tuner_config)
        .with_space(ParameterSpace::default_regression())
        .with_seed(42);

    let (best_config, history) = tuner.tune(&dataset)?;
    println!("Best validation loss: {:.6}", history.best().unwrap().val_metric);

    // 4. Train final model
    let model = GBDTModel::train_binned(&dataset, best_config.clone())?;

    // 5. Make predictions
    let predictions = model.predict(&dataset);

    // 6. Save model with config
    model.save_to_directory(
        "final_model",
        &best_config,
        &[ModelFormat::Rkyv, ModelFormat::Bincode]
    )?;

    println!("Model trained with {} trees", model.num_trees());
    Ok(())
}
```

**Python:**
```python
from treeboost import GBDTConfig, GBDTModel, AutoTuner, DatasetLoader
from treeboost import TunerConfig, GridStrategy, EvalStrategy, ParameterSpace

# 1. Load data
loader = DatasetLoader(num_bins=255)
dataset = loader.load_parquet("train.parquet", target="target")

# 2. Configure and tune
base_config = GBDTConfig().with_num_rounds(100).with_seed(42)

tuner = AutoTuner(base_config)
tuner.config = (
    TunerConfig.thorough()
    .with_grid_strategy(GridStrategy.lhs(50))
    .with_eval_strategy(EvalStrategy.holdout(0.2).with_folds(5))
)
tuner.space = ParameterSpace.default_regression()

best_config, history = tuner.tune(X, y)

# 3. Train final model
model = GBDTModel.train(X, y, best_config)

# 4. Predict
predictions = model.predict(X)

# 5. Save
model.save_to_directory("final_model", best_config, formats=["rkyv", "bincode"])
```
