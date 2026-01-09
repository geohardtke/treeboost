# TreeBoost API Reference

Complete API documentation for TreeBoost's Rust and Python interfaces.

## Table of Contents

1. [UniversalModel](#universalmodel) - **Start here.** The unified interface for all boosting modes
2. [Learners](#learners) - LinearBooster, LinearTreeBooster, TreeBooster
3. [Preprocessing](#preprocessing) - Scalers, encoders, imputers (serialize with model)
4. [AutoTuner](#autotuner) - Hyperparameter optimization
5. [Dataset](#dataset) - Data loading and preparation
6. [Loss Functions](#loss-functions) - Training objectives
7. [Inference](#inference) - Prediction and uncertainty
8. [Encoding](#encoding) - Categorical feature handling
9. [Serialization](#serialization) - Model persistence (rkyv, bincode)
10. [Incremental Learning](#incremental-learning) - **NEW:** TRB format, warm-start updates, drift detection
11. [Backend Selection](#backend-selection) - Hardware acceleration
12. [GBDTModel (Classic)](#gbdtmodel-classic) - The original GBDT-only API (still available)

---

## UniversalModel

**The main entry point.** Unified interface for all boosting modes.

`UniversalModel` wraps `GBDTModel` internally—you get GPU acceleration, conformal prediction, and all mature features automatically. The hybrid modes (LinearThenTree, RandomForest) add specialized functionality on top.

### BoostingMode

```rust
use treeboost::BoostingMode;

BoostingMode::PureTree        // Standard GBDT (wraps GBDTModel - GPU, conformal, multi-class)
BoostingMode::LinearThenTree  // Linear model + GBDTModel on residuals
BoostingMode::RandomForest    // Parallel independent trees with averaging
```

| Mode             | Best For                                  | How It Works                                          |
| ---------------- | ----------------------------------------- | ----------------------------------------------------- |
| `PureTree`       | General tabular, categoricals             | Delegates to `GBDTModel` (full GPU/conformal support) |
| `LinearThenTree` | Time-series, trending data, extrapolation | Linear phase → `GBDTModel` on residuals               |
| `RandomForest`   | Noisy data, variance reduction            | Bootstrap sampling, parallel trees, averaging         |

### UniversalConfig

```rust
use treeboost::{UniversalConfig, BoostingMode, StackingStrategy};
use treeboost::learner::{TreeConfig, LinearConfig};

let config = UniversalConfig::new()
    .with_mode(BoostingMode::LinearThenTree)
    .with_num_rounds(100)           // Number of tree rounds
    .with_linear_rounds(10)         // Linear boosting iterations (LinearThenTree only)
    .with_learning_rate(0.1)
    .with_subsample(0.8)
    .with_validation_ratio(0.2)
    .with_early_stopping_rounds(10)
    .with_seed(42);

// Fine-tune tree component
let config = config.with_tree_config(
    TreeConfig::default()
        .with_max_depth(6)
        .with_max_leaves(31)
        .with_lambda(1.0)
);

// Fine-tune linear component (LinearThenTree only)
let config = config.with_linear_config(
    LinearConfig::default()
        .with_lambda(1.0)           // Regularization strength
        .with_l1_ratio(0.0)         // 0.0 = Ridge, 1.0 = LASSO, between = ElasticNet
);

// Optional: Multi-seed ensemble training
let config = config
    .with_ensemble_seeds(vec![1, 2, 3, 4, 5])  // Train 5 models with different seeds
    .with_stacking_strategy(StackingStrategy::Ridge {
        alpha: 0.01,
        rank_transform: false,
        fit_intercept: true,
        min_weight: 0.01,
    });
```

**Python:**

```python
from treeboost import UniversalConfig, BoostingMode

config = UniversalConfig()
config.mode = BoostingMode.LinearThenTree
config.num_rounds = 100
config.linear_rounds = 10
config.learning_rate = 0.1
```

### Training

```rust
use treeboost::{UniversalModel, UniversalConfig, BoostingMode};
use treeboost::loss::MseLoss;
use treeboost::dataset::DatasetLoader;

let loader = DatasetLoader::new(255);
let dataset = loader.load_parquet("train.parquet", "target", None)?;

let config = UniversalConfig::new()
    .with_mode(BoostingMode::LinearThenTree)
    .with_num_rounds(100);

let model = UniversalModel::train(&dataset, config, &MseLoss)?;
```

### Automatic Mode Selection (NEW)

TreeBoost can **automatically analyze your dataset** and pick the best boosting mode. This "MRI scan" approach analyzes data characteristics WITHOUT expensive training trials.

```rust
use treeboost::{UniversalModel, MseLoss};

// One-liner: Let TreeBoost pick the best mode
let model = UniversalModel::auto(&dataset, &MseLoss)?;

// See why it picked this mode
println!("Selected: {:?}", model.mode());
println!("Confidence: {:?}", model.selection_confidence());
println!("{}", model.analysis_report().unwrap());
```

**How It Works:**

1. **Subsample** - Work on 10k-50k rows (fast)
2. **Linear Probe** - Quick Ridge regression measures linear signal (R²)
3. **Tree Probe** - Shallow tree on residuals measures non-linear structure
4. **Score Modes** - Compute scores for each mode based on data characteristics
5. **Recommend** - Pick highest-scoring mode with confidence level

**ModeSelection Enum:**

```rust
use treeboost::ModeSelection;

// Let TreeBoost analyze and pick (recommended for new datasets)
ModeSelection::Auto

// Custom analysis settings
ModeSelection::AutoWithConfig(AnalysisConfig::fast())

// Explicitly specify mode (when you know what works)
ModeSelection::Fixed(BoostingMode::LinearThenTree)
```

**All Auto Methods:**

```rust
// Simple one-liner
let model = UniversalModel::auto(&dataset, &loss)?;

// Auto with custom training config
let config = UniversalConfig::new().with_num_rounds(200);
let model = UniversalModel::auto_with_config(&dataset, config, &loss)?;

// Full control: custom analysis + training config
let analysis_config = AnalysisConfig::thorough();
let model = UniversalModel::auto_with_analysis_config(
    &dataset, config, analysis_config, &loss
)?;

// Via train_with_selection (most explicit)
let model = UniversalModel::train_with_selection(
    &dataset, config, ModeSelection::Auto, &loss
)?;
```

**Analysis Access:**

```rust
// Check if auto-selected
model.was_auto_selected()  // bool

// Get analysis results
if let Some(analysis) = model.analysis() {
    println!("Linear R²: {:.2}", analysis.linear_r2);
    println!("Tree gain: {:.2}", analysis.tree_gain);
    println!("Noise floor: {:.2}", analysis.noise_floor);
    println!("Categorical ratio: {:.2}", analysis.categorical_ratio);
}

// Confidence level: High, Medium, or Low
model.selection_confidence()  // Option<Confidence>

// Pretty report (displays all metrics, scores, reasoning)
println!("{}", model.analysis_report().unwrap());

// Compact summary for logging
println!("{}", model.analysis_summary().unwrap());
```

**Decision Logic:**

| Data Pattern                             | Recommended Mode | Why                                            |
| ---------------------------------------- | ---------------- | ---------------------------------------------- |
| High linear R² (>0.3) + tree gain (>0.1) | LinearThenTree   | Linear captures trend, trees capture residuals |
| Weak linear signal + categorical-heavy   | PureTree         | Trees handle categoricals natively             |
| High noise floor (>0.4)                  | RandomForest     | Bagging reduces variance                       |

**When to Use:**

- ✅ **New datasets** - Let TreeBoost explore the data
- ✅ **Unsure which mode** - Analysis provides explanation
- ✅ **Documentation** - Report explains the decision
- ❌ **Benchmarking** - Use fixed mode for reproducibility
- ❌ **Known best mode** - Skip analysis overhead

### Serialization

UniversalModel supports zero-copy serialization via rkyv, making it perfect for production deployment.

```rust
// Save model for inference
model.save("model.rkyv")?;

// Load model for inference
let loaded = UniversalModel::load("model.rkyv")?;

// Get config for inspection/reuse
let config = model.config();  // &UniversalConfig
let config_json = serde_json::to_string_pretty(config)?;
std::fs::write("config.json", config_json)?;
```

**Typical workflow:**

```rust
// 1. Train with AutoML
let auto = AutoModel::train(&df, "target")?;

// 2. Export discovered configuration to JSON (useful for inspection and reuse)
auto.save_config("best_config.json")?;

// 3. Save trained model for inference
auto.save("model.rkyv")?;

// 4. Later: Load model for predictions (no need to retrain)
let loaded = UniversalModel::load("model.rkyv")?;
let predictions = loaded.predict(&dataset);
```

### Prediction

```rust
// Batch prediction
let predictions = model.predict(&dataset);

// Single row
let pred = model.predict_row(&dataset, row_idx);

// Model info
model.mode()           // BoostingMode
model.num_trees()      // usize
model.has_linear()     // bool (true if LinearThenTree)
model.num_features()   // usize
```

### Conformal Prediction (PureTree only)

```rust
// Predictions with uncertainty intervals
let (predictions, lower, upper) = model.predict_with_intervals(&dataset)?;

// Check calibration status
let quantile = model.conformal_quantile();  // Option<f32>
```

### Classification (PureTree only)

```rust
// Binary classification
let probabilities = model.predict_proba(&dataset)?;        // Vec<f32> in [0, 1]
let classes = model.predict_class(&dataset, 0.5)?;         // Vec<u32> (0 or 1)

// Multi-class
let is_mc = model.is_multiclass();                         // bool
let num_classes = model.get_num_classes();                 // usize
let probs = model.predict_proba_multiclass(&dataset)?;     // Vec<Vec<f32>>
let classes = model.predict_class_multiclass(&dataset)?;   // Vec<u32>
let logits = model.predict_raw_multiclass(&dataset)?;      // Vec<Vec<f32>>
```

### Feature Importance

```rust
// Works for all modes
let importances = model.feature_importance();  // Vec<f32>, normalized to sum to 1.0
```

### Raw Prediction (PureTree only)

Predict directly from unbinned f64 features without creating a `BinnedDataset`:

```rust
let features: &[f64] = &[1.0, 2.0, 3.0];  // row-major
let predictions = model.predict_raw(features)?;

// With intervals
let (preds, lower, upper) = model.predict_raw_with_intervals(features)?;

// Classification from raw features
let probs = model.predict_proba_raw(features)?;
let classes = model.predict_class_raw(features, 0.5)?;
```

### Method Summary

**Training Methods:**

| Method                                                     | Returns      | Description                    |
| ---------------------------------------------------------- | ------------ | ------------------------------ |
| `train(&dataset, config, &loss)`                           | Result<Self> | Train with explicit config     |
| `auto(&dataset, &loss)`                                    | Result<Self> | Auto-select mode (NEW)         |
| `auto_with_config(&dataset, config, &loss)`                | Result<Self> | Auto-select with custom config |
| `train_with_selection(&dataset, config, selection, &loss)` | Result<Self> | Full control via ModeSelection |

**Prediction Methods:**

| Method                               | Returns                 | Modes    | Description                  |
| ------------------------------------ | ----------------------- | -------- | ---------------------------- |
| `predict(&dataset)`                  | Vec<f32>                | All      | Point predictions            |
| `predict_row(&dataset, idx)`         | f32                     | All      | Single row prediction        |
| `predict_with_intervals(&dataset)`   | Result<(Vec, Vec, Vec)> | PureTree | Conformal intervals          |
| `predict_proba(&dataset)`            | Result<Vec<f32>>        | PureTree | Binary probabilities         |
| `predict_class(&dataset, threshold)` | Result<Vec<u32>>        | PureTree | Binary classes               |
| `predict_proba_multiclass(&dataset)` | Result<Vec<Vec<f32>>>   | PureTree | Multi-class probabilities    |
| `predict_class_multiclass(&dataset)` | Result<Vec<u32>>        | PureTree | Multi-class predictions      |
| `feature_importance()`               | Vec<f32>                | All      | Normalized importance scores |
| `predict_raw(features)`              | Result<Vec<f32>>        | PureTree | From unbinned features       |
| `conformal_quantile()`               | Option<f32>             | PureTree | Calibrated quantile          |
| `is_multiclass()`                    | bool                    | All      | Check if multi-class model   |
| `get_num_classes()`                  | usize                   | All      | Number of classes            |

**Analysis Methods (for auto-selected models):**

| Method                   | Returns                  | Description                  |
| ------------------------ | ------------------------ | ---------------------------- |
| `was_auto_selected()`    | bool                     | True if auto mode was used   |
| `analysis()`             | Option<&DatasetAnalysis> | Full analysis results        |
| `selection_confidence()` | Option<Confidence>       | High/Medium/Low confidence   |
| `analysis_report()`      | Option<AnalysisReport>   | Formatted diagnostic report  |
| `analysis_summary()`     | Option<String>           | One-line summary for logging |

---

## Learners

Low-level weak learners. Use these directly when building custom boosting loops.

### LinearBooster

Ridge, LASSO, or Elastic Net via Coordinate Descent. Used internally by `LinearThenTree` mode.

```rust
use treeboost::learner::{LinearBooster, LinearConfig, LinearPreset, WeakLearner};

// Ridge (L2) - default, most stable
let config = LinearConfig::default()
    .with_preset(LinearPreset::Ridge)
    .with_lambda(1.0);

// LASSO (L1) - sparse solutions, feature selection
let config = LinearConfig::default()
    .with_preset(LinearPreset::Lasso)
    .with_lambda(1.0);

// Elastic Net - mix of L1 + L2
let config = LinearConfig::default()
    .with_preset(LinearPreset::ElasticNet)
    .with_lambda(1.0)
    .with_l1_ratio(0.5);  // 50% L1, 50% L2

let mut booster = LinearBooster::new(num_features, config);

// Fit on gradients (gradient boosting style)
booster.fit_on_gradients(&features, num_features, &gradients, &hessians)?;

// Predict
let predictions = booster.predict_batch(&features, num_features);

// Sparsity info (LASSO/ElasticNet)
booster.num_nonzero_weights()  // How many features selected
booster.selected_features()    // Which feature indices
```

**Critical:** `lambda >= 1e-6` is enforced. Setting `lambda=0` causes NaN on correlated features.

**Internal standardization:** The booster standardizes features internally. You don't need to pre-scale.

### LinearTreeBooster

Decision trees with Ridge regression in each leaf. Perfect for piecewise linear data.

```rust
use treeboost::learner::{LinearTreeBooster, LinearTreeConfig, TreeConfig, LinearConfig, LinearPreset};

let config = LinearTreeConfig::new()
    .with_tree_config(
        TreeConfig::default()
            .with_max_depth(3)           // Shallow - leaves do the work
            .with_min_samples_leaf(50)
    )
    .with_linear_config(
        LinearConfig::default()
            .with_preset(LinearPreset::Ridge)
            .with_lambda(0.1)
    )
    .with_min_samples_for_linear(20);    // Below this, use constant leaf

let mut booster = LinearTreeBooster::new(config);

// Needs both binned dataset AND raw features
booster.fit_on_gradients(&dataset, &raw_features, num_features, &gradients, &hessians)?;

let predictions = booster.predict_batch(&dataset, &raw_features, num_features);
```

**When to use:**

- Data has piecewise linear structure (tax brackets, physical systems)
- Want smoother predictions than standard trees
- Need 10-100x fewer trees for same accuracy

### TreeBooster

Standard histogram-based decision tree. The workhorse of GBDT.

```rust
use treeboost::learner::{TreeBooster, TreeConfig, WeakLearner};

let config = TreeConfig::default()
    .with_max_depth(6)
    .with_max_leaves(31)
    .with_lambda(1.0)
    .with_learning_rate(0.1);

let mut booster = TreeBooster::new(config);
booster.fit_on_gradients(&dataset, &gradients, &hessians, None)?;

if let Some(tree) = booster.tree() {
    tree.predict_batch_add(&dataset, &mut predictions);
}
```

---

## Preprocessing

Transforms that fit on training data and apply consistently to train/test. Serialize with your model.

### Scalers

```rust
use treeboost::preprocessing::{StandardScaler, MinMaxScaler, RobustScaler, Scaler};

// StandardScaler: zero mean, unit variance
let mut scaler = StandardScaler::new();
scaler.fit(&features, num_features)?;
scaler.transform(&mut features, num_features)?;

// MinMaxScaler: scale to [0, 1] or custom range
let mut scaler = MinMaxScaler::new(0.0, 1.0);

// RobustScaler: median/IQR (robust to outliers)
let mut scaler = RobustScaler::new();
```

**For linear models:** Scaling is essential. Linear models are sensitive to feature magnitudes.

**For trees:** Scaling helps with regularization fairness but isn't strictly required.

### Encoders

```rust
use treeboost::preprocessing::{FrequencyEncoder, LabelEncoder, OneHotEncoder, UnknownStrategy};

// FrequencyEncoder: category → count (best for trees)
let mut encoder = FrequencyEncoder::new();
encoder.fit(&categories)?;
let encoded = encoder.transform(&categories)?;

// LabelEncoder: string → integer
let mut encoder = LabelEncoder::new(UnknownStrategy::Error);  // or ::MapToZero
encoder.fit(&categories)?;

// OneHotEncoder: category → binary columns (for linear models)
// WARNING: High cardinality = memory explosion. Use for linear component only.
let mut encoder = OneHotEncoder::new(UnknownStrategy::Ignore);
encoder.fit(&categories)?;
let (encoded, num_new_cols) = encoder.transform(&data, num_features, cat_col_idx)?;
```

| Encoder                | Best For                              | Cardinality           |
| ---------------------- | ------------------------------------- | --------------------- |
| `FrequencyEncoder`     | Trees (GBDT)                          | Any                   |
| `LabelEncoder`         | Trees (GBDT)                          | Any                   |
| `OneHotEncoder`        | Linear models only                    | Low (<100 categories) |
| `OrderedTargetEncoder` | High-cardinality + leakage prevention | Any                   |

### Imputers

```rust
use treeboost::preprocessing::{SimpleImputer, ImputeStrategy, IndicatorImputer};

// SimpleImputer: fill missing values
let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);  // or Median, Mode, Constant(0.0)
imputer.fit(&features, num_features)?;
imputer.transform(&mut features, num_features)?;

// IndicatorImputer: add binary "was_missing" columns
let mut indicator = IndicatorImputer::new();
indicator.fit(&features, num_features)?;
let (features_with_indicators, new_num_features) = indicator.transform(&features, num_features)?;
```

### PipelineBuilder

Chain transforms together:

```rust
use treeboost::preprocessing::{PipelineBuilder, ImputeStrategy};

let pipeline = PipelineBuilder::new()
    .add_simple_imputer(&["age", "income"], ImputeStrategy::Median)
    .add_standard_scaler(&["age", "income"])
    .add_frequency_encoder(&["category", "region"])
    .build();

// Fit on training data
pipeline.fit(&train_df)?;

// Transform (same pipeline state for train and test)
let train_transformed = pipeline.transform(&train_df)?;
let test_transformed = pipeline.transform(&test_df)?;
```

### Time-Series Features

```rust
use treeboost::preprocessing::{LagGenerator, RollingGenerator, RollingStat, EwmaGenerator};

// Lag features: x_{t-1}, x_{t-2}, ...
let lag_gen = LagGenerator::new(vec![1, 2, 7]);  // 1-day, 2-day, 7-day lags

// Rolling statistics
let rolling = RollingGenerator::new(7, vec![RollingStat::Mean, RollingStat::Std]);

// Exponentially weighted moving average
let ewma = EwmaGenerator::new(0.3);  // alpha = 0.3
```

## GBDTModel (Classic)

The original GBDT-only API. Still fully supported—use it if you don't need hybrid modes.

> **Tip:** `GBDTModel` is equivalent to `UniversalModel` with `BoostingMode::PureTree`.

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

| Parameter         | Type  | Default | Description                            |
| ----------------- | ----- | ------- | -------------------------------------- |
| `num_rounds`      | int   | 100     | Number of boosting iterations (trees)  |
| `max_depth`       | int   | 6       | Maximum tree depth                     |
| `learning_rate`   | float | 0.1     | Shrinkage per boosting round (0.0-1.0) |
| `max_leaves`      | int   | 31      | Maximum leaves per tree                |
| `lambda`          | float | 1.0     | L2 leaf regularization                 |
| `min_split_gain`  | float | 0.0     | Minimum gain to split a node           |
| `min_leaf_weight` | float | 0.0     | Minimum sum of weights in leaf         |

**Advanced Hyperparameters:**

| Parameter           | Type        | Default | Description                                            |
| ------------------- | ----------- | ------- | ------------------------------------------------------ |
| `entropy_weight`    | float       | 0.0     | Shannon entropy penalty (0.0-1.0) for drift prevention |
| `subsample`         | float       | 1.0     | Row sampling ratio per round (0.0-1.0)                 |
| `colsample`         | float       | 1.0     | Feature sampling ratio per tree (0.0-1.0)              |
| `colsample_bylevel` | float       | 1.0     | Feature sampling ratio per level (0.0-1.0)             |
| `seed`              | Option<u64> | None    | Random seed for reproducibility                        |

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

| Method                                    | Returns                        | Description                            |
| ----------------------------------------- | ------------------------------ | -------------------------------------- |
| `num_trees()`                             | usize                          | Number of trees in ensemble            |
| `predict(dataset)`                        | Vec<f32>                       | Point predictions                      |
| `predict_with_intervals(dataset)`         | (Vec<f32>, Vec<f32>, Vec<f32>) | Predictions with lower/upper bounds    |
| `feature_importance()`                    | Vec<f32>                       | Feature importance scores (gain-based) |
| `save_to_directory(dir, config, formats)` | Result<()>                     | Save model + config.json               |

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
    TunerConfig.preset("quick")  # Preset: 2 iterations, simple space
    .with_grid_strategy(GridStrategy.lhs(50))
    .with_eval_strategy(EvalStrategy.holdout(0.2))
    .with_verbose(True)
)
# Other presets: "smoketest", "balanced", "thorough"
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
use treeboost::tuner::{ParameterSpace, ParamBounds, SpacePreset};

let space = ParameterSpace::new()
    .with_param("max_depth", ParamBounds::discrete(3, 10), 6.0)
    .with_param("learning_rate", ParamBounds::log_continuous(0.005, 0.5), 0.05)
    .with_param("lambda", ParamBounds::continuous(0.0, 5.0), 1.0);

// Or use presets
let space = ParameterSpace::with_preset(SpacePreset::Regression);     // Best for regression
let space = ParameterSpace::with_preset(SpacePreset::Classification); // Best for classification
let space = ParameterSpace::with_preset(SpacePreset::Minimal);        // Quick tuning: depth + LR only
```

**Python:**

```python
from treeboost import ParameterSpace, ParamBounds

space = ParameterSpace()
space.add_discrete("max_depth", [3, 4, 5, 6, 7, 8, 9, 10])
space.add_continuous("learning_rate", 0.005, 0.5, scale="log")
space.add_continuous("lambda", 0.0, 5.0)

# Or presets
space = ParameterSpace.preset("regression")
space = ParameterSpace.preset("classification")
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

| Method     | Returns          | Description                     |
| ---------- | ---------------- | ------------------------------- |
| `len()`    | int              | Total number of trials          |
| `best()`   | TrialResult      | Best trial by validation metric |
| `top_n(n)` | Vec<TrialResult> | Top n trials                    |
| `trials()` | Vec<TrialResult> | All trials                      |

### TrialResult

Result of a single hyperparameter trial.

**Fields:**

| Field          | Type             | Description                       |
| -------------- | ---------------- | --------------------------------- |
| `trial_id`     | int              | Unique trial identifier           |
| `iteration`    | int              | Which iteration (1-3 typically)   |
| `val_metric`   | float            | Validation loss/metric            |
| `train_metric` | float            | Training loss                     |
| `f1_score`     | Option<float>    | F1 score (if classification)      |
| `roc_auc`      | Option<float>    | ROC-AUC score (if classification) |
| `num_trees`    | int              | Number of trees in this trial     |
| `params`       | Dict[str, float] | Hyperparameters used              |

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

## Incremental Learning

Update models with new data without full retraining. TreeBoost provides a custom TRB (TreeBoost) file format optimized for incremental updates.

### Why Incremental Learning?

| Scenario | Without Incremental | With Incremental |
|----------|---------------------|------------------|
| Daily model updates | Retrain on full history | Add trees from new data only |
| Storage for updates | New file per version | Single file, append updates |
| Compute cost | O(total_data) | O(new_data) |
| Recovery from crash | Restart training | Resume from last checkpoint |

### TRB Workflow Overview

TRB format stores `UniversalModel` only. The typical workflow:

1. **Initial training** — Use `AutoModel` for convenience (handles DataFrames)
2. **Save to TRB** — Extract `UniversalModel` and save
3. **Incremental updates** — Load `UniversalModel`, update with `BinnedDataset`
4. **Inference** — Load `UniversalModel`, predict with `BinnedDataset`

```rust
use treeboost::{AutoModel, UniversalModel};
use treeboost::dataset::DatasetLoader;
use treeboost::loss::MseLoss;

// 1. Initial training via AutoModel (convenience)
let auto = AutoModel::train(&df, "target")?;

// 2. Save UniversalModel to TRB
auto.inner().save_trb("model.trb", "Initial training")?;

// 3. Later: Load and update (UniversalModel + BinnedDataset)
let mut model = UniversalModel::load_trb("model.trb")?;
let loader = DatasetLoader::new(255);
let new_data = loader.load_parquet("new_data.parquet", "target", None)?;

let report = model.update(&new_data, &MseLoss, 10)?;
println!("Trees: {} -> {} (+{})", report.trees_before, report.trees_after, report.trees_added);

// 4. Append update to file (O(1), no rewrite)
model.save_trb_update("model.trb", new_data.num_rows(), "Update batch")?;

// 5. Inference
let model = UniversalModel::load_trb("model.trb")?;
let predictions = model.predict(&new_data);
```

### IncrementalUpdateReport

Returned by `UniversalModel::update()`:

```rust
pub struct IncrementalUpdateReport {
    pub trees_before: usize,      // Tree count before update
    pub trees_after: usize,       // Tree count after update
    pub trees_added: usize,       // trees_after - trees_before
    pub mode: BoostingMode,       // PureTree, LinearThenTree, or RandomForest
}
```

### TRB File Format

Custom journaled format for incremental model persistence:

```
┌─────────────────────────────────────────────────────────────┐
│ TrbHeader (JSON + length prefix)                            │
│   - magic: "TRB1"                                           │
│   - format_version: 1                                       │
│   - model_type: "UniversalModel"                           │
│   - boosting_mode: "PureTree" | "LinearThenTree" | ...     │
│   - num_features: 50                                        │
│   - created_at: 1704067200 (Unix timestamp)                │
│   - metadata: "Initial training description"                │
│   - base_blob_size: 1234567                                │
├─────────────────────────────────────────────────────────────┤
│ Base Model Blob (rkyv serialized UniversalModel)            │
│   + CRC32 checksum (4 bytes)                               │
├─────────────────────────────────────────────────────────────┤
│ Update 1: TrbUpdateHeader (JSON) + Blob + CRC32            │
│   - update_type: Trees | Linear | Full                     │
│   - created_at: timestamp                                   │
│   - rows_trained: 5000                                     │
│   - description: "February update"                         │
├─────────────────────────────────────────────────────────────┤
│ Update 2: TrbUpdateHeader + Blob + CRC32                   │
│ ...                                                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Properties:**

- **O(1) appends** — Updates append to file end, base model never rewritten
- **CRC32 per segment** — Corruption detected at segment level
- **Crash recovery** — Truncated writes detected and skipped
- **Forward compatible** — Unknown JSON fields ignored (safe for version upgrades)
- **File locking** — Exclusive locks prevent concurrent writes

### TrbReader / TrbWriter

Low-level API for TRB file manipulation:

```rust
use treeboost::serialize::{TrbWriter, TrbReader, TrbHeader, TrbUpdateHeader, UpdateType};

// Writing
let header = TrbHeader {
    format_version: 1,
    model_type: "UniversalModel".to_string(),
    boosting_mode: "PureTree".to_string(),
    num_features: 50,
    created_at: current_timestamp(),
    metadata: "My model".to_string(),
    base_blob_size: base_blob.len() as u64,
};

let writer = TrbWriter::new("model.trb", header, &base_blob)?;
drop(writer);  // Finishes write

// Appending updates
let mut writer = open_for_append("model.trb")?;
let update_header = TrbUpdateHeader {
    update_type: UpdateType::Trees,
    created_at: current_timestamp(),
    rows_trained: 5000,
    description: "February update".to_string(),
};
writer.append_update(&update_header, &update_blob)?;

// Reading
let mut reader = TrbReader::open("model.trb")?;
let header = reader.header();  // &TrbHeader
let base_blob = reader.read_base_blob()?;

// Iterate updates (handles crash recovery automatically)
for (update_header, blob) in reader.iter_updates()? {
    println!("Update: {} rows", update_header.rows_trained);
}

// Load all segments at once
let segments = reader.load_all_segments()?;
```

### MmapTrbReader (mmap feature)

Memory-mapped TRB reader for true zero-copy I/O. Requires the `mmap` feature:

```bash
cargo build --release --features mmap
```

**When to use:**
- Large models (100MB+) where heap allocation is expensive
- Inference servers requiring minimal startup time
- Running multiple model instances (OS deduplicates pages)
- Memory-constrained environments

**Comparison:**

| Reader | Load Time | Memory | Use Case |
|--------|-----------|--------|----------|
| `TrbReader` | O(model_size) | O(model_size) | Default, works everywhere |
| `MmapTrbReader` | O(1) initial | O(1) initial | Large models, servers |

**Usage:**

```rust
#[cfg(feature = "mmap")]
{
    use treeboost::serialize::MmapTrbReader;

    // Open with memory mapping - instant, no heap allocation
    let reader = MmapTrbReader::open("model.trb")?;

    // Option 1: Deserialize (still faster than TrbReader due to mmap)
    let model = reader.load_model()?;
    let predictions = model.predict(&dataset);

    // Option 2: Zero-copy blob access
    let base_blob = reader.base_blob_bytes()?;  // &[u8] into mmap

    // Iterate updates (zero-copy slices)
    for (header, blob) in reader.iter_updates()? {
        println!("Update: {} rows", header.rows_trained);
    }
}
```

**API:**

| Method | Description |
|--------|-------------|
| `open(path)` | Open TRB file with memory mapping |
| `header()` | Get the `TrbHeader` |
| `load_model()` | Deserialize `UniversalModel` |
| `base_blob_bytes()` | Zero-copy access to base blob |
| `iter_updates()` | Iterate update segments (zero-copy) |
| `mapped_size()` | Size of memory-mapped region |

### UniversalModel Incremental API

Lower-level API for custom update logic:

```rust
use treeboost::{UniversalModel, UniversalConfig, BoostingMode};
use treeboost::loss::MseLoss;

// Train initial model
let config = UniversalConfig::new()
    .with_mode(BoostingMode::PureTree)
    .with_num_rounds(100);
let mut model = UniversalModel::train(&dataset, config, &MseLoss)?;

// Update with new data
let report = model.update(&new_dataset, &MseLoss, 10)?;  // Add 10 trees
println!("Added {} trees", report.trees_added);

// Save/load TRB
model.save_trb("model.trb", "description")?;
let loaded = UniversalModel::load_trb("model.trb")?;
```

### IncrementalUpdateReport

Returned by `UniversalModel::update()`:

```rust
pub struct IncrementalUpdateReport {
    pub trees_before: usize,
    pub trees_after: usize,
    pub trees_added: usize,
    pub mode: BoostingMode,
}
```

### Drift Detection

Monitor distribution shifts before updating:

```rust
use treeboost::monitoring::{
    IncrementalDriftDetector,
    DriftRecommendation,
    check_drift
};

// Create detector from training data
let detector = IncrementalDriftDetector::from_dataset(&train_data)
    .with_thresholds(0.1, 0.25);  // warning, critical

// Check new data before updating
let result = detector.check_update(&new_data);

println!("Mean drift: {:.4}", result.mean_drift);
println!("Max drift feature: {:?}", result.max_drift_feature);
println!("Alert level: {:?}", result.shift_result.alert);

match result.recommendation {
    DriftRecommendation::ProceedNormally => {
        // Safe to update incrementally
        model.update(&new_data, &loss, 10)?;
    }
    DriftRecommendation::ProceedWithCaution => {
        // Update but monitor performance closely
        model.update(&new_data, &loss, 10)?;
    }
    DriftRecommendation::ConsiderRetrain => {
        // Significant drift - consider full retrain
        println!("Warning: {}", result);
    }
    DriftRecommendation::RetrainRecommended => {
        // Critical drift - full retrain recommended
        println!("Critical: {}", result);
    }
}
```

### DriftHistory

Track drift across multiple updates:

```rust
use treeboost::monitoring::DriftHistory;

let mut history = DriftHistory::new();

// Record each update's drift check
for batch in batches {
    let result = detector.check_update(&batch);
    history.record(&result);

    if !result.has_critical_drift() {
        model.update(&batch, &loss, 10)?;
    }
}

println!("{}", history);
// Output:
// Drift History (10 updates):
//   Drift rate: 20.0% (1 warnings, 1 critical)
//   Mean drift: 0.0823, Max drift: 0.3412
//   Frequently drifted:
//     - feature_5 (3 times)
//     - feature_12 (2 times)
```

### EMA-Based Scaler Updates

StandardScaler supports exponential moving average for adapting to drift:

```rust
use treeboost::preprocessing::StandardScaler;

// Create scaler with EMA mode (alpha=0.1)
let mut scaler = StandardScaler::with_forget_factor(0.1);

// Each partial_fit blends new statistics with history
// new_stat = (1 - alpha) * old_stat + alpha * batch_stat
scaler.partial_fit(&batch1, num_features)?;  // 100% batch1
scaler.partial_fit(&batch2, num_features)?;  // 90% batch1, 10% batch2
scaler.partial_fit(&batch3, num_features)?;  // 81% batch1, 9% batch2, 10% batch3

// After 10 batches, batch1 influence: (1-0.1)^10 ≈ 35%
```

### CLI for Incremental Learning

```bash
# Update existing TRB model with new data
treeboost update --model model.trb --data new.csv --target price \
  --rounds 10 --description "March 2024 update"

# Inspect TRB file (shows update history)
treeboost info --model model.trb

# Force load despite corrupted updates
treeboost info --model model.trb --force
```

### Method Summary

**UniversalModel (TRB format):**

| Method | Returns | Description |
|--------|---------|-------------|
| `load_trb(path)` | Result<UniversalModel> | Load model from TRB file |
| `save_trb(path, desc)` | Result<()> | Save model to TRB format |
| `save_trb_update(path, rows, desc)` | Result<()> | Append update to existing TRB |
| `update(&dataset, &loss, rounds)` | Result<IncrementalUpdateReport> | Add trees from BinnedDataset |
| `predict(&dataset)` | Vec<f32> | Inference on BinnedDataset |

**AutoModel (initial training only):**

| Method | Returns | Description |
|--------|---------|-------------|
| `train(&df, target)` | Result<AutoModel> | Train from DataFrame |
| `inner()` | &UniversalModel | Get underlying model for TRB save |

**IncrementalDriftDetector:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_dataset(&data)` | Self | Create from reference distribution |
| `with_thresholds(warn, crit)` | Self | Set PSI thresholds |
| `check_update(&data)` | IncrementalDriftResult | Check for drift |

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

| Backend | Hardware   | Training Speed | Inference | Best For                  |
| ------- | ---------- | -------------- | --------- | ------------------------- |
| CUDA    | NVIDIA GPU | 10-50x faster  | CPU       | Large datasets on NVIDIA  |
| WGPU    | Any GPU    | 5-20x faster   | CPU       | Portability, any GPU      |
| AVX-512 | x86-64 CPU | 3-5x faster    | CPU       | CPU-only with modern CPUs |
| SVE2    | ARM CPU    | 2-3x faster    | CPU       | ARM servers/clusters      |
| Scalar  | Any CPU    | 1x (baseline)  | CPU       | Maximum compatibility     |

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
        .with_space(ParameterSpace::with_preset(SpacePreset::Regression))
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
    TunerConfig.preset("thorough")
    .with_grid_strategy(GridStrategy.lhs(50))
    .with_eval_strategy(EvalStrategy.holdout(0.2).with_folds(5))
)
tuner.space = ParameterSpace.preset("regression")

best_config, history = tuner.tune(X, y)

# 3. Train final model
model = GBDTModel.train(X, y, best_config)

# 4. Predict
predictions = model.predict(X)

# 5. Save
model.save_to_directory("final_model", best_config, formats=["rkyv", "bincode"])
```
