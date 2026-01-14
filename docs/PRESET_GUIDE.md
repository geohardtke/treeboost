# GBDT Preset Selection Guide

## Overview

TreeBoost provides carefully tuned presets for different data scenarios. Choosing the right preset can dramatically improve model performance and prevent overfitting.

## Quick Decision Tree

```
Do you have high-dimensional data (>100 features)?
├─ Yes → Use `Robust` preset
└─ No
   ├─ Is your dataset very small (<10k rows)?
   │  └─ Yes → Use `SmallData` preset
   └─ No
      ├─ Is training speed critical?
      │  ├─ Yes, dataset is massive (>10M rows) → Use `LargeData` preset
      │  └─ Yes, dataset is large (>1M rows) → Use `Speed` preset
      └─ No
         ├─ Do you need maximum accuracy? → Use `Accuracy` preset
         ├─ Do you need uncertainty quantification? → Use `Conformal` preset
         └─ Not sure? → Use `Standard` preset
```

## The Overfitting Problem

**Default GBDT is prone to overfitting on noisy, high-dimensional data!**

Our empirical tests (`tests/rf_robustness.rs`) demonstrate this dramatically:

### Test Setup: "Needle in a Haystack"

- **Dataset**: 2000 rows × 1000 features
- **Signal**: Only 10 features are informative
- **Noise**: 990 features are pure Gaussian noise

### Results

| Configuration                                       | Train RMSE | Val RMSE | Gap         | Interpretation               |
| --------------------------------------------------- | ---------- | -------- | ----------- | ---------------------------- |
| **Naive GBDT** (no sampling)                        | 0.03       | 32.01    | **99,366%** | Severe overfitting to noise! |
| **Regularized GBDT** (colsample=0.8, subsample=0.8) | 12.02      | 28.85    | **140%**    | 993× gap reduction!          |
| **Random Forest** (feature bagging)                 | 30.90      | 34.18    | **11%**     | Best robustness              |

**Key Insight**: Without feature/row sampling, GBDT achieves near-perfect training accuracy (0.03 RMSE) by memorizing noise patterns, leading to catastrophic validation performance (32.01 RMSE).

## Preset Descriptions

### `Standard` - Balanced Defaults

**Best for**: Clean, low-dimensional data with mostly relevant features.

**Configuration**:

```rust
num_rounds:     100
learning_rate:  0.1
max_depth:      6
subsample:      1.0  // No row sampling
colsample:      1.0  // No feature sampling
```

**Use when**:

- Feature count < 100
- Features are curated/selected
- Low noise-to-signal ratio
- Data is well-understood

**⚠️ Warning**: May overfit on noisy or high-dimensional data!

**Example**:

```rust
use treeboost::{GBDTConfig, GbdtPreset};

let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Standard);
```

---

### `Robust` - Noise-Resistant

**Best for**: High-dimensional, noisy data with many irrelevant features.

**Configuration**:

```rust
num_rounds:     100
learning_rate:  0.1
max_depth:      6
subsample:      0.8  // 80% row sampling
colsample:      0.8  // 80% feature sampling
goss_enabled:   false
```

**Use when**:

- Feature count > 100
- Many irrelevant/noisy features
- Risk of overfitting to noise
- Feature importance is unknown
- Financial data with many technical indicators
- Biological data with high-throughput measurements

**Why it works**:

- **Feature bagging**: Randomly uses 80% of features per split → ignores noise features
- **Row bagging**: Randomly uses 80% of samples per tree → variance reduction
- **Random Forest-like robustness**: Without sequential boosting drawbacks

**Evidence**: Reduces train/val gap from 99,366% to 140% on high-dimensional noisy data (993× improvement!).

**Example**:

```rust
use treeboost::{GBDTConfig, GbdtPreset};

// For data with many technical indicators
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Robust);
```

---

### `Accuracy` - Maximum Predictive Power

**Best for**: Complex data where accuracy is paramount and training time is not a constraint.

**Configuration**:

```rust
num_rounds:     200  // 2x default
learning_rate:  0.05 // 0.5x default (slower, more careful)
max_depth:      10   // Deep trees for complex interactions
subsample:      0.8  // Row sampling for generalization
colsample:      0.8  // Feature sampling prevents overfitting
```

**Use when**:

- Accuracy is critical
- Training time is acceptable
- Dataset has complex interactions
- Need to capture subtle patterns

**Note**: Includes feature/row sampling to balance expressiveness (deep trees) with robustness (bagging). This prevents overfitting while maintaining high predictive accuracy.

**Example**:

```rust
use treeboost::{GBDTConfig, GbdtPreset};

let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Accuracy);
```

---

### `Speed` - Fast Training

**Best for**: Very large datasets where training speed is critical.

**Configuration**:

```rust
num_rounds:     100
learning_rate:  0.1
max_depth:      4    // Shallow trees (3-5x faster)
subsample:      1.0
colsample:      1.0
goss_enabled:   true // Gradient-based One-Side Sampling
goss_top_rate:  0.2
goss_other_rate: 0.1
```

**Use when**:

- Dataset > 1M rows
- Training speed is critical
- Willing to trade some accuracy for speed
- Rapid prototyping

**Speed-up**: 3-5x faster than Standard due to shallow trees + GOSS.

**Example**:

```rust
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Speed);
```

---

### `SmallData` - No Subsampling

**Best for**: Very small datasets where every sample matters.

**Configuration**:

```rust
num_rounds:     100
learning_rate:  0.1
max_depth:      6
subsample:      1.0  // Use all data
colsample:      1.0  // Use all features
goss_enabled:   false
```

**Use when**:

- Dataset < 10k rows
- Every sample is valuable
- Data is clean and low-dimensional

**⚠️ Warning**: Can overfit on noisy data! If you have <10k rows BUT high-dimensional/noisy features, use `Robust` instead.

**Example**:

```rust
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::SmallData);
```

---

### `LargeData` - Efficient Large-Scale Training

**Best for**: Massive datasets where memory and speed are constraints.

**Configuration**:

```rust
num_rounds:     100
learning_rate:  0.1
max_depth:      6
subsample:      0.8  // Row sampling
colsample:      1.0  // All features
goss_enabled:   true // GOSS for additional speedup
```

**Use when**:

- Dataset > 10M rows
- Memory is constrained
- Training speed is critical

**Example**:

```rust
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::LargeData);
```

---

### `Conformal` - Uncertainty Quantification

**Best for**: Applications requiring prediction intervals and uncertainty estimates.

**Configuration**:

```rust
num_rounds:          100
learning_rate:       0.1
max_depth:           6
calibration_ratio:   0.2  // 20% of data for calibration
conformal_quantile:  0.9  // 90% coverage intervals
```

**Use when**:

- Need prediction intervals (not just point estimates)
- Uncertainty quantification is critical
- Decision-making under uncertainty
- Risk assessment applications

**Output**: Provides `[lower_bound, prediction, upper_bound]` intervals with statistical guarantees.

**Example**:

```rust
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Conformal);
```

---

## Combining Presets

You can start with a preset and further customize:

```rust
use treeboost::{GBDTConfig, GbdtPreset};

// Start with Robust, but increase rounds for better accuracy
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Robust)
    .with_num_rounds(200)
    .with_learning_rate(0.05);

// Start with Speed, but add some feature sampling for robustness
let config = GBDTConfig::default()
    .with_preset(GbdtPreset::Speed)
    .with_colsample(0.8);
```

## Feature/Row Sampling Explained

### What is Feature Sampling (`colsample`)?

At each split, the algorithm randomly selects a subset of features to consider. With `colsample=0.8`, only 80% of features are candidates for each split.

**Benefits**:

- **Ignores noise features**: Random selection means noise features are often excluded
- **Reduces overfitting**: Trees become more diverse
- **Random Forest-like robustness**: Mimics RF's feature bagging without sacrificing boosting's power

**Analogy**: Like having multiple experts vote, where each expert only looks at a random subset of information. The majority opinion is more robust to individual mistakes.

### What is Row Sampling (`subsample`)?

Each tree is trained on a random subset of training data. With `subsample=0.8`, each tree sees 80% of the data (sampled without replacement per tree).

**Benefits**:

- **Variance reduction**: Trees trained on different data are less correlated
- **Prevents overfitting**: No single tree can memorize the entire dataset
- **Computational savings**: Faster training (20% less data per tree)

**Analogy**: Like conducting multiple polls with different random samples of people. Averaging results reduces sampling bias.

## When to Use Sampling vs. GOSS

| Technique                                | Use When                           | Speed     | Robustness | Compatibility               |
| ---------------------------------------- | ---------------------------------- | --------- | ---------- | --------------------------- |
| **Feature Sampling** (`colsample < 1.0`) | High-dimensional data              | Fast      | High       | Works with everything       |
| **Row Sampling** (`subsample < 1.0`)     | Large datasets, variance reduction | Medium    | High       | Works with everything       |
| **GOSS**                                 | Very large datasets (>10M rows)    | Very Fast | Medium     | Incompatible with subsample |

**Key Insight**: Feature/row sampling provides Random Forest-like robustness while maintaining GBDT's sequential learning advantages.

## Real-World Recommendations

### Financial/Trading Data

```rust
// Many technical indicators, high noise
GbdtPreset::Robust  // ✅ Best choice
```

### Biological/Medical Data

```rust
// High-dimensional gene expression, protein data
GbdtPreset::Robust  // ✅ Best choice
```

### Time-Series Forecasting

```rust
// If features are curated and low-dimensional
GbdtPreset::Standard  // ✅ Good choice

// If many lagged features/technical indicators
GbdtPreset::Robust  // ✅ Better choice
```

### Image/Text Features (After Feature Engineering)

```rust
// Hundreds of extracted features
GbdtPreset::Robust  // ✅ Best choice
```

### Small, Clean Tabular Data

```rust
// <10k rows, <50 features, well-curated
GbdtPreset::SmallData  // ✅ Best choice
```

## Diagnostic: Is Your Model Overfitting?

### Signs of Overfitting

1. **Large train/val gap**: Train loss << Val loss

   ```
   Train RMSE: 0.5, Val RMSE: 2.0  → 300% gap (overfitting!)
   ```

2. **Perfect training accuracy**: Train RMSE ≈ 0

   ```
   Train RMSE: 0.03  → Likely memorizing noise
   ```

3. **Increasing validation loss**: Val loss increases with more trees
   ```
   Trees 50: Val RMSE = 1.5
   Trees 100: Val RMSE = 1.8  → Overfitting after tree 50
   ```

### Solutions

1. **Switch to `Robust` preset**:

   ```rust
   let config = GBDTConfig::default()
       .with_preset(GbdtPreset::Robust);
   ```

2. **Increase regularization**:

   ```rust
   let config = config
       .with_lambda(2.0)           // L2 regularization
       .with_min_gain(0.01)        // Stricter split criteria
       .with_min_samples_leaf(10); // Larger leaves
   ```

3. **Enable early stopping**:

   ```rust
   let config = config
       .with_validation_ratio(0.2)
       .with_early_stopping_rounds(10);
   ```

4. **Reduce model complexity**:
   ```rust
   let config = config
       .with_max_depth(4)          // Shallower trees
       .with_num_rounds(50);       // Fewer trees
   ```

## Empirical Validation

All recommendations in this guide are backed by empirical tests in `tests/rf_robustness.rs`.

**Test scenario**: 2000 rows × 1000 features (990 noise, 10 signal)

```bash
# Run the test yourself
cargo test test_rf_vs_gbdt_noise_robustness -- --ignored --nocapture
```

**Expected output**:

```
Naive GBDT:       Train=0.03, Val=32.01, Gap=99,366%  ❌ Severe overfitting
Regularized GBDT: Train=12.02, Val=28.85, Gap=140%   ✅ 993× improvement
Random Forest:    Train=30.90, Val=34.18, Gap=11%    ✅ Best robustness
```

## Summary: Preset Selection Cheat Sheet

| Data Type     | Feature Count | Dataset Size | Noise Level | **Recommended Preset** |
| ------------- | ------------- | ------------ | ----------- | ---------------------- |
| Financial     | >100          | Any          | High        | **`Robust`** ⭐        |
| Biological    | >100          | Any          | High        | **`Robust`** ⭐        |
| Text/Images   | >100          | Any          | Medium-High | **`Robust`** ⭐        |
| Time-Series   | <100          | Small (<10k) | Low         | `SmallData`            |
| Time-Series   | <100          | Large        | Low         | `Standard`             |
| Clean Tabular | <50           | Small        | Low         | `SmallData`            |
| Clean Tabular | <50           | Large        | Low         | `Standard`             |
| Any           | Any           | >10M rows    | Any         | `LargeData`            |
| Any           | Any           | >1M rows     | Any         | `Speed`                |
| Complex       | <100          | Any          | Low         | `Accuracy`             |
| Uncertainty   | Any           | Any          | Any         | `Conformal`            |

**⭐ = Most important preset for preventing overfitting**

## Questions?

- **Q: Can I combine presets?**

  - A: Yes! Presets are starting points. Apply a preset, then customize further.

- **Q: Why doesn't `Standard` use sampling by default?**

  - A: For backward compatibility and to match industry standards (XGBoost, LightGBM also default to no sampling). However, for most real-world data, `Robust` or `Accuracy` are better starting points.

- **Q: Should I always use `Robust` for high-dimensional data?**

  - A: Yes, unless you're 100% certain all features are relevant. The cost of sampling is minimal (~5% slower) compared to the risk of overfitting.

- **Q: What if I'm not sure which preset to use?**
  - A: Start with `Robust`. It's the safest choice for most real-world data. You can always relax regularization if needed, but recovering from overfitting is harder.
