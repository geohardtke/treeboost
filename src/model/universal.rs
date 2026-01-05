//! UniversalModel: Unified boosting framework
//!
//! Supports multiple boosting modes:
//! - **PureTree**: Standard GBDT (histogram-based trees)
//! - **LinearThenTree**: Linear model captures trend, trees capture residuals
//! - **RandomForest**: Parallel trees with bootstrap sampling
//!
//! # Design Rationale
//!
//! Most tabular problems are solved by Linear, Tree, or their combination.
//! UniversalModel provides a single interface for all three patterns.
//!
//! ## When to Use Each Mode
//!
//! - **PureTree**: Most tabular problems, categorical-heavy data
//! - **LinearThenTree**: Time-series with trends, extrapolation beyond training range
//! - **RandomForest**: When robustness and variance reduction are priorities
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::{UniversalModel, BoostingMode, UniversalConfig};
//!
//! let config = UniversalConfig::default()
//!     .with_mode(BoostingMode::LinearThenTree)
//!     .with_num_rounds(100);
//!
//! let model = UniversalModel::train(&dataset, config)?;
//! let predictions = model.predict(&dataset);
//! ```

use crate::dataset::BinnedDataset;
use crate::learner::{LinearBooster, LinearConfig, TreeBooster, TreeConfig, WeakLearner};
use crate::loss::LossFunction;
use crate::tree::Tree;
use crate::Result;
use rand::SeedableRng;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// BoostingMode
// =============================================================================

/// Boosting mode for UniversalModel
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum BoostingMode {
    /// Pure GBDT: Standard histogram-based tree boosting
    ///
    /// Best for: Most tabular problems, categorical-heavy data
    #[default]
    PureTree,

    /// Linear + Tree: Linear model first, then trees on residuals
    ///
    /// Best for: Time-series with trends, extrapolation beyond training range
    ///
    /// How it works:
    /// 1. Train LinearBooster to capture global trend
    /// 2. Compute residuals: r = y - linear_pred
    /// 3. Train TreeBoosters on residuals (non-linear nuances)
    /// 4. Final: linear_pred + tree_pred
    LinearThenTree,

    /// Random Forest: Parallel trees with bootstrap sampling
    ///
    /// Best for: Robustness, variance reduction, when overfitting is a concern
    ///
    /// How it works:
    /// - learning_rate = 1.0 (full contribution per tree)
    /// - Each tree trained on bootstrap sample
    /// - Trees trained in parallel (independent)
    /// - Average predictions
    RandomForest,
}

// =============================================================================
// UniversalConfig
// =============================================================================

/// Configuration for UniversalModel
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct UniversalConfig {
    /// Boosting mode
    pub mode: BoostingMode,

    /// Number of boosting rounds (trees for PureTree/LinearThenTree, or total trees for RF)
    pub num_rounds: usize,

    /// Tree configuration
    pub tree_config: TreeConfig,

    /// Linear configuration (for LinearThenTree mode)
    pub linear_config: LinearConfig,

    /// Learning rate for gradient boosting (not used in RandomForest mode)
    pub learning_rate: f32,

    /// Row subsampling ratio (0.0-1.0)
    pub subsample: f32,

    /// Validation ratio for early stopping (0.0 to disable)
    pub validation_ratio: f32,

    /// Early stopping rounds (0 to disable)
    pub early_stopping_rounds: usize,

    /// Random seed
    pub seed: u64,

    /// Number of linear boosting rounds before trees (LinearThenTree mode)
    pub linear_rounds: usize,

    /// Maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// LinearThenTree mode unpacks binned u8 data to f32 (4x expansion).
    /// Set this to limit memory usage. If exceeded:
    /// - `0` = No limit (default, may cause OOM on large datasets)
    /// - `> 0` = Error if estimated memory exceeds this limit
    ///
    /// **Rule of thumb**: 100M rows × 100 features = ~40GB memory
    pub max_linear_memory_mb: usize,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            mode: BoostingMode::PureTree,
            num_rounds: 100,
            tree_config: TreeConfig::default(),
            linear_config: LinearConfig::default(),
            learning_rate: 0.1,
            subsample: 1.0,
            validation_ratio: 0.0,
            early_stopping_rounds: 0,
            seed: 42,
            linear_rounds: 10,
            max_linear_memory_mb: 0, // No limit by default
        }
    }
}

impl UniversalConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: BoostingMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_num_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    pub fn with_tree_config(mut self, config: TreeConfig) -> Self {
        self.tree_config = config;
        self
    }

    pub fn with_linear_config(mut self, config: LinearConfig) -> Self {
        self.linear_config = config;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr.clamp(0.0, 1.0);
        self
    }

    pub fn with_subsample(mut self, ratio: f32) -> Self {
        self.subsample = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn with_validation_ratio(mut self, ratio: f32) -> Self {
        self.validation_ratio = ratio.clamp(0.0, 0.5);
        self
    }

    pub fn with_early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = rounds;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_linear_rounds(mut self, rounds: usize) -> Self {
        self.linear_rounds = rounds;
        self
    }

    /// Set maximum memory (MB) for LinearThenTree raw feature extraction
    ///
    /// # Arguments
    /// - `mb`: Maximum memory in megabytes (0 = no limit)
    ///
    /// # Example
    /// ```ignore
    /// let config = UniversalConfig::new()
    ///     .with_mode(BoostingMode::LinearThenTree)
    ///     .with_max_linear_memory_mb(4096); // 4GB limit
    /// ```
    pub fn with_max_linear_memory_mb(mut self, mb: usize) -> Self {
        self.max_linear_memory_mb = mb;
        self
    }

    /// Estimate memory usage (bytes) for LinearThenTree raw feature extraction
    pub fn estimate_linear_memory(&self, num_rows: usize, num_features: usize) -> usize {
        // f32 = 4 bytes per element
        num_rows * num_features * 4
    }
}

// =============================================================================
// UniversalModel
// =============================================================================

/// Unified boosting model supporting multiple modes
///
/// # Modes
///
/// - **PureTree**: Standard GBDT with histogram-based trees
/// - **LinearThenTree**: Linear model captures trend, trees capture residuals
/// - **RandomForest**: Parallel trees with bootstrap sampling
#[derive(Debug, Clone)]
pub struct UniversalModel {
    /// Training configuration
    config: UniversalConfig,

    /// Linear booster (for LinearThenTree mode)
    linear_booster: Option<LinearBooster>,

    /// Ensemble of trained trees
    trees: Vec<Tree>,

    /// Base prediction (initial value)
    base_prediction: f32,

    /// Number of features
    num_features: usize,
}

impl UniversalModel {
    /// Train a UniversalModel on binned data
    pub fn train(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        match config.mode {
            BoostingMode::PureTree => Self::train_pure_tree(dataset, config, loss_fn),
            BoostingMode::LinearThenTree => Self::train_linear_then_tree(dataset, config, loss_fn),
            BoostingMode::RandomForest => Self::train_random_forest(dataset, config, loss_fn),
        }
    }

    // =========================================================================
    // PureTree Mode
    // =========================================================================

    fn train_pure_tree(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Initial prediction
        let base_prediction = loss_fn.initial_prediction(targets);
        let mut predictions = vec![base_prediction; num_rows];

        // Gradient/Hessian buffers
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];

        // Training indices (could add validation split here)
        let train_indices: Vec<usize> = (0..num_rows).collect();

        // Tree booster template
        let tree_config = config.tree_config.clone().with_learning_rate(config.learning_rate);

        let mut trees = Vec::with_capacity(config.num_rounds);

        for _round in 0..config.num_rounds {
            // Compute gradients and hessians
            for &idx in &train_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }

            // Grow tree
            let mut booster = TreeBooster::new(tree_config.clone());
            booster.fit_on_gradients(dataset, &gradients, &hessians, None)?;

            // Update predictions
            if let Some(tree) = booster.tree() {
                tree.predict_batch_add(dataset, &mut predictions);
                trees.push(booster.take_tree().unwrap());
            }
        }

        Ok(Self {
            config,
            linear_booster: None,
            trees,
            base_prediction,
            num_features,
        })
    }

    // =========================================================================
    // LinearThenTree Mode
    // =========================================================================

    fn train_linear_then_tree(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Memory safety check for LinearThenTree mode
        let estimated_bytes = config.estimate_linear_memory(num_rows, num_features);
        let estimated_mb = estimated_bytes / (1024 * 1024);

        if config.max_linear_memory_mb > 0 && estimated_mb > config.max_linear_memory_mb {
            return Err(crate::TreeBoostError::Config(format!(
                "LinearThenTree mode would require ~{}MB for raw feature extraction \
                 ({}rows × {}features × 4bytes), exceeding limit of {}MB. \
                 Options: (1) Increase max_linear_memory_mb, (2) Use PureTree mode, \
                 (3) Reduce dataset size, (4) Use fewer features.",
                estimated_mb, num_rows, num_features, config.max_linear_memory_mb
            )));
        }

        // Warn on very large allocations (>1GB) even without explicit limit
        if estimated_mb > 1024 {
            eprintln!(
                "Warning: LinearThenTree will allocate ~{}MB for raw features. \
                 Consider setting max_linear_memory_mb to prevent OOM.",
                estimated_mb
            );
        }

        // Extract raw features for linear model
        let raw_features = Self::extract_raw_features(dataset);

        // Initial prediction
        let base_prediction = loss_fn.initial_prediction(targets);
        let mut predictions = vec![base_prediction; num_rows];

        // Gradient/Hessian buffers
        let mut gradients = vec![0.0f32; num_rows];
        let mut hessians = vec![0.0f32; num_rows];

        // =====================================================================
        // Phase 1: Train Linear Booster
        // =====================================================================

        let mut linear_booster = LinearBooster::new(num_features, config.linear_config.clone());

        for _linear_round in 0..config.linear_rounds {
            // Compute gradients
            for i in 0..num_rows {
                let (g, h) = loss_fn.gradient_hessian(targets[i], predictions[i]);
                gradients[i] = g;
                hessians[i] = h;
            }

            // Fit linear booster
            linear_booster.fit_on_gradients(&raw_features, num_features, &gradients, &hessians)?;

            // Update predictions with linear contribution
            let linear_preds = linear_booster.predict_batch(&raw_features, num_features);
            for i in 0..num_rows {
                predictions[i] = base_prediction + linear_preds[i];
            }
        }

        // =====================================================================
        // Phase 2: Train Trees on Residuals
        // =====================================================================

        let tree_config = config.tree_config.clone().with_learning_rate(config.learning_rate);
        let mut trees = Vec::with_capacity(config.num_rounds);

        for _round in 0..config.num_rounds {
            // Compute gradients (on residuals after linear)
            for i in 0..num_rows {
                let (g, h) = loss_fn.gradient_hessian(targets[i], predictions[i]);
                gradients[i] = g;
                hessians[i] = h;
            }

            // Grow tree
            let mut booster = TreeBooster::new(tree_config.clone());
            booster.fit_on_gradients(dataset, &gradients, &hessians, None)?;

            // Update predictions
            if let Some(tree) = booster.tree() {
                tree.predict_batch_add(dataset, &mut predictions);
                trees.push(booster.take_tree().unwrap());
            }
        }

        Ok(Self {
            config,
            linear_booster: Some(linear_booster),
            trees,
            base_prediction,
            num_features,
        })
    }

    // =========================================================================
    // RandomForest Mode
    // =========================================================================

    fn train_random_forest(
        dataset: &BinnedDataset,
        config: UniversalConfig,
        loss_fn: &dyn LossFunction,
    ) -> Result<Self> {
        let targets = dataset.targets();
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Initial prediction (mean for RF)
        let base_prediction = loss_fn.initial_prediction(targets);

        // RF uses learning_rate = 1.0 (each tree contributes fully)
        let tree_config = config.tree_config.clone().with_learning_rate(1.0);

        // Train trees in parallel with bootstrap samples
        let trees: Vec<Tree> = (0..config.num_rounds)
            .into_par_iter()
            .filter_map(|seed_offset| {
                // Bootstrap sample
                let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed + seed_offset as u64);
                let bootstrap_indices: Vec<usize> = (0..num_rows)
                    .map(|_| {
                        use rand::Rng;
                        rng.gen_range(0..num_rows)
                    })
                    .collect();

                // Compute gradients for this bootstrap sample
                // For RF, we fit to residuals from base prediction
                let mut gradients = vec![0.0f32; num_rows];
                let mut hessians = vec![0.0f32; num_rows];

                for &idx in &bootstrap_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], base_prediction);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }

                // Grow tree on bootstrap sample
                let mut booster = TreeBooster::new(tree_config.clone());
                if booster
                    .fit_on_gradients(dataset, &gradients, &hessians, Some(&bootstrap_indices))
                    .is_ok()
                {
                    booster.take_tree()
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            config,
            linear_booster: None,
            trees,
            base_prediction,
            num_features,
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Extract raw feature values from BinnedDataset
    ///
    /// Returns row-major f32 array for linear model training.
    fn extract_raw_features(dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();
        let feature_info = dataset.all_feature_info();

        let mut raw_features = vec![0.0f32; num_rows * num_features];

        for f in 0..num_features {
            let info = &feature_info[f];
            let boundaries = &info.bin_boundaries;

            for r in 0..num_rows {
                let bin = dataset.get_bin(r, f) as usize;

                // Convert bin back to approximate raw value
                // Use bin center as the raw value approximation
                let raw_value = if boundaries.is_empty() {
                    bin as f32
                } else if bin == 0 {
                    boundaries.first().copied().unwrap_or(0.0) as f32
                } else if bin >= boundaries.len() {
                    boundaries.last().copied().unwrap_or(0.0) as f32
                } else {
                    // Midpoint between bin boundaries
                    ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0) as f32
                };

                raw_features[r * num_features + f] = raw_value;
            }
        }

        raw_features
    }

    /// Extract raw feature values for a single row
    ///
    /// More efficient than extract_raw_features() when only predicting one row.
    fn extract_raw_features_row(dataset: &BinnedDataset, row_idx: usize) -> Vec<f32> {
        let feature_info = dataset.all_feature_info();
        let mut raw_features = Vec::with_capacity(feature_info.len());

        for (f, info) in feature_info.iter().enumerate() {
            let boundaries = &info.bin_boundaries;
            let bin = dataset.get_bin(row_idx, f) as usize;

            // Convert bin back to approximate raw value using bin center
            let raw_value = if boundaries.is_empty() {
                bin as f32
            } else if bin == 0 {
                boundaries.first().copied().unwrap_or(0.0) as f32
            } else if bin >= boundaries.len() {
                boundaries.last().copied().unwrap_or(0.0) as f32
            } else {
                // Midpoint between bin boundaries
                ((boundaries[bin - 1] + boundaries[bin.min(boundaries.len() - 1)]) / 2.0) as f32
            };

            raw_features.push(raw_value);
        }

        raw_features
    }

    // =========================================================================
    // Prediction
    // =========================================================================

    /// Predict for all rows in dataset
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let mut predictions = vec![self.base_prediction; num_rows];

        // Add linear contribution (if present)
        if let Some(ref linear) = self.linear_booster {
            let raw_features = Self::extract_raw_features(dataset);
            let linear_preds = linear.predict_batch(&raw_features, self.num_features);
            for i in 0..num_rows {
                predictions[i] += linear_preds[i];
            }
        }

        // Add tree contributions
        match self.config.mode {
            BoostingMode::RandomForest => {
                // RandomForest: Each tree is independent (trained on bootstrap sample with lr=1.0)
                // We AVERAGE predictions because each tree estimates the full target independently.
                // This differs from boosting modes which SUM scaled contributions.
                if !self.trees.is_empty() {
                    let mut tree_sum = vec![0.0f32; num_rows];
                    for tree in &self.trees {
                        tree.predict_batch_add(dataset, &mut tree_sum);
                    }
                    let scale = 1.0 / self.trees.len() as f32;
                    for i in 0..num_rows {
                        predictions[i] += tree_sum[i] * scale;
                    }
                }
            }
            _ => {
                // Boosting modes (PureTree, LinearThenTree): Trees are trained sequentially
                // on residuals, each contributing a scaled correction. SUM all contributions.
                for tree in &self.trees {
                    tree.predict_batch_add(dataset, &mut predictions);
                }
            }
        }

        predictions
    }

    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        let mut pred = self.base_prediction;

        // Add linear contribution (use optimized single-row extraction)
        if let Some(ref linear) = self.linear_booster {
            let raw_features = Self::extract_raw_features_row(dataset, row_idx);
            // predict_row expects row_idx into the feature slice, which is now 0
            // since we extracted just one row
            pred += linear.predict_row(&raw_features, self.num_features, 0);
        }

        // Add tree contributions
        match self.config.mode {
            BoostingMode::RandomForest => {
                // RandomForest: Each tree is independent (trained on bootstrap sample with lr=1.0)
                // We AVERAGE predictions because each tree estimates the full target independently.
                // This differs from boosting modes which SUM scaled contributions.
                if !self.trees.is_empty() {
                    let tree_sum: f32 = self.trees.iter().map(|t| t.predict_row(dataset, row_idx)).sum();
                    pred += tree_sum / self.trees.len() as f32;
                }
            }
            _ => {
                // Boosting modes (PureTree, LinearThenTree): Trees are trained sequentially
                // on residuals, each contributing a scaled correction. SUM all contributions.
                for tree in &self.trees {
                    pred += tree.predict_row(dataset, row_idx);
                }
            }
        }

        pred
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the boosting mode
    pub fn mode(&self) -> BoostingMode {
        self.config.mode
    }

    /// Get training configuration
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Get number of trees
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get base prediction
    pub fn base_prediction(&self) -> f32 {
        self.base_prediction
    }

    /// Check if model has linear component
    pub fn has_linear(&self) -> bool {
        self.linear_booster.is_some()
    }

    /// Get linear booster reference (if present)
    pub fn linear_booster(&self) -> Option<&LinearBooster> {
        self.linear_booster.as_ref()
    }

    /// Get trees
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }
}

// =============================================================================
// TunableModel Implementation
// =============================================================================

use crate::tuner::{ParamValue, TunableModel};
use std::collections::HashMap;

impl TunableModel for UniversalModel {
    type Config = UniversalConfig;

    fn train(dataset: &BinnedDataset, config: &Self::Config) -> crate::Result<Self> {
        // Create a default MSE loss for tuning (loss type could be parameterized later)
        let loss_fn = crate::loss::MseLoss::new();
        Self::train(dataset, config.clone(), &loss_fn)
    }

    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        UniversalModel::predict(self, dataset)
    }

    fn num_trees(&self) -> usize {
        self.trees.len()
    }

    fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>) {
        for (name, value) in params {
            match (name.as_str(), value) {
                // Categorical: boosting mode
                ("mode", ParamValue::Categorical(v)) => {
                    config.mode = match v.as_str() {
                        "PureTree" => BoostingMode::PureTree,
                        "LinearThenTree" => BoostingMode::LinearThenTree,
                        "RandomForest" => BoostingMode::RandomForest,
                        _ => BoostingMode::PureTree, // Default fallback
                    };
                }
                // Numeric parameters
                ("num_rounds", ParamValue::Numeric(v)) => config.num_rounds = *v as usize,
                ("learning_rate", ParamValue::Numeric(v)) => config.learning_rate = *v,
                ("subsample", ParamValue::Numeric(v)) => config.subsample = *v,
                ("validation_ratio", ParamValue::Numeric(v)) => config.validation_ratio = *v,
                ("early_stopping_rounds", ParamValue::Numeric(v)) => {
                    config.early_stopping_rounds = *v as usize
                }
                ("linear_rounds", ParamValue::Numeric(v)) => config.linear_rounds = *v as usize,
                // Tree config parameters (prefixed with tree_)
                ("tree_max_depth", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_max_depth(*v as usize)
                }
                ("tree_max_leaves", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_max_leaves(*v as usize)
                }
                ("tree_lambda", ParamValue::Numeric(v)) => {
                    config.tree_config = config.tree_config.clone().with_lambda(*v)
                }
                // Linear config parameters (prefixed with linear_)
                ("linear_lambda", ParamValue::Numeric(v)) => {
                    config.linear_config = config.linear_config.clone().with_lambda(*v)
                }
                ("linear_max_iter", ParamValue::Numeric(v)) => {
                    config.linear_config = config.linear_config.clone().with_max_iter(*v as usize)
                }
                _ => {} // Unknown params are ignored
            }
        }
    }

    fn valid_params() -> &'static [&'static str] {
        &[
            // Categorical
            "mode",
            // Numeric
            "num_rounds",
            "learning_rate",
            "subsample",
            "validation_ratio",
            "early_stopping_rounds",
            "linear_rounds",
            // Tree config
            "tree_max_depth",
            "tree_max_leaves",
            "tree_lambda",
            // Linear config
            "linear_lambda",
            "linear_max_iter",
        ]
    }

    fn default_config() -> Self::Config {
        UniversalConfig::default()
    }

    fn get_learning_rate(config: &Self::Config) -> f32 {
        config.learning_rate
    }

    fn configure_validation(
        config: &mut Self::Config,
        validation_ratio: f32,
        early_stopping_rounds: usize,
    ) {
        config.validation_ratio = validation_ratio;
        config.early_stopping_rounds = early_stopping_rounds;
    }

    fn set_num_rounds(config: &mut Self::Config, num_rounds: usize) {
        config.num_rounds = num_rounds;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};
    use crate::loss::MseLoss;

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        // Linear relationship with some noise
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| (i as f32) * 0.1 + (i % 10) as f32 * 0.01)
            .collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: (0..255).map(|b| b as f64).collect(),
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_universal_config_defaults() {
        let config = UniversalConfig::default();
        assert_eq!(config.mode, BoostingMode::PureTree);
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.learning_rate, 0.1);
    }

    #[test]
    fn test_universal_config_builder() {
        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(50)
            .with_learning_rate(0.05)
            .with_linear_rounds(5);

        assert_eq!(config.mode, BoostingMode::LinearThenTree);
        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.linear_rounds, 5);
    }

    #[test]
    fn test_pure_tree_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::PureTree);
        assert_eq!(model.num_trees(), 5);
        assert!(!model.has_linear());
    }

    #[test]
    fn test_pure_tree_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_linear_then_tree_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(5)
            .with_linear_rounds(3);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::LinearThenTree);
        assert_eq!(model.num_trees(), 5);
        assert!(model.has_linear());
    }

    #[test]
    fn test_linear_then_tree_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::LinearThenTree)
            .with_num_rounds(5)
            .with_linear_rounds(3);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_random_forest_training() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::RandomForest)
            .with_num_rounds(10);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        assert_eq!(model.mode(), BoostingMode::RandomForest);
        assert!(model.num_trees() <= 10); // Some trees might fail to grow
        assert!(!model.has_linear());
    }

    #[test]
    fn test_random_forest_prediction() {
        let dataset = create_test_dataset(100, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::RandomForest)
            .with_num_rounds(10);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();
        let predictions = model.predict(&dataset);

        assert_eq!(predictions.len(), 100);
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_single_row_prediction_matches_batch() {
        let dataset = create_test_dataset(50, 3);
        let loss = MseLoss;

        let config = UniversalConfig::new()
            .with_mode(BoostingMode::PureTree)
            .with_num_rounds(5);

        let model = UniversalModel::train(&dataset, config, &loss).unwrap();

        let batch_preds = model.predict(&dataset);
        for i in 0..50 {
            let single_pred = model.predict_row(&dataset, i);
            assert!((batch_preds[i] - single_pred).abs() < 1e-5);
        }
    }
}
