//! Tree weak learner for gradient boosting
//!
//! Wraps the TreeGrower infrastructure to provide a clean weak learner interface
//! for tree-based gradient boosting.
//!
//! # Design Note
//!
//! TreeBooster works on BinnedDataset (not raw features) because:
//! - Trees use histogram-based split finding (requires bins)
//! - Binning is done once per dataset, not per tree
//! - Raw values are only needed for prediction (stored in Tree nodes)
//!
//! This is different from LinearBooster which needs raw feature values.

use crate::backend::BackendType;
use crate::dataset::BinnedDataset;
use crate::defaults::learners::tree as tree_defaults;
use crate::tree::{InteractionConstraints, MonotonicConstraint, Tree, TreeGrower};
use crate::Result;
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for tree-based weak learner
///
/// Contains only tree hyperparameters. Backend selection is a runtime concern
/// handled separately by TreeBooster.
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct TreeConfig {
    /// Maximum depth of tree
    pub max_depth: usize,

    /// Maximum number of leaves
    pub max_leaves: usize,

    /// L2 regularization (lambda)
    pub lambda: f32,

    /// Minimum samples per leaf
    pub min_samples_leaf: usize,

    /// Minimum hessian sum per leaf
    pub min_hessian_leaf: f32,

    /// Shannon Entropy regularization weight
    pub entropy_weight: f32,

    /// Minimum gain to make a split
    pub min_gain: f32,

    /// Learning rate for gradient descent optimization
    ///
    /// Controls the step size for tree weight updates in Newton-step gradient descent.
    /// Also acts as ensemble shrinkage because trees are trained iteratively.
    ///
    /// **Range**: (0.0, 1.0] - typically 0.1 for stable convergence
    /// **Default**: 0.1 (conservative to prevent overfitting)
    ///
    /// **Note**: This is distinct from `LinearConfig::shrinkage_factor` which controls
    /// ensemble weighting (not optimization step size). See `LinearConfig` module docs
    /// for detailed explanation of the naming distinction.
    ///
    /// In LinearThenTree mode:
    /// - `TreeConfig::learning_rate` = optimization step size (this field)
    /// - `LinearConfig::shrinkage_factor` = ensemble weighting (different concept)
    pub learning_rate: f32,

    /// Column subsampling ratio (0.0-1.0)
    pub colsample: f32,

    /// Monotonic constraints per feature
    #[serde(default)]
    pub monotonic_constraints: Vec<MonotonicConstraint>,

    /// Feature interaction constraint groups
    #[serde(default)]
    pub interaction_groups: Vec<Vec<usize>>,

    /// Enable era-based splitting (DES)
    pub era_splitting: bool,
}

/// Presets for common tree configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreePreset {
    /// Balanced defaults - good starting point.
    Standard,
    /// Deeper trees for complex signals.
    Deep,
    /// Shallow trees for noisy data or faster training.
    Shallow,
    /// Higher regularization to prevent overfitting.
    Regularized,
    /// Minimal regularization for expressiveness.
    Expressive,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: tree_defaults::DEFAULT_MAX_DEPTH,
            max_leaves: tree_defaults::DEFAULT_MAX_LEAVES,
            lambda: tree_defaults::DEFAULT_TREE_LAMBDA,
            min_samples_leaf: tree_defaults::DEFAULT_MIN_SAMPLES_LEAF,
            min_hessian_leaf: tree_defaults::DEFAULT_MIN_HESSIAN_LEAF,
            entropy_weight: tree_defaults::DEFAULT_ENTROPY_WEIGHT,
            min_gain: tree_defaults::DEFAULT_MIN_GAIN,
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,
            colsample: tree_defaults::DEFAULT_COLSAMPLE,
            monotonic_constraints: Vec::new(),
            interaction_groups: Vec::new(),
            era_splitting: false,
        }
    }
}

impl TreeConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: TreePreset) -> Self {
        match preset {
            TreePreset::Standard => {}
            TreePreset::Deep => {
                self.max_depth = tree_defaults::DEEP_MAX_DEPTH;
            }
            TreePreset::Shallow => {
                self.max_depth = tree_defaults::SHALLOW_MAX_DEPTH;
            }
            TreePreset::Regularized => {
                self.lambda = tree_defaults::REGULARIZED_TREE_LAMBDA;
                self.entropy_weight = tree_defaults::REGULARIZED_ENTROPY_WEIGHT;
            }
            TreePreset::Expressive => {
                self.lambda = tree_defaults::EXPRESSIVE_TREE_LAMBDA;
                self.entropy_weight = tree_defaults::DEFAULT_ENTROPY_WEIGHT;
                self.min_gain = tree_defaults::DEFAULT_MIN_GAIN;
            }
        }
        self
    }

    /// Set maximum tree depth
    ///
    /// Returns error if max_depth is 0 or > 20.
    pub fn with_max_depth(mut self, max_depth: usize) -> Result<Self> {
        if max_depth == 0 || max_depth > 20 {
            return Err(crate::TreeBoostError::Config(format!(
                "max_depth must be in [1, 20], got {}",
                max_depth
            )));
        }
        self.max_depth = max_depth;
        Ok(self)
    }

    /// Set maximum number of leaves
    ///
    /// Returns error if max_leaves < 2.
    pub fn with_max_leaves(mut self, max_leaves: usize) -> Result<Self> {
        if max_leaves < 2 {
            return Err(crate::TreeBoostError::Config(format!(
                "max_leaves must be >= 2, got {}",
                max_leaves
            )));
        }
        self.max_leaves = max_leaves;
        Ok(self)
    }

    /// Set L2 regularization (lambda)
    ///
    /// Returns error if lambda < 0.0.
    pub fn with_lambda(mut self, lambda: f32) -> Result<Self> {
        if lambda < 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "lambda must be >= 0.0, got {}",
                lambda
            )));
        }
        self.lambda = lambda;
        Ok(self)
    }

    /// Set minimum samples per leaf
    ///
    /// Returns error if min_samples < 1.
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Result<Self> {
        if min_samples < 1 {
            return Err(crate::TreeBoostError::Config(format!(
                "min_samples_leaf must be >= 1, got {}",
                min_samples
            )));
        }
        self.min_samples_leaf = min_samples;
        Ok(self)
    }

    /// Set minimum hessian sum per leaf
    ///
    /// Returns error if min_hessian < 0.0.
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Result<Self> {
        if min_hessian < 0.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "min_hessian_leaf must be >= 0.0, got {}",
                min_hessian
            )));
        }
        self.min_hessian_leaf = min_hessian;
        Ok(self)
    }

    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Set learning rate for gradient descent optimization
    ///
    /// Returns error if learning_rate is not in (0.0, 1.0].
    pub fn with_learning_rate(mut self, lr: f32) -> Result<Self> {
        if lr <= 0.0 || lr > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "learning_rate must be in (0.0, 1.0], got {}",
                lr
            )));
        }
        self.learning_rate = lr;
        Ok(self)
    }

    /// Set column subsampling ratio
    ///
    /// Returns error if colsample is not in (0.0, 1.0].
    pub fn with_colsample(mut self, colsample: f32) -> Result<Self> {
        if colsample <= 0.0 || colsample > 1.0 {
            return Err(crate::TreeBoostError::Config(format!(
                "colsample must be in (0.0, 1.0], got {}",
                colsample
            )));
        }
        self.colsample = colsample;
        Ok(self)
    }

    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    pub fn with_interaction_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.interaction_groups = groups;
        self
    }

    pub fn with_era_splitting(mut self, enabled: bool) -> Self {
        self.era_splitting = enabled;
        self
    }

    /// Build a TreeGrower from this config
    ///
    /// # Arguments
    /// - `num_features`: Number of features in dataset
    /// - `backend`: Optional backend override (uses Auto if None)
    pub(crate) fn build_grower(
        &self,
        num_features: usize,
        backend: Option<BackendType>,
    ) -> TreeGrower {
        let interaction_constraints = if self.interaction_groups.is_empty() {
            InteractionConstraints::new()
        } else {
            InteractionConstraints::from_groups(self.interaction_groups.clone(), num_features)
        };

        TreeGrower::new()
            .with_max_depth(self.max_depth)
            .with_max_leaves(self.max_leaves)
            .with_lambda(self.lambda)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_hessian_leaf(self.min_hessian_leaf)
            .with_entropy_weight(self.entropy_weight)
            .with_min_gain(self.min_gain)
            // IMPORTANT: TreeConfig.learning_rate controls optimization step size.
            // TreeGrower calls it learning_rate (it's the same thing - gradient descent step size).
            // This is distinct from LinearConfig.shrinkage_factor which controls ensemble weighting.
            // See LinearConfig module docs for architectural details on this naming distinction.
            .with_learning_rate(self.learning_rate)
            .with_colsample(self.colsample)
            .with_monotonic_constraints(self.monotonic_constraints.clone())
            .with_interaction_constraints(interaction_constraints)
            .with_backend(backend.unwrap_or(BackendType::Auto))
            .with_era_splitting(self.era_splitting)
    }
}

// =============================================================================
// TreeBooster
// =============================================================================

/// Tree weak learner for gradient boosting
///
/// Wraps the TreeGrower + Tree infrastructure to provide a clean interface
/// for use in gradient boosting ensembles.
///
/// # Usage
///
/// ```ignore
/// let config = TreeConfig::default().with_max_depth(6);
/// let mut booster = TreeBooster::new(config);
///
/// // Fit on gradients
/// booster.fit_on_gradients(&dataset, &gradients, &hessians, None)?;
///
/// // Predict
/// let predictions = booster.predict_batch(&dataset);
/// ```
///
/// # Design
///
/// Unlike LinearBooster which implements WeakLearner (for raw features),
/// TreeBooster has its own interface because:
/// - Trees work on binned data (BinnedDataset)
/// - Trees use histogram-based split finding
/// - Mixing interfaces would create unnecessary complexity
#[derive(Debug, Clone)]
pub struct TreeBooster {
    /// The trained tree (None if not fitted yet)
    tree: Option<Tree>,

    /// Configuration
    config: TreeConfig,

    /// Cached grower (lazily initialized)
    grower: Option<TreeGrower>,

    /// Number of features (set on first fit)
    num_features: Option<usize>,

    /// Backend type (runtime only, not serialized)
    backend: BackendType,
}

impl TreeBooster {
    /// Create a new tree booster
    pub fn new(config: TreeConfig) -> Self {
        Self {
            tree: None,
            config,
            grower: None,
            num_features: None,
            backend: BackendType::Auto,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(TreeConfig::default())
    }

    /// Set backend type (runtime setting, not serialized)
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.backend = backend;
        self
    }

    /// Get the trained tree (if fitted)
    pub fn tree(&self) -> Option<&Tree> {
        self.tree.as_ref()
    }

    /// Get configuration
    pub fn config(&self) -> &TreeConfig {
        &self.config
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }

    /// Fit one boosting iteration on gradients/hessians
    ///
    /// # Arguments
    /// - `dataset`: Binned training data
    /// - `gradients`: Negative gradient of loss for each sample
    /// - `hessians`: Second derivative of loss for each sample
    /// - `row_indices`: Optional subset of rows to use (for subsampling)
    pub fn fit_on_gradients(
        &mut self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: Option<&[usize]>,
    ) -> Result<()> {
        let num_features = dataset.num_features();

        // Initialize grower on first fit
        if self.grower.is_none() {
            self.grower = Some(self.config.build_grower(num_features, Some(self.backend)));
            self.num_features = Some(num_features);
        }

        let grower = self
            .grower
            .as_ref()
            .expect("grower must be Some after initialization above");

        // Grow tree
        self.tree = Some(match row_indices {
            Some(indices) => grower.grow_with_indices(dataset, gradients, hessians, indices)?,
            None => grower.grow(dataset, gradients, hessians)?,
        });

        Ok(())
    }

    /// Predict for all rows
    ///
    /// Returns zero vector if not fitted.
    pub fn predict_batch(&self, dataset: &BinnedDataset) -> Vec<f32> {
        match &self.tree {
            Some(tree) => tree.predict_all(dataset),
            None => vec![0.0; dataset.num_rows()],
        }
    }

    /// Add predictions to existing buffer (more efficient for ensembles)
    pub fn predict_batch_add(&self, dataset: &BinnedDataset, predictions: &mut [f32]) {
        if let Some(tree) = &self.tree {
            tree.predict_batch_add(dataset, predictions);
        }
    }

    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        match &self.tree {
            Some(tree) => tree.predict_row(dataset, row_idx),
            None => 0.0,
        }
    }

    /// Get number of parameters (tree complexity measure)
    ///
    /// Returns the number of leaves, which corresponds to the number
    /// of distinct prediction values the tree can output.
    pub fn num_params(&self) -> usize {
        match &self.tree {
            Some(tree) => tree.num_leaves(),
            None => 0,
        }
    }

    /// Reset the booster (clear fitted tree)
    pub fn reset(&mut self) {
        self.tree = None;
        // Keep grower cached for reuse
    }

    /// Get number of nodes in the tree
    pub fn num_nodes(&self) -> usize {
        match &self.tree {
            Some(tree) => tree.num_nodes(),
            None => 0,
        }
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        match &self.tree {
            Some(tree) => tree.max_depth(),
            None => 0,
        }
    }

    /// Extract the tree (consumes the booster's tree)
    pub fn take_tree(&mut self) -> Option<Tree> {
        self.tree.take()
    }

    /// Set the tree directly (for deserialization or ensemble building)
    pub fn set_tree(&mut self, tree: Tree) {
        self.tree = Some(tree);
    }
}

// =============================================================================
// Serialization support
// =============================================================================

/// Serializable version of TreeBooster (tree + config only)
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct SerializableTreeBooster {
    /// The trained tree
    pub tree: Option<Tree>,
    /// Configuration
    pub config: TreeConfig,
}

impl From<TreeBooster> for SerializableTreeBooster {
    fn from(booster: TreeBooster) -> Self {
        Self {
            tree: booster.tree,
            config: booster.config,
        }
    }
}

impl From<SerializableTreeBooster> for TreeBooster {
    fn from(ser: SerializableTreeBooster) -> Self {
        Self {
            tree: ser.tree,
            config: ser.config,
            grower: None,
            num_features: None,
            backend: BackendType::Auto,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32).sin()).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_tree_config_defaults() {
        let config = TreeConfig::default();
        assert_eq!(config.max_depth, 6);
        assert_eq!(config.max_leaves, 31);
        assert_eq!(config.lambda, 1.0);
        assert_eq!(config.learning_rate, 0.1);
    }

    #[test]
    fn test_tree_config_builder() {
        let config = TreeConfig::new()
            .with_max_depth(4)
            .and_then(|c| c.with_max_leaves(15))
            .and_then(|c| c.with_lambda(0.5))
            .and_then(|c| c.with_learning_rate(0.05))
            .unwrap();

        assert_eq!(config.max_depth, 4);
        assert_eq!(config.max_leaves, 15);
        assert_eq!(config.lambda, 0.5);
        assert_eq!(config.learning_rate, 0.05);
    }

    #[test]
    fn test_tree_booster_creation() {
        let config = TreeConfig::default();
        let booster = TreeBooster::new(config);

        assert!(!booster.is_fitted());
        assert!(booster.tree().is_none());
        assert_eq!(booster.num_params(), 0);
    }

    #[test]
    fn test_tree_booster_fit() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let hessians = vec![1.0; 100];

        let config = TreeConfig::default()
            .with_max_depth(3)
            .unwrap()
            .with_min_gain(0.0);

        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, None)
            .unwrap();

        assert!(booster.is_fitted());
        assert!(booster.tree().is_some());
        assert!(booster.num_params() > 0);
    }

    #[test]
    fn test_tree_booster_predict() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let hessians = vec![1.0; 100];

        let config = TreeConfig::default();
        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, None)
            .unwrap();

        let predictions = booster.predict_batch(&dataset);
        assert_eq!(predictions.len(), 100);

        // All predictions should be finite
        for pred in &predictions {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_tree_booster_predict_add() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let hessians = vec![1.0; 100];

        let config = TreeConfig::default();
        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, None)
            .unwrap();

        // Predict batch
        let batch_preds = booster.predict_batch(&dataset);

        // Predict add (start from zeros)
        let mut add_preds = vec![0.0; 100];
        booster.predict_batch_add(&dataset, &mut add_preds);

        // Should be equal
        for i in 0..100 {
            assert!((batch_preds[i] - add_preds[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tree_booster_with_indices() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }).collect();
        let hessians = vec![1.0; 100];

        // Only use first 50 rows for training
        let row_indices: Vec<usize> = (0..50).collect();

        let config = TreeConfig::default();
        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, Some(&row_indices))
            .unwrap();

        assert!(booster.is_fitted());
    }

    #[test]
    fn test_tree_booster_reset() {
        let dataset = create_test_dataset(100, 3);
        let gradients = vec![-1.0; 100];
        let hessians = vec![1.0; 100];

        let config = TreeConfig::default();
        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, None)
            .unwrap();

        assert!(booster.is_fitted());
        booster.reset();
        assert!(!booster.is_fitted());
    }

    #[test]
    fn test_tree_booster_serializable() {
        let dataset = create_test_dataset(100, 3);
        let gradients = vec![-1.0; 100];
        let hessians = vec![1.0; 100];

        let config = TreeConfig::default();
        let mut booster = TreeBooster::new(config);
        booster
            .fit_on_gradients(&dataset, &gradients, &hessians, None)
            .unwrap();

        // Convert to serializable
        let ser: SerializableTreeBooster = booster.into();
        assert!(ser.tree.is_some());

        // Convert back
        let restored: TreeBooster = ser.into();
        assert!(restored.is_fitted());
    }

    // =========================================================================
    // Validation Tests for with_*() Methods
    // =========================================================================

    #[test]
    fn test_with_lambda_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_lambda(0.0).is_ok());
        assert!(config.clone().with_lambda(1.0).is_ok());
        assert!(config.clone().with_lambda(10.0).is_ok());

        // Invalid case
        let result = config.with_lambda(-0.1);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("lambda must be >= 0.0"));
        assert!(err_msg.contains("-0.1"));
    }

    #[test]
    fn test_with_min_samples_leaf_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_min_samples_leaf(1).is_ok());
        assert!(config.clone().with_min_samples_leaf(10).is_ok());
        assert!(config.clone().with_min_samples_leaf(100).is_ok());

        // Invalid case
        let result = config.with_min_samples_leaf(0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("min_samples_leaf must be >= 1"));
        assert!(err_msg.contains("0"));
    }

    #[test]
    fn test_with_learning_rate_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_learning_rate(0.001).is_ok());
        assert!(config.clone().with_learning_rate(0.1).is_ok());
        assert!(config.clone().with_learning_rate(1.0).is_ok());

        // Invalid cases: too low
        let result = config.clone().with_learning_rate(0.0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("learning_rate must be in (0.0, 1.0]"));

        // Invalid cases: too high
        let result = config.with_learning_rate(1.5);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("learning_rate must be in (0.0, 1.0]"));
    }

    #[test]
    fn test_with_colsample_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_colsample(0.001).is_ok());
        assert!(config.clone().with_colsample(0.5).is_ok());
        assert!(config.clone().with_colsample(1.0).is_ok());

        // Invalid cases: too low
        let result = config.clone().with_colsample(0.0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("colsample must be in (0.0, 1.0]"));

        // Invalid cases: too high
        let result = config.with_colsample(1.5);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("colsample must be in (0.0, 1.0]"));
    }

    #[test]
    fn test_with_max_depth_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_max_depth(1).is_ok());
        assert!(config.clone().with_max_depth(6).is_ok());
        assert!(config.clone().with_max_depth(20).is_ok());

        // Invalid case: zero
        let result = config.clone().with_max_depth(0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_depth must be in [1, 20]"));

        // Invalid case: too deep
        let result = config.with_max_depth(21);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_depth must be in [1, 20]"));
    }

    #[test]
    fn test_with_max_leaves_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_max_leaves(2).is_ok());
        assert!(config.clone().with_max_leaves(31).is_ok());
        assert!(config.clone().with_max_leaves(1000).is_ok());

        // Invalid case
        let result = config.with_max_leaves(1);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("max_leaves must be >= 2"));
        assert!(err_msg.contains("1"));
    }

    #[test]
    fn test_with_min_hessian_leaf_validation() {
        let config = TreeConfig::new();

        // Valid cases
        assert!(config.clone().with_min_hessian_leaf(0.0).is_ok());
        assert!(config.clone().with_min_hessian_leaf(0.1).is_ok());
        assert!(config.clone().with_min_hessian_leaf(10.0).is_ok());

        // Invalid case
        let result = config.with_min_hessian_leaf(-0.1);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("min_hessian_leaf must be >= 0.0"));
        assert!(err_msg.contains("-0.1"));
    }

    #[test]
    fn test_with_methods_builder_chain() {
        let config = TreeConfig::new()
            .with_max_depth(5)
            .and_then(|c| c.with_lambda(0.5))
            .and_then(|c| c.with_min_samples_leaf(5))
            .and_then(|c| c.with_learning_rate(0.1))
            .and_then(|c| c.with_colsample(0.8));

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.max_depth, 5);
        assert_eq!(config.lambda, 0.5);
        assert_eq!(config.min_samples_leaf, 5);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.colsample, 0.8);
    }

    #[test]
    fn test_with_methods_error_stops_chain() {
        let result = TreeConfig::new()
            .with_max_depth(5)
            .and_then(|c| c.with_learning_rate(1.5)) // Invalid!
            .and_then(|c| c.with_colsample(0.5));

        assert!(result.is_err());
    }
}
