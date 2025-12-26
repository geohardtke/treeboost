//! GBDT training configuration

use crate::backend::BackendType;
use crate::dataset::OrderingStrategy;
use crate::loss::{LossFunction, MseLoss, PseudoHuberLoss};
use crate::tree::MonotonicConstraint;
use rkyv::{Archive, Deserialize, Serialize};

/// Loss function type for serialization
#[derive(Debug, Clone, Copy, PartialEq, Archive, Serialize, Deserialize)]
pub enum LossType {
    /// Mean Squared Error
    Mse,
    /// Pseudo-Huber Loss with given delta
    PseudoHuber { delta: f32 },
}

impl Default for LossType {
    fn default() -> Self {
        Self::Mse
    }
}

impl LossType {
    /// Create a boxed loss function
    pub fn create(&self) -> Box<dyn LossFunction> {
        match self {
            LossType::Mse => Box::new(MseLoss::new()),
            LossType::PseudoHuber { delta } => Box::new(PseudoHuberLoss::new(*delta)),
        }
    }
}

/// GBDT training configuration
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct GBDTConfig {
    // Ensemble parameters
    /// Number of boosting rounds (trees)
    pub num_rounds: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f32,

    // Tree parameters
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Maximum number of leaves per tree
    pub max_leaves: usize,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: usize,
    /// Minimum hessian sum required in a leaf
    pub min_hessian_leaf: f32,
    /// Minimum gain to make a split
    pub min_gain: f32,

    // Regularization
    /// L2 regularization (lambda)
    pub lambda: f32,
    /// Shannon Entropy regularization weight (beta)
    pub entropy_weight: f32,

    // Loss function
    /// Loss function type
    pub loss_type: LossType,

    // Subsampling
    /// Row subsampling ratio (0.0-1.0) for random subsampling
    pub subsample: f32,
    /// Column subsampling ratio (0.0-1.0)
    pub colsample: f32,

    // GOSS (Gradient-based One-Side Sampling)
    /// Enable GOSS sampling (overrides random subsample when enabled)
    pub goss_enabled: bool,
    /// Ratio of large-gradient samples to keep (default: 0.2 = top 20%)
    pub goss_top_rate: f32,
    /// Ratio of small-gradient samples to randomly sample (default: 0.1 = 10%)
    pub goss_other_rate: f32,

    // Binning
    /// Number of histogram bins
    pub num_bins: usize,

    // Conformal prediction
    /// Calibration set ratio for conformal prediction (0.0 to disable)
    pub calibration_ratio: f32,
    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    pub conformal_quantile: f32,

    // Early stopping
    /// Number of rounds with no improvement before stopping (0 to disable)
    pub early_stopping_rounds: usize,
    /// Ratio of data to use for validation (0.0 to disable early stopping)
    pub validation_ratio: f32,

    // Performance optimizations (all ON by default)
    /// Use parallel prediction via Rayon (default: true)
    pub parallel_prediction: bool,
    /// Reorder columns by feature importance for cache locality (default: true)
    pub column_reordering: bool,
    /// Column reordering strategy (default: ByImportance)
    pub reordering_strategy: OrderingStrategy,
    /// Use 4-bit packing for low-cardinality features (default: true)
    pub packed_dataset: bool,
    /// Use parallel gradient computation (default: false)
    /// Experimental: may not provide stable speedups, benchmark before enabling
    pub parallel_gradient: bool,

    // Backend selection
    /// Backend type for histogram building (default: Auto = GPU for large datasets, CPU otherwise)
    #[rkyv(with = rkyv::with::Skip)]
    pub backend_type: BackendType,

    // Monotonic constraints
    /// Monotonic constraints per feature (empty = no constraints)
    pub monotonic_constraints: Vec<MonotonicConstraint>,

    // Interaction constraints (groups of features that can interact)
    /// Feature interaction groups: each inner Vec is a group of features that can interact
    /// Features not in any group can interact with all features
    pub interaction_groups: Vec<Vec<usize>>,
}

impl Default for GBDTConfig {
    fn default() -> Self {
        Self {
            // Ensemble
            num_rounds: 100,
            learning_rate: 0.1,

            // Tree
            max_depth: 6,
            max_leaves: 31,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            min_gain: 0.0,

            // Regularization
            lambda: 1.0,
            entropy_weight: 0.0,

            // Loss
            loss_type: LossType::Mse,

            // Subsampling
            subsample: 1.0,
            colsample: 1.0,

            // GOSS (disabled by default, most effective on large datasets)
            goss_enabled: false,
            goss_top_rate: 0.2,
            goss_other_rate: 0.1,

            // Binning
            num_bins: 255,

            // Conformal
            calibration_ratio: 0.0,
            conformal_quantile: 0.9,

            // Early stopping (disabled by default)
            early_stopping_rounds: 0,
            validation_ratio: 0.0,

            // Performance optimizations (all ON by default)
            parallel_prediction: true,
            column_reordering: true,
            reordering_strategy: OrderingStrategy::ByImportance,
            packed_dataset: true,
            parallel_gradient: false, // Enable for large datasets (100k+ rows)

            // Backend selection (Auto = GPU for large datasets, CPU otherwise)
            backend_type: BackendType::Auto,

            // Monotonic constraints
            monotonic_constraints: Vec::new(),

            // Interaction constraints
            interaction_groups: Vec::new(),
        }
    }
}

impl GBDTConfig {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of boosting rounds
    pub fn with_num_rounds(mut self, num_rounds: usize) -> Self {
        self.num_rounds = num_rounds;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum tree depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set maximum leaves per tree
    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Set L2 regularization
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set Shannon Entropy regularization weight
    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set loss function to MSE
    pub fn with_mse_loss(mut self) -> Self {
        self.loss_type = LossType::Mse;
        self
    }

    /// Set loss function to Pseudo-Huber
    pub fn with_pseudo_huber_loss(mut self, delta: f32) -> Self {
        self.loss_type = LossType::PseudoHuber { delta };
        self
    }

    /// Set row subsampling ratio
    pub fn with_subsample(mut self, ratio: f32) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0);
        self.subsample = ratio;
        self
    }

    /// Set column subsampling ratio
    pub fn with_colsample(mut self, ratio: f32) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0);
        self.colsample = ratio;
        self
    }

    /// Enable conformal prediction
    pub fn with_conformal(mut self, calibration_ratio: f32, quantile: f32) -> Self {
        assert!(calibration_ratio >= 0.0 && calibration_ratio < 1.0);
        assert!(quantile > 0.0 && quantile < 1.0);
        self.calibration_ratio = calibration_ratio;
        self.conformal_quantile = quantile;
        self
    }

    /// Enable early stopping
    ///
    /// # Arguments
    /// * `rounds` - Number of consecutive rounds without improvement before stopping
    /// * `validation_ratio` - Fraction of data to use for validation (e.g., 0.1 for 10%)
    pub fn with_early_stopping(mut self, rounds: usize, validation_ratio: f32) -> Self {
        assert!(rounds > 0, "early_stopping_rounds must be > 0");
        assert!(validation_ratio > 0.0 && validation_ratio < 1.0);
        self.early_stopping_rounds = rounds;
        self.validation_ratio = validation_ratio;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set minimum hessian per leaf
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Enable/disable parallel prediction (default: enabled)
    pub fn with_parallel_prediction(mut self, enabled: bool) -> Self {
        self.parallel_prediction = enabled;
        self
    }

    /// Enable/disable column reordering for cache locality (default: enabled)
    pub fn with_column_reordering(mut self, enabled: bool) -> Self {
        self.column_reordering = enabled;
        self
    }

    /// Set column reordering strategy
    pub fn with_reordering_strategy(mut self, strategy: OrderingStrategy) -> Self {
        self.reordering_strategy = strategy;
        self
    }

    /// Enable/disable 4-bit packed dataset for memory optimization (default: enabled)
    pub fn with_packed_dataset(mut self, enabled: bool) -> Self {
        self.packed_dataset = enabled;
        self
    }

    /// Enable parallel gradient computation (default: false)
    ///
    /// Experimental: may not provide stable speedups, benchmark before enabling.
    /// Use examples/find_crossover.rs to test on your specific data.
    pub fn with_parallel_gradient(mut self, enabled: bool) -> Self {
        self.parallel_gradient = enabled;
        self
    }

    /// Set the backend for histogram building
    ///
    /// # Backend Types
    /// - `Auto` (default): Uses GPU for datasets >= 50K rows, CPU otherwise
    /// - `Scalar`: Force CPU (AVX2/NEON optimized)
    /// - `Wgpu`: Force GPU (requires `gpu` feature)
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::{GBDTConfig, BackendType};
    ///
    /// // Force GPU backend for training
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Wgpu);
    ///
    /// // Force CPU backend for reproducibility
    /// let config = GBDTConfig::new()
    ///     .with_backend(BackendType::Scalar);
    /// ```
    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = backend_type;
        self
    }

    /// Disable all performance optimizations (for debugging/comparison)
    pub fn without_optimizations(mut self) -> Self {
        self.parallel_prediction = false;
        self.column_reordering = false;
        self.packed_dataset = false;
        self.parallel_gradient = false;
        self
    }

    /// Enable/disable GOSS (Gradient-based One-Side Sampling)
    pub fn with_goss(mut self, enabled: bool) -> Self {
        self.goss_enabled = enabled;
        self
    }

    /// Configure GOSS sampling rates
    ///
    /// # Arguments
    /// * `top_rate` - Ratio of large-gradient samples to keep (default: 0.2)
    /// * `other_rate` - Ratio of small-gradient samples to randomly sample (default: 0.1)
    pub fn with_goss_rates(mut self, top_rate: f32, other_rate: f32) -> Self {
        self.goss_enabled = true;
        self.goss_top_rate = top_rate;
        self.goss_other_rate = other_rate;
        self
    }

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::{GBDTConfig, MonotonicConstraint};
    ///
    /// // Feature 0: increasing, Feature 1: decreasing, Feature 2: none
    /// let config = GBDTConfig::new()
    ///     .with_monotonic_constraints(vec![
    ///         MonotonicConstraint::Increasing,
    ///         MonotonicConstraint::Decreasing,
    ///         MonotonicConstraint::None,
    ///     ]);
    /// ```
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    /// Features not in any group can interact with all features.
    ///
    /// # Example
    /// ```ignore
    /// use treeboost::GBDTConfig;
    ///
    /// // Features 0,1,2 can interact; features 3,4 can interact
    /// // Feature 5 is unconstrained (can interact with any)
    /// let config = GBDTConfig::new()
    ///     .with_interaction_groups(vec![vec![0, 1, 2], vec![3, 4]]);
    /// ```
    pub fn with_interaction_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.interaction_groups = groups;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_rounds == 0 {
            return Err("num_rounds must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.max_depth == 0 {
            return Err("max_depth must be > 0".to_string());
        }
        if self.max_leaves == 0 {
            return Err("max_leaves must be > 0".to_string());
        }
        if self.lambda < 0.0 {
            return Err("lambda must be >= 0".to_string());
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err("subsample must be in (0, 1]".to_string());
        }
        if self.colsample <= 0.0 || self.colsample > 1.0 {
            return Err("colsample must be in (0, 1]".to_string());
        }
        if self.goss_enabled {
            if self.goss_top_rate <= 0.0 || self.goss_top_rate >= 1.0 {
                return Err("goss_top_rate must be in (0, 1)".to_string());
            }
            if self.goss_other_rate <= 0.0 || self.goss_other_rate >= 1.0 {
                return Err("goss_other_rate must be in (0, 1)".to_string());
            }
            if self.goss_top_rate + self.goss_other_rate >= 1.0 {
                return Err("goss_top_rate + goss_other_rate must be < 1.0".to_string());
            }
        }
        if self.validation_ratio < 0.0 || self.validation_ratio >= 1.0 {
            return Err("validation_ratio must be in [0, 1)".to_string());
        }
        // Can't use both conformal calibration and early stopping validation from same data
        if self.calibration_ratio > 0.0 && self.validation_ratio > 0.0 {
            let total_holdout = self.calibration_ratio + self.validation_ratio;
            if total_holdout >= 1.0 {
                return Err("calibration_ratio + validation_ratio must be < 1.0".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GBDTConfig::default();

        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.max_depth, 6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = GBDTConfig::new()
            .with_num_rounds(50)
            .with_learning_rate(0.05)
            .with_max_depth(4)
            .with_pseudo_huber_loss(1.0)
            .with_entropy_weight(0.1)
            .with_conformal(0.1, 0.9);

        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.max_depth, 4);
        assert_eq!(config.loss_type, LossType::PseudoHuber { delta: 1.0 });
        assert_eq!(config.entropy_weight, 0.1);
        assert_eq!(config.calibration_ratio, 0.1);
    }

    #[test]
    fn test_config_validation() {
        let invalid = GBDTConfig::default().with_num_rounds(0);
        assert!(invalid.validate().is_err());

        let invalid = GBDTConfig {
            subsample: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
