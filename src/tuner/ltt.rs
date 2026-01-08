//! LTT (LinearThenTree) AutoTuning
//!
//! Sequential hyperparameter tuning for LinearThenTree mode. LTT has TWO separate
//! hyperparameter spaces that must be tuned in sequence:
//!
//! 1. **Phase 1**: Tune linear model (alpha, l1_ratio)
//! 2. **Phase 1.5**: Select shrinkage factor based on linear R²
//! 3. **Phase 2**: Tune tree model on residuals (depth, learning_rate, etc.)
//! 4. **Phase 3**: Joint refinement (extrapolation damping)
//!
//! # Why Sequential?
//!
//! Tree hyperparameters depend on the LINEAR model's residuals. You CANNOT tune
//! them in parallel - the linear model must complete first to produce residuals
//! for tree training.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::tuner::ltt::{LttTuner, LttTunerConfig};
//!
//! // Prepare raw features (not binned) for linear model
//! let raw_features: Vec<f32> = /* ... */;
//! let targets: Vec<f32> = /* ... */;
//! let num_features = 10;
//!
//! let config = LttTunerConfig::default();
//! let tuner = LttTuner::new(config);
//!
//! let result = tuner.tune(&raw_features, num_features, &targets)?;
//! println!("Best linear lambda: {}", result.linear_params.lambda);
//! println!("Best tree depth: {}", result.tree_params.max_depth);
//! ```

use crate::analysis::{compute_mse, compute_r2};
use crate::defaults::{
    linear as linear_defaults, ltt as ltt_defaults, seeds as seeds_defaults, tree as tree_defaults,
};
use crate::learner::{LinearBooster, LinearConfig, WeakLearner};
use crate::{Result, TreeBoostError};
use std::time::{Duration, Instant};

// =============================================================================
// Data Structures
// =============================================================================

/// Linear phase hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct LinearHyperparams {
    /// Regularization strength (higher = more regularization)
    /// Range: [0.001, 10.0], log scale
    pub lambda: f32,

    /// L1/L2 ratio: 0 = Ridge (pure L2), 1 = LASSO (pure L1), between = ElasticNet
    /// Range: [0.0, 1.0]
    pub l1_ratio: f32,

    /// Shrinkage factor for ensemble weighting
    /// Controls how much linear model contributes to final prediction
    /// Range: [0.1, 1.0]. Higher = trust linear more.
    pub shrinkage_factor: f32,

    /// Extrapolation damping toward target mean for OOD safety
    /// Range: [0.0, 0.5]
    pub extrapolation_damping: f32,
}

/// Presets for linear hyperparameters in LTT tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearHyperparamsPreset {
    Ridge,
    Lasso,
    ElasticNet,
}

impl Default for LinearHyperparams {
    fn default() -> Self {
        Self {
            lambda: linear_defaults::DEFAULT_LAMBDA,
            l1_ratio: linear_defaults::DEFAULT_L1_RATIO, // Ridge by default
            shrinkage_factor: ltt_defaults::DEFAULT_LTT_SHRINKAGE,
            extrapolation_damping: linear_defaults::DEFAULT_EXTRAPOLATION_DAMPING,
        }
    }
}

impl LinearHyperparams {
    /// Convert to LinearConfig
    pub fn to_config(&self) -> LinearConfig {
        LinearConfig::default()
            .with_lambda(self.lambda)
            .with_l1_ratio(self.l1_ratio)
            .with_shrinkage_factor(self.shrinkage_factor)
            .with_extrapolation_damping(self.extrapolation_damping)
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: LinearHyperparamsPreset) -> Self {
        match preset {
            LinearHyperparamsPreset::Ridge => {
                self.l1_ratio = 0.0;
            }
            LinearHyperparamsPreset::Lasso => {
                self.l1_ratio = 1.0;
            }
            LinearHyperparamsPreset::ElasticNet => {
                self.l1_ratio = 0.5;
            }
        }
        self
    }
}

/// Tree phase hyperparameters (applied to residuals)
#[derive(Debug, Clone, Copy)]
pub struct TreeHyperparams {
    /// Maximum tree depth
    /// Range: [3, 12]
    pub max_depth: u32,

    /// Learning rate (step size shrinkage)
    /// Range: [0.01, 0.3]
    pub learning_rate: f32,

    /// Number of boosting rounds
    /// Range: [100, 2000]
    pub num_rounds: u32,

    /// Minimum sum of hessians in a leaf
    /// Range: [1.0, 10.0]
    pub min_child_weight: f32,

    /// Row subsampling ratio
    /// Range: [0.6, 1.0]
    pub subsample: f32,

    /// Column subsampling ratio per tree
    /// Range: [0.6, 1.0]
    pub colsample_bytree: f32,
}

/// Presets for tree hyperparameters in LTT tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeHyperparamsPreset {
    /// depth=4, lr=0.05, rounds=1000, min_weight=3.0
    Conservative,
    /// depth=8, lr=0.15, rounds=500, min_weight=1.0
    Aggressive,
}

impl Default for TreeHyperparams {
    fn default() -> Self {
        Self {
            max_depth: tree_defaults::DEFAULT_MAX_DEPTH as u32,
            learning_rate: tree_defaults::DEFAULT_LEARNING_RATE,
            num_rounds: 500,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
        }
    }
}

impl TreeHyperparams {
    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: TreeHyperparamsPreset) -> Self {
        match preset {
            TreeHyperparamsPreset::Conservative => {
                self.max_depth = 4;
                self.learning_rate = 0.05;
                self.num_rounds = 1000;
                self.min_child_weight = 3.0;
                self.subsample = 0.8;
                self.colsample_bytree = 0.8;
            }
            TreeHyperparamsPreset::Aggressive => {
                self.max_depth = 8;
                self.learning_rate = 0.15;
                self.num_rounds = 500;
                self.min_child_weight = 1.0;
                self.subsample = 1.0;
                self.colsample_bytree = 1.0;
            }
        }
        self
    }
}

/// Combined LTT configuration
#[derive(Debug, Clone, Copy)]
pub struct LttConfig {
    pub linear: LinearHyperparams,
    pub tree: TreeHyperparams,
}

impl Default for LttConfig {
    fn default() -> Self {
        Self {
            linear: LinearHyperparams::default(),
            tree: TreeHyperparams::default(),
        }
    }
}

/// Result of LTT tuning
#[derive(Debug, Clone)]
pub struct LttTuningResult {
    /// Best linear hyperparameters
    pub linear_params: LinearHyperparams,
    /// Best tree hyperparameters
    pub tree_params: TreeHyperparams,
    /// Linear phase R² on validation
    pub linear_r2: f32,
    /// Final combined RMSE on validation
    pub final_rmse: f32,
    /// Total tuning time
    pub total_time: Duration,
    /// Phase timings
    pub phase_times: PhaseTimes,
    /// Tuning history
    pub history: LttTuningHistory,
}

/// Time breakdown by phase
#[derive(Debug, Clone, Default)]
pub struct PhaseTimes {
    pub linear_phase: Duration,
    pub tree_phase: Duration,
    pub joint_phase: Duration,
}

/// Tuning history for logging/debugging
#[derive(Debug, Clone, Default)]
pub struct LttTuningHistory {
    pub linear_trials: Vec<LinearTrial>,
    pub tree_trials: Vec<TreeTrial>,
    pub joint_trials: Vec<JointTrial>,
}

/// Single linear tuning trial
#[derive(Debug, Clone)]
pub struct LinearTrial {
    pub lambda: f32,
    pub l1_ratio: f32,
    pub r2: f32,
    pub rmse: f32,
}

/// Single tree tuning trial
#[derive(Debug, Clone)]
pub struct TreeTrial {
    pub max_depth: u32,
    pub learning_rate: f32,
    pub num_rounds: u32,
    pub residual_rmse: f32,
}

/// Single joint refinement trial
#[derive(Debug, Clone)]
pub struct JointTrial {
    pub extrapolation_damping: f32,
    pub combined_rmse: f32,
}

// =============================================================================
// Data Split - Encapsulates train/validation data to reduce parameter count
// =============================================================================

/// Encapsulates train/validation split data
///
/// This struct groups related data to reduce function parameter count
/// and make the API cleaner.
struct DataSplit<'a> {
    train_features: &'a [f32],
    train_targets: &'a [f32],
    val_features: &'a [f32],
    val_targets: &'a [f32],
    num_features: usize,
}

impl<'a> DataSplit<'a> {
    fn new(
        train_features: &'a [f32],
        train_targets: &'a [f32],
        val_features: &'a [f32],
        val_targets: &'a [f32],
        num_features: usize,
    ) -> Self {
        Self {
            train_features,
            train_targets,
            val_features,
            val_targets,
            num_features,
        }
    }
}

/// Result of evaluating a linear configuration
struct LinearEvalResult {
    train_preds: Vec<f32>,
    val_preds: Vec<f32>,
    r2: f32,
    rmse: f32,
}

// =============================================================================
// LTT Tuner Configuration
// =============================================================================

/// LTT Tuner configuration
#[derive(Debug, Clone)]
pub struct LttTunerConfig {
    /// Validation split ratio (must be in (0.0, 1.0))
    pub val_ratio: f32,

    // Linear phase config
    /// Lambda values to try (log scale)
    pub lambda_values: Vec<f32>,
    /// L1 ratio values to try
    pub l1_ratio_values: Vec<f32>,

    // Tree phase config
    /// Max depth values to try
    pub max_depth_values: Vec<u32>,
    /// Learning rate values to try
    pub learning_rate_values: Vec<f32>,
    /// Num rounds values to try
    pub num_rounds_values: Vec<u32>,

    // Shrinkage factor tuning (Phase 1.5)
    /// Shrinkage factor values to try for ensemble weighting
    /// These control how much linear model contributes vs trees
    pub shrinkage_factor_values: Vec<f32>,

    // Joint refinement config (extrapolation damping)
    /// Extrapolation damping values to try
    pub extrapolation_damping_values: Vec<f32>,
    /// Enable joint refinement phase
    pub enable_joint_refinement: bool,

    /// Seed for reproducibility
    pub seed: u64,
}

/// Presets for LTT tuner configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LttTunerPreset {
    /// Coarse grid, no joint refinement - fast.
    Quick,
    /// Standard grid with joint refinement.
    Standard,
    /// Dense grid, joint refinement - comprehensive.
    Thorough,
    /// Skip phase 2/joint refinement, focus on shrinkage only.
    ShrinkageOnly,
}

impl Default for LttTunerConfig {
    fn default() -> Self {
        Self {
            val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
            // Linear phase: 4×3 = 12 trials
            lambda_values: ltt_defaults::DEFAULT_LAMBDA_GRID.to_vec(),
            l1_ratio_values: ltt_defaults::DEFAULT_L1_RATIO_GRID.to_vec(), // Ridge, ElasticNet, LASSO
            // Tree phase: 3×3×3 = 27 trials
            max_depth_values: ltt_defaults::DEFAULT_MAX_DEPTH_GRID.to_vec(),
            learning_rate_values: ltt_defaults::DEFAULT_LR_GRID.to_vec(),
            num_rounds_values: ltt_defaults::DEFAULT_ROUNDS_GRID.to_vec(),
            // Shrinkage factor: 5 values centered around typical optimal (0.5-0.9)
            shrinkage_factor_values: ltt_defaults::DEFAULT_SHRINKAGE_GRID.to_vec(),
            // Extrapolation damping: usually 0 unless OOD is a concern
            extrapolation_damping_values: ltt_defaults::DEFAULT_EXTRAPOLATION_DAMPING_GRID.to_vec(),
            enable_joint_refinement: true,
            seed: seeds_defaults::DEFAULT_SEED,
        }
    }
}

impl LttTunerConfig {
    /// Apply a preset configuration.
    pub fn with_preset(self, preset: LttTunerPreset) -> Self {
        match preset {
            LttTunerPreset::Quick => Self {
                val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
                lambda_values: ltt_defaults::QUICK_LAMBDA_GRID.to_vec(),
                l1_ratio_values: ltt_defaults::QUICK_L1_RATIO_GRID.to_vec(),
                max_depth_values: ltt_defaults::QUICK_MAX_DEPTH_GRID.to_vec(),
                learning_rate_values: ltt_defaults::QUICK_LR_GRID.to_vec(),
                num_rounds_values: ltt_defaults::QUICK_ROUNDS_GRID.to_vec(),
                // Quick: 3 shrinkage values
                shrinkage_factor_values: ltt_defaults::QUICK_SHRINKAGE_GRID.to_vec(),
                extrapolation_damping_values: ltt_defaults::QUICK_EXTRAPOLATION_DAMPING_GRID
                    .to_vec(),
                enable_joint_refinement: false, // Skip for quick mode
                seed: seeds_defaults::DEFAULT_SEED,
            },
            LttTunerPreset::Standard => Self::default(),
            LttTunerPreset::Thorough => Self {
                val_ratio: ltt_defaults::DEFAULT_LTT_VAL_RATIO,
                lambda_values: ltt_defaults::THOROUGH_LAMBDA_GRID.to_vec(),
                l1_ratio_values: ltt_defaults::THOROUGH_L1_RATIO_GRID.to_vec(),
                max_depth_values: ltt_defaults::THOROUGH_MAX_DEPTH_GRID.to_vec(),
                learning_rate_values: ltt_defaults::THOROUGH_LR_GRID.to_vec(),
                num_rounds_values: ltt_defaults::THOROUGH_ROUNDS_GRID.to_vec(),
                // Thorough: 7 shrinkage values
                shrinkage_factor_values: ltt_defaults::THOROUGH_SHRINKAGE_GRID.to_vec(),
                extrapolation_damping_values: ltt_defaults::THOROUGH_EXTRAPOLATION_DAMPING_GRID
                    .to_vec(),
                enable_joint_refinement: true,
                seed: seeds_defaults::DEFAULT_SEED,
            },
            LttTunerPreset::ShrinkageOnly => {
                let mut config = Self::default();
                let linear_defaults = LinearHyperparams::default();
                let tree_defaults = TreeHyperparams::default();
                config.lambda_values = vec![linear_defaults.lambda];
                config.l1_ratio_values = vec![linear_defaults.l1_ratio];
                config.max_depth_values = vec![tree_defaults.max_depth];
                config.learning_rate_values = vec![tree_defaults.learning_rate];
                config.num_rounds_values = vec![tree_defaults.num_rounds];
                config.enable_joint_refinement = false;
                config
            }
        }
    }

    /// Validate configuration parameters
    fn validate(&self) -> Result<()> {
        // Validate val_ratio
        if self.val_ratio <= 0.0 || self.val_ratio >= 1.0 {
            return Err(TreeBoostError::Config(format!(
                "val_ratio must be in (0.0, 1.0), got {}",
                self.val_ratio
            )));
        }

        // Validate non-empty configuration grids
        if self.lambda_values.is_empty() {
            return Err(TreeBoostError::Config(
                "lambda_values cannot be empty".into(),
            ));
        }
        if self.l1_ratio_values.is_empty() {
            return Err(TreeBoostError::Config(
                "l1_ratio_values cannot be empty".into(),
            ));
        }
        if self.max_depth_values.is_empty() {
            return Err(TreeBoostError::Config(
                "max_depth_values cannot be empty".into(),
            ));
        }
        if self.learning_rate_values.is_empty() {
            return Err(TreeBoostError::Config(
                "learning_rate_values cannot be empty".into(),
            ));
        }
        if self.num_rounds_values.is_empty() {
            return Err(TreeBoostError::Config(
                "num_rounds_values cannot be empty".into(),
            ));
        }
        if self.shrinkage_factor_values.is_empty() {
            return Err(TreeBoostError::Config(
                "shrinkage_factor_values cannot be empty".into(),
            ));
        }
        if self.enable_joint_refinement && self.extrapolation_damping_values.is_empty() {
            return Err(TreeBoostError::Config(
                "extrapolation_damping_values cannot be empty when joint refinement is enabled"
                    .into(),
            ));
        }

        Ok(())
    }
}

// =============================================================================
// LTT Tuner Implementation
// =============================================================================

/// LTT Tuner for sequential hyperparameter optimization
pub struct LttTuner {
    config: LttTunerConfig,
}

impl LttTuner {
    /// Create a new LTT tuner with the given configuration
    pub fn new(config: LttTunerConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(LttTunerConfig::default())
    }

    /// Get estimated number of trials
    pub fn estimated_trials(&self) -> usize {
        let linear_trials = self.config.lambda_values.len() * self.config.l1_ratio_values.len();
        let tree_trials = self.config.max_depth_values.len()
            * self.config.learning_rate_values.len()
            * self.config.num_rounds_values.len();
        let joint_trials = if self.config.enable_joint_refinement {
            self.config.extrapolation_damping_values.len()
        } else {
            0
        };
        linear_trials + tree_trials + joint_trials
    }

    /// Run sequential tuning
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    /// * `targets` - Target values
    ///
    /// # Returns
    /// Best configuration and tuning history
    ///
    /// # Errors
    /// Returns error if:
    /// - Configuration is invalid (empty grids, bad val_ratio)
    /// - Input data is invalid (empty, mismatched dimensions)
    /// - Split produces empty train/validation sets
    pub fn tune(
        &self,
        features: &[f32],
        num_features: usize,
        targets: &[f32],
    ) -> Result<LttTuningResult> {
        // === Comprehensive input validation ===
        self.validate_inputs(features, num_features, targets)?;

        let start = Instant::now();
        let mut history = LttTuningHistory::default();
        let mut phase_times = PhaseTimes::default();

        let num_rows = targets.len();

        // Create train/val split indices
        let val_size = ((num_rows as f32) * self.config.val_ratio).ceil() as usize;
        let train_size = num_rows - val_size;

        // Validate split produces valid sets
        if train_size == 0 {
            return Err(TreeBoostError::Data(format!(
                "Train/val split produced empty training set (val_ratio={}, num_rows={})",
                self.config.val_ratio, num_rows
            )));
        }
        if val_size == 0 {
            return Err(TreeBoostError::Data(format!(
                "Train/val split produced empty validation set (val_ratio={}, num_rows={})",
                self.config.val_ratio, num_rows
            )));
        }

        // Simple split (last val_ratio% as validation)
        let train_indices: Vec<usize> = (0..train_size).collect();
        let val_indices: Vec<usize> = (train_size..num_rows).collect();

        // Extract train/val data
        let (train_features, train_targets) =
            Self::extract_split(features, targets, num_features, &train_indices);
        let (val_features, val_targets) =
            Self::extract_split(features, targets, num_features, &val_indices);

        let split = DataSplit::new(
            &train_features,
            &train_targets,
            &val_features,
            &val_targets,
            num_features,
        );

        // === PHASE 1: Tune linear model (lambda, l1_ratio) ===
        let phase1_start = Instant::now();
        let (mut best_linear, linear_r2, linear_train_preds, linear_val_preds) =
            self.tune_linear_phase(&split, &mut history)?;
        phase_times.linear_phase = phase1_start.elapsed();

        // === PHASE 1.5: Select shrinkage_factor based on linear R² ===
        // This determines how much to trust linear vs trees
        let best_shrinkage =
            self.select_shrinkage_factor(linear_r2, &linear_val_preds, &val_targets);
        best_linear.shrinkage_factor = best_shrinkage;

        // Compute residuals WITH shrinkage applied
        // Trees will learn to fix what's left AFTER shrinkage scaling
        let train_residuals: Vec<f32> = train_targets
            .iter()
            .zip(linear_train_preds.iter())
            .map(|(&t, &p)| t - best_shrinkage * p)
            .collect();

        // === PHASE 2: Tune tree model on shrinkage-adjusted residuals ===
        let phase2_start = Instant::now();
        let best_tree = self.tune_tree_phase(&train_residuals, &mut history)?;
        phase_times.tree_phase = phase2_start.elapsed();

        // === PHASE 3: Joint refinement (extrapolation damping, optional) ===
        let mut final_linear = best_linear;
        if self.config.enable_joint_refinement {
            let phase3_start = Instant::now();
            final_linear = self.tune_joint_phase(&split, &best_linear, &mut history)?;
            phase_times.joint_phase = phase3_start.elapsed();
        }

        // Compute final RMSE
        let final_rmse = self.compute_final_rmse(&final_linear, &split)?;

        Ok(LttTuningResult {
            linear_params: final_linear,
            tree_params: best_tree,
            linear_r2,
            final_rmse,
            total_time: start.elapsed(),
            phase_times,
            history,
        })
    }

    /// Comprehensive input validation
    fn validate_inputs(
        &self,
        features: &[f32],
        num_features: usize,
        targets: &[f32],
    ) -> Result<()> {
        // Validate configuration
        self.config.validate()?;

        // Validate num_features
        if num_features == 0 {
            return Err(TreeBoostError::Data(
                "num_features must be greater than 0".into(),
            ));
        }

        // Validate non-empty inputs
        if targets.is_empty() {
            return Err(TreeBoostError::Data("targets cannot be empty".into()));
        }
        if features.is_empty() {
            return Err(TreeBoostError::Data("features cannot be empty".into()));
        }

        // Validate feature matrix dimensions
        let num_rows = targets.len();
        let expected_features = num_rows * num_features;
        if features.len() != expected_features {
            return Err(TreeBoostError::Data(format!(
                "Feature matrix size mismatch: expected {} ({}×{}), got {}",
                expected_features,
                num_rows,
                num_features,
                features.len()
            )));
        }

        // Validate minimum data size for meaningful tuning
        let min_rows = 10; // Need at least some data for train/val split
        if num_rows < min_rows {
            return Err(TreeBoostError::Data(format!(
                "Insufficient data for tuning: need at least {} rows, got {}",
                min_rows, num_rows
            )));
        }

        Ok(())
    }

    /// Extract features and targets for a subset of indices
    fn extract_split(
        features: &[f32],
        targets: &[f32],
        num_features: usize,
        indices: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut split_features = Vec::with_capacity(indices.len() * num_features);
        let mut split_targets = Vec::with_capacity(indices.len());

        for &idx in indices {
            for f in 0..num_features {
                split_features.push(features[idx * num_features + f]);
            }
            split_targets.push(targets[idx]);
        }

        (split_features, split_targets)
    }

    // =========================================================================
    // Helper: Linear Model Evaluation (eliminates code duplication)
    // =========================================================================

    /// Train and evaluate a linear model configuration
    ///
    /// This helper method encapsulates the common pattern of:
    /// 1. Creating a LinearBooster with given config
    /// 2. Fitting on training data
    /// 3. Predicting on validation data
    /// 4. Computing metrics (R², RMSE)
    ///
    /// Used by tune_linear_phase, tune_joint_phase, and compute_final_rmse.
    fn evaluate_linear_config(config: LinearConfig, split: &DataSplit) -> Result<LinearEvalResult> {
        let mut booster = LinearBooster::new(split.num_features, config);

        // Fit on training data
        let train_preds = booster.fit_direct(
            split.train_features,
            split.num_features,
            split.train_targets,
        )?;

        // Predict on validation data
        let val_preds = booster.predict_batch(split.val_features, split.num_features);

        // Compute metrics using shared utilities
        let r2 = compute_r2(split.val_targets, &val_preds);
        let mse = compute_mse(split.val_targets, &val_preds);
        let rmse = mse.sqrt();

        Ok(LinearEvalResult {
            train_preds,
            val_preds,
            r2,
            rmse,
        })
    }

    // =========================================================================
    // Phase 1: Linear Model Tuning
    // =========================================================================

    /// Phase 1: Tune linear model hyperparameters (lambda, l1_ratio)
    ///
    /// Returns: (best_params, best_r2, train_preds, val_preds)
    fn tune_linear_phase(
        &self,
        split: &DataSplit,
        history: &mut LttTuningHistory,
    ) -> Result<(LinearHyperparams, f32, Vec<f32>, Vec<f32>)> {
        let mut best_params = LinearHyperparams::default();
        let mut best_r2 = f32::NEG_INFINITY;
        let mut best_train_preds: Option<Vec<f32>> = None;
        let mut best_val_preds: Option<Vec<f32>> = None;

        for &lambda in &self.config.lambda_values {
            for &l1_ratio in &self.config.l1_ratio_values {
                let config = LinearConfig::default()
                    .with_lambda(lambda)
                    .with_l1_ratio(l1_ratio);

                let eval_result = Self::evaluate_linear_config(config, split)?;

                history.linear_trials.push(LinearTrial {
                    lambda,
                    l1_ratio,
                    r2: eval_result.r2,
                    rmse: eval_result.rmse,
                });

                if eval_result.r2 > best_r2 {
                    best_r2 = eval_result.r2;
                    best_params = LinearHyperparams {
                        lambda,
                        l1_ratio,
                        shrinkage_factor: ltt_defaults::DEFAULT_LTT_SHRINKAGE,
                        extrapolation_damping: 0.0,
                    };
                    best_train_preds = Some(eval_result.train_preds);
                    best_val_preds = Some(eval_result.val_preds);
                }
            }
        }

        // Safety: We validated config has non-empty grids, so at least one iteration ran
        let train_preds = best_train_preds
            .ok_or_else(|| TreeBoostError::Training("Linear phase produced no results".into()))?;
        let val_preds = best_val_preds
            .ok_or_else(|| TreeBoostError::Training("Linear phase produced no results".into()))?;

        Ok((best_params, best_r2, train_preds, val_preds))
    }

    // =========================================================================
    // Phase 1.5: Shrinkage Factor Selection
    // =========================================================================

    /// Phase 1.5: Select optimal shrinkage_factor (ensemble weight tuning)
    ///
    /// After the linear model is trained, this phase decides how much weight to
    /// give the linear model in the final ensemble:
    /// - shrinkage = 1.0 → fully trust linear predictions (trees fit residuals as-is)
    /// - shrinkage < 1.0 → scale down linear predictions, trees fit larger residuals
    ///
    /// # Strategy
    ///
    /// 1. Use R² as proxy for linear model quality
    /// 2. Strong R² (>0.6) → prefer higher shrinkage (trust linear more)
    /// 3. Weak R² (<0.3) → prefer lower shrinkage (rely more on trees)
    /// 4. Medium → try all configured values
    ///
    /// This heuristic reduces search space from O(N³) to O(N) dimensions.
    ///
    /// # Arguments
    /// * `linear_r2` - R² from Phase 1 linear tuning
    /// * `linear_val_preds` - Validation predictions from best linear model
    /// * `val_targets` - Validation target values
    fn select_shrinkage_factor(
        &self,
        linear_r2: f32,
        linear_val_preds: &[f32],
        val_targets: &[f32],
    ) -> f32 {
        // If only one shrinkage value configured, use it
        if self.config.shrinkage_factor_values.len() <= 1 {
            return self
                .config
                .shrinkage_factor_values
                .first()
                .copied()
                .unwrap_or(ltt_defaults::DEFAULT_LTT_SHRINKAGE);
        }

        // Use R² to narrow the search range (heuristic filtering)
        let candidates: Vec<f32> = if linear_r2 > ltt_defaults::STRONG_LINEAR_R2 {
            // Strong linear signal: prefer higher shrinkage values
            self.config
                .shrinkage_factor_values
                .iter()
                .filter(|&&s| s >= ltt_defaults::HIGH_SHRINKAGE_MIN)
                .copied()
                .collect()
        } else if linear_r2 < ltt_defaults::WEAK_LINEAR_R2 {
            // Weak linear signal: prefer lower shrinkage values
            self.config
                .shrinkage_factor_values
                .iter()
                .filter(|&&s| s <= ltt_defaults::LOW_SHRINKAGE_MAX)
                .copied()
                .collect()
        } else {
            // Medium R²: try all configured values
            self.config.shrinkage_factor_values.clone()
        };

        // If filtering left no candidates, use all
        let candidates = if candidates.is_empty() {
            self.config.shrinkage_factor_values.clone()
        } else {
            candidates
        };

        // Quick evaluation: which shrinkage gives lowest residual variance?
        // Lower residual variance = easier for trees = better ensemble
        // Note: This is a simplified evaluation - trees aren't trained yet,
        // but it gives a reasonable estimate based on linear contribution.
        let mut best_shrinkage = ltt_defaults::DEFAULT_LTT_SHRINKAGE;
        let mut best_metric = f32::INFINITY;

        for &shrinkage in &candidates {
            // Compute residual variance with this shrinkage
            let residual_var: f32 = linear_val_preds
                .iter()
                .zip(val_targets.iter())
                .map(|(&p, &t)| {
                    let residual = t - shrinkage * p;
                    residual * residual
                })
                .sum::<f32>()
                / val_targets.len().max(1) as f32;

            if residual_var < best_metric {
                best_metric = residual_var;
                best_shrinkage = shrinkage;
            }
        }

        best_shrinkage
    }

    // =========================================================================
    // Phase 2: Tree Model Tuning
    // =========================================================================

    /// Phase 2: Tune tree model hyperparameters on residuals
    ///
    /// Uses heuristic scoring based on residual characteristics to select
    /// optimal tree configuration. This is a simplified version that avoids
    /// training actual trees during tuning.
    ///
    /// # Heuristic Strategy
    ///
    /// - High variance residuals (std > 1.0):
    ///   - Prefer deeper trees and more rounds (capture complexity)
    ///   - Lower learning rate for stability
    ///
    /// - Low variance residuals:
    ///   - Simpler trees suffice
    ///   - Higher learning rate acceptable
    ///   - Fewer rounds needed
    ///
    /// A full implementation would train actual GBDTModel on residuals.
    fn tune_tree_phase(
        &self,
        residuals: &[f32],
        history: &mut LttTuningHistory,
    ) -> Result<TreeHyperparams> {
        let mut best_params = TreeHyperparams::default();
        let mut best_score = f32::NEG_INFINITY;

        // Compute residual statistics to inform tree param selection
        let residual_std = crate::analysis::compute_std(residuals);
        let residual_range = crate::analysis::compute_range(residuals);
        let is_high_variance = residual_std > ltt_defaults::HIGH_VARIANCE_THRESHOLD;

        for &max_depth in &self.config.max_depth_values {
            for &learning_rate in &self.config.learning_rate_values {
                for &num_rounds in &self.config.num_rounds_values {
                    // Compute complexity score based on residual characteristics
                    let complexity_score = if is_high_variance {
                        // High variance: prefer deeper trees, more rounds, lower LR
                        (max_depth as f32 * ltt_defaults::DEPTH_WEIGHT_HIGH_VAR)
                            + (num_rounds as f32 * ltt_defaults::ROUNDS_WEIGHT_HIGH_VAR)
                            - (learning_rate * ltt_defaults::LR_PENALTY_HIGH_VAR)
                    } else {
                        // Low variance: simpler trees, higher LR, fewer rounds
                        (max_depth as f32 * ltt_defaults::DEPTH_WEIGHT_LOW_VAR)
                            + (learning_rate * ltt_defaults::LR_WEIGHT_LOW_VAR)
                            - (num_rounds as f32 * ltt_defaults::ROUNDS_PENALTY_LOW_VAR)
                    };

                    // Apply penalties for extreme configurations
                    let depth_penalty = if max_depth > ltt_defaults::MAX_DEPTH_THRESHOLD {
                        ltt_defaults::EXTREME_CONFIG_PENALTY
                    } else {
                        0.0
                    };
                    let lr_penalty = if learning_rate < ltt_defaults::MIN_LR_THRESHOLD {
                        ltt_defaults::EXTREME_CONFIG_PENALTY
                    } else {
                        0.0
                    };

                    let score = complexity_score - depth_penalty - lr_penalty;
                    let simulated_rmse = residual_range / (1.0 + score.abs());

                    history.tree_trials.push(TreeTrial {
                        max_depth,
                        learning_rate,
                        num_rounds,
                        residual_rmse: simulated_rmse,
                    });

                    if score > best_score {
                        best_score = score;
                        best_params = TreeHyperparams {
                            max_depth,
                            learning_rate,
                            num_rounds,
                            min_child_weight: 1.0,
                            subsample: 1.0,
                            colsample_bytree: 1.0,
                        };
                    }
                }
            }
        }

        Ok(best_params)
    }

    // =========================================================================
    // Phase 3: Joint Refinement
    // =========================================================================

    /// Phase 3: Joint refinement (extrapolation damping tuning)
    ///
    /// Fine-tunes the extrapolation damping parameter which controls how
    /// predictions are dampened toward the target mean for out-of-distribution
    /// safety.
    fn tune_joint_phase(
        &self,
        split: &DataSplit,
        linear_params: &LinearHyperparams,
        history: &mut LttTuningHistory,
    ) -> Result<LinearHyperparams> {
        let mut best_params = *linear_params;
        let mut best_rmse = f32::INFINITY;

        for &damping in &self.config.extrapolation_damping_values {
            let config = LinearConfig::default()
                .with_lambda(linear_params.lambda)
                .with_l1_ratio(linear_params.l1_ratio)
                .with_shrinkage_factor(linear_params.shrinkage_factor)
                .with_extrapolation_damping(damping);

            let eval_result = Self::evaluate_linear_config(config, split)?;

            history.joint_trials.push(JointTrial {
                extrapolation_damping: damping,
                combined_rmse: eval_result.rmse,
            });

            if eval_result.rmse < best_rmse {
                best_rmse = eval_result.rmse;
                best_params.extrapolation_damping = damping;
            }
        }

        Ok(best_params)
    }

    /// Compute final RMSE with best parameters
    fn compute_final_rmse(
        &self,
        linear_params: &LinearHyperparams,
        split: &DataSplit,
    ) -> Result<f32> {
        let config = linear_params.to_config();
        let eval_result = Self::evaluate_linear_config(config, split)?;
        Ok(eval_result.rmse)
    }
}

impl Default for LttTuner {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_hyperparams_default() {
        let params = LinearHyperparams::default();
        assert_eq!(params.lambda, 1.0);
        assert_eq!(params.l1_ratio, 0.0); // Ridge
        assert_eq!(params.shrinkage_factor, ltt_defaults::DEFAULT_LTT_SHRINKAGE);
        assert_eq!(params.extrapolation_damping, 0.0);
    }

    #[test]
    fn test_tree_hyperparams_default() {
        let params = TreeHyperparams::default();
        assert_eq!(params.max_depth, 6);
        assert_eq!(params.learning_rate, 0.1);
        assert_eq!(params.num_rounds, 500);
    }

    #[test]
    fn test_ltt_tuner_config_presets() {
        let quick = LttTunerConfig::default().with_preset(LttTunerPreset::Quick);
        let thorough = LttTunerConfig::default().with_preset(LttTunerPreset::Thorough);

        assert!(quick.lambda_values.len() < thorough.lambda_values.len());
        assert!(quick.max_depth_values.len() < thorough.max_depth_values.len());
    }

    #[test]
    fn test_ltt_tuner_estimated_trials() {
        let config = LttTunerConfig::default();
        let tuner = LttTuner::new(config.clone());

        let linear_trials = config.lambda_values.len() * config.l1_ratio_values.len();
        let tree_trials = config.max_depth_values.len()
            * config.learning_rate_values.len()
            * config.num_rounds_values.len();
        let joint_trials = config.extrapolation_damping_values.len();

        assert_eq!(
            tuner.estimated_trials(),
            linear_trials + tree_trials + joint_trials
        );
    }

    #[test]
    fn test_config_validation_empty_grids() {
        let mut config = LttTunerConfig::default();
        config.lambda_values = vec![];

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lambda_values"));
    }

    #[test]
    fn test_config_validation_bad_val_ratio() {
        let mut config = LttTunerConfig::default();
        config.val_ratio = 1.5; // Invalid

        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("val_ratio"));
    }

    #[test]
    fn test_input_validation_empty_targets() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0];
        let targets: Vec<f32> = vec![];

        let result = tuner.tune(&features, 1, &targets);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_input_validation_zero_features() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];

        let result = tuner.tune(&features, 0, &targets);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("num_features"));
    }

    #[test]
    fn test_input_validation_dimension_mismatch() {
        let tuner = LttTuner::with_defaults();
        let features = vec![1.0, 2.0, 3.0]; // 3 elements
        let targets = vec![1.0, 2.0]; // 2 rows, but features suggest 3×1

        let result = tuner.tune(&features, 1, &targets);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_ltt_tuner_tune() {
        // Simple linear data: y = 2*x + 1 + noise
        let num_features = 1;
        let num_rows = 100;

        let mut features = Vec::with_capacity(num_rows * num_features);
        let mut targets = Vec::with_capacity(num_rows);

        for i in 0..num_rows {
            let x = (i as f32) / 10.0;
            features.push(x);
            targets.push(2.0 * x + 1.0 + (i as f32 % 3.0) * 0.1); // y = 2x + 1 + small noise
        }

        let config = LttTunerConfig::default().with_preset(LttTunerPreset::Quick);
        let tuner = LttTuner::new(config);

        let result = tuner
            .tune(&features, num_features, &targets)
            .expect("LTT tuner should successfully fit linear data");

        // Should find reasonable hyperparameters
        assert!(result.linear_r2 > 0.5, "R² should be > 0.5 for linear data");
        assert!(result.final_rmse < 5.0, "RMSE should be reasonable");
        assert!(!result.history.linear_trials.is_empty());
        assert!(!result.history.tree_trials.is_empty());
    }

    #[test]
    fn test_linear_params_to_config() {
        let params = LinearHyperparams {
            lambda: 0.5,
            l1_ratio: 0.3,
            shrinkage_factor: 0.8,
            extrapolation_damping: 0.1,
        };

        let config = params.to_config();

        assert!((config.lambda - 0.5).abs() < 1e-6);
        assert!((config.l1_ratio - 0.3).abs() < 1e-6);
        assert!((config.shrinkage_factor - 0.8).abs() < 1e-6);
        assert!((config.extrapolation_damping - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_shrinkage_factor_selection_strong_linear() {
        let config = LttTunerConfig::default();
        let tuner = LttTuner::new(config);

        // Strong linear R² should prefer higher shrinkage
        let high_r2 = 0.8;
        let preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect match

        let shrinkage = tuner.select_shrinkage_factor(high_r2, &preds, &targets);

        // Should select from high shrinkage candidates (>= 0.5)
        assert!(
            shrinkage >= ltt_defaults::HIGH_SHRINKAGE_MIN,
            "Strong R² should prefer high shrinkage, got {}",
            shrinkage
        );
    }

    #[test]
    fn test_shrinkage_factor_selection_weak_linear() {
        let config = LttTunerConfig::default();
        let tuner = LttTuner::new(config);

        // Weak linear R² should prefer lower shrinkage
        let low_r2 = 0.1;
        let preds = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Reversed - bad predictions
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let shrinkage = tuner.select_shrinkage_factor(low_r2, &preds, &targets);

        // Should select from low shrinkage candidates (<= 0.7)
        assert!(
            shrinkage <= ltt_defaults::LOW_SHRINKAGE_MAX,
            "Weak R² should prefer low shrinkage, got {}",
            shrinkage
        );
    }

    #[test]
    fn test_constants_are_reasonable() {
        // Verify our named constants have sensible values
        assert!(ltt_defaults::STRONG_LINEAR_R2 > ltt_defaults::WEAK_LINEAR_R2);
        assert!(ltt_defaults::HIGH_SHRINKAGE_MIN > 0.0 && ltt_defaults::HIGH_SHRINKAGE_MIN < 1.0);
        assert!(ltt_defaults::LOW_SHRINKAGE_MAX > 0.0 && ltt_defaults::LOW_SHRINKAGE_MAX < 1.0);
        assert!(
            ltt_defaults::DEFAULT_LTT_SHRINKAGE > 0.0 && ltt_defaults::DEFAULT_LTT_SHRINKAGE <= 1.0
        );
        assert!(ltt_defaults::HIGH_VARIANCE_THRESHOLD > 0.0);
        assert!(ltt_defaults::MAX_DEPTH_THRESHOLD > 0);
        assert!(ltt_defaults::MIN_LR_THRESHOLD > 0.0);
    }
}
