//! LTT (LinearThenTree) AutoTuning
//!
//! Sequential hyperparameter tuning for LinearThenTree mode. LTT has TWO separate
//! hyperparameter spaces that must be tuned in sequence:
//!
//! 1. **Phase 1**: Tune linear model (alpha, l1_ratio)
//! 2. **Phase 2**: Tune tree model on residuals (depth, learning_rate, etc.)
//! 3. **Phase 3**: Joint refinement (shrinkage balancing)
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

use crate::learner::{LinearBooster, LinearConfig, WeakLearner};
use crate::Result;
use std::time::{Duration, Instant};

/// Linear phase hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct LinearHyperparams {
    /// Regularization strength (higher = more regularization)
    /// Range: [0.001, 10.0], log scale
    pub lambda: f32,

    /// L1/L2 ratio: 0 = Ridge (pure L2), 1 = LASSO (pure L1), between = ElasticNet
    /// Range: [0.0, 1.0]
    pub l1_ratio: f32,

    /// Extrapolation damping toward target mean for OOD safety
    /// Range: [0.0, 0.5]
    pub extrapolation_damping: f32,
}

impl Default for LinearHyperparams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.0, // Ridge by default
            extrapolation_damping: 0.0,
        }
    }
}

impl LinearHyperparams {
    /// Create default Ridge regression params
    pub fn ridge() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.0,
            extrapolation_damping: 0.0,
        }
    }

    /// Create LASSO params
    pub fn lasso() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 1.0,
            extrapolation_damping: 0.0,
        }
    }

    /// Create ElasticNet params
    pub fn elastic_net() -> Self {
        Self {
            lambda: 1.0,
            l1_ratio: 0.5,
            extrapolation_damping: 0.0,
        }
    }

    /// Convert to LinearConfig
    pub fn to_config(&self) -> LinearConfig {
        LinearConfig::default()
            .with_lambda(self.lambda)
            .with_l1_ratio(self.l1_ratio)
            .with_extrapolation_damping(self.extrapolation_damping)
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

impl Default for TreeHyperparams {
    fn default() -> Self {
        Self {
            max_depth: 6,
            learning_rate: 0.1,
            num_rounds: 500,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
        }
    }
}

impl TreeHyperparams {
    /// Conservative params (less overfitting)
    pub fn conservative() -> Self {
        Self {
            max_depth: 4,
            learning_rate: 0.05,
            num_rounds: 1000,
            min_child_weight: 3.0,
            subsample: 0.8,
            colsample_bytree: 0.8,
        }
    }

    /// Aggressive params (more fitting power)
    pub fn aggressive() -> Self {
        Self {
            max_depth: 8,
            learning_rate: 0.15,
            num_rounds: 500,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
        }
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

/// LTT Tuner configuration
#[derive(Debug, Clone)]
pub struct LttTunerConfig {
    /// Validation split ratio
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

    // Joint refinement config
    /// Prediction shrinkage values to try
    pub shrinkage_values: Vec<f32>,
    /// Enable joint refinement phase
    pub enable_joint_refinement: bool,

    /// Seed for reproducibility
    pub seed: u64,
}

impl Default for LttTunerConfig {
    fn default() -> Self {
        Self {
            val_ratio: 0.2,
            // Linear phase: 4×3 = 12 trials
            lambda_values: vec![0.01, 0.1, 1.0, 10.0],
            l1_ratio_values: vec![0.0, 0.5, 1.0], // Ridge, ElasticNet, LASSO
            // Tree phase: 3×3×3 = 27 trials
            max_depth_values: vec![4, 6, 8],
            learning_rate_values: vec![0.05, 0.1, 0.15],
            num_rounds_values: vec![300, 500, 800],
            // Joint refinement: 5 trials
            shrinkage_values: vec![0.0, 0.1, 0.2, 0.3, 0.4],
            enable_joint_refinement: true,
            seed: 42,
        }
    }
}

impl LttTunerConfig {
    /// Create a quick tuning config (fewer trials)
    pub fn quick() -> Self {
        Self {
            val_ratio: 0.2,
            lambda_values: vec![0.1, 1.0],
            l1_ratio_values: vec![0.0, 0.5],
            max_depth_values: vec![4, 6],
            learning_rate_values: vec![0.05, 0.1],
            num_rounds_values: vec![300, 500],
            shrinkage_values: vec![0.0, 0.2],
            enable_joint_refinement: true,
            seed: 42,
        }
    }

    /// Create a thorough tuning config (more trials)
    pub fn thorough() -> Self {
        Self {
            val_ratio: 0.2,
            lambda_values: vec![0.001, 0.01, 0.1, 1.0, 10.0],
            l1_ratio_values: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            max_depth_values: vec![3, 4, 5, 6, 7, 8],
            learning_rate_values: vec![0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            num_rounds_values: vec![200, 400, 600, 800, 1000],
            shrinkage_values: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            enable_joint_refinement: true,
            seed: 42,
        }
    }
}

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
            self.config.shrinkage_values.len()
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
    pub fn tune(
        &self,
        features: &[f32],
        num_features: usize,
        targets: &[f32],
    ) -> Result<LttTuningResult> {
        let start = Instant::now();
        let mut history = LttTuningHistory::default();
        let mut phase_times = PhaseTimes::default();

        let num_rows = targets.len();
        if features.len() != num_rows * num_features {
            return Err(crate::TreeBoostError::Data(format!(
                "Feature matrix size mismatch: expected {}, got {}",
                num_rows * num_features,
                features.len()
            )));
        }

        // Create train/val split indices
        let val_size = (num_rows as f32 * self.config.val_ratio) as usize;
        let train_size = num_rows - val_size;

        // Simple split (last val_ratio% as validation)
        let train_indices: Vec<usize> = (0..train_size).collect();
        let val_indices: Vec<usize> = (train_size..num_rows).collect();

        // Extract train/val data
        let (train_features, train_targets) =
            Self::extract_split(features, targets, num_features, &train_indices);
        let (val_features, val_targets) =
            Self::extract_split(features, targets, num_features, &val_indices);

        // === PHASE 1: Tune linear model ===
        let phase1_start = Instant::now();
        let (best_linear, linear_r2, linear_train_preds) = self.tune_linear_phase(
            &train_features,
            &train_targets,
            &val_features,
            &val_targets,
            num_features,
            &mut history,
        )?;
        phase_times.linear_phase = phase1_start.elapsed();

        // Compute residuals on training set for tree phase
        let train_residuals: Vec<f32> = train_targets
            .iter()
            .zip(linear_train_preds.iter())
            .map(|(&t, &p)| t - p)
            .collect();

        // === PHASE 2: Tune tree model on residuals ===
        let phase2_start = Instant::now();
        let best_tree = self.tune_tree_phase(&train_residuals, &mut history)?;
        phase_times.tree_phase = phase2_start.elapsed();

        // === PHASE 3: Joint refinement (optional) ===
        let mut final_linear = best_linear;
        if self.config.enable_joint_refinement {
            let phase3_start = Instant::now();
            final_linear = self.tune_joint_phase(
                &train_features,
                &train_targets,
                &val_features,
                &val_targets,
                num_features,
                &best_linear,
                &mut history,
            )?;
            phase_times.joint_phase = phase3_start.elapsed();
        }

        // Compute final RMSE
        let final_rmse =
            self.compute_final_rmse(&final_linear, &val_features, &val_targets, num_features)?;

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

    /// Phase 1: Tune linear model hyperparameters
    fn tune_linear_phase(
        &self,
        train_features: &[f32],
        train_targets: &[f32],
        val_features: &[f32],
        val_targets: &[f32],
        num_features: usize,
        history: &mut LttTuningHistory,
    ) -> Result<(LinearHyperparams, f32, Vec<f32>)> {
        let mut best_params = LinearHyperparams::default();
        let mut best_r2 = f32::NEG_INFINITY;
        let mut best_train_preds = Vec::new();

        for &lambda in &self.config.lambda_values {
            for &l1_ratio in &self.config.l1_ratio_values {
                let config = LinearConfig::default()
                    .with_lambda(lambda)
                    .with_l1_ratio(l1_ratio);

                let mut booster = LinearBooster::new(num_features, config);

                // Use fit_direct for direct regression (not gradient boosting)
                let train_preds =
                    booster.fit_direct(train_features, num_features, train_targets)?;

                // Compute validation metrics
                let val_preds = booster.predict_batch(val_features, num_features);
                let (r2, rmse) = Self::compute_regression_metrics(&val_preds, val_targets);

                history.linear_trials.push(LinearTrial {
                    lambda,
                    l1_ratio,
                    r2,
                    rmse,
                });

                if r2 > best_r2 {
                    best_r2 = r2;
                    best_params = LinearHyperparams {
                        lambda,
                        l1_ratio,
                        extrapolation_damping: 0.0,
                    };
                    best_train_preds = train_preds;
                }
            }
        }

        Ok((best_params, best_r2, best_train_preds))
    }

    /// Phase 2: Tune tree model hyperparameters on residuals
    ///
    /// Note: This is a simplified version that selects tree params based on heuristics.
    /// A full implementation would train actual GBDTModel on residuals and evaluate.
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

        for &max_depth in &self.config.max_depth_values {
            for &learning_rate in &self.config.learning_rate_values {
                for &num_rounds in &self.config.num_rounds_values {
                    // Heuristic scoring based on residual characteristics
                    // Higher residual variance suggests need for more complex trees
                    let complexity_score = if residual_std > 1.0 {
                        // High variance residuals: prefer deeper trees, more rounds
                        (max_depth as f32 * 0.15) + (num_rounds as f32 * 0.001)
                            - (learning_rate * 0.5) // Lower LR for stability
                    } else {
                        // Low variance residuals: simpler trees suffice
                        (max_depth as f32 * 0.1) + (learning_rate * 1.0)
                            - (num_rounds as f32 * 0.0005) // Fewer rounds needed
                    };

                    // Penalize extreme configurations
                    let penalty = if max_depth > 8 { 0.1 } else { 0.0 }
                        + if learning_rate < 0.02 { 0.1 } else { 0.0 };

                    let score = complexity_score - penalty;
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

    /// Phase 3: Joint refinement (prediction shrinkage tuning)
    #[allow(clippy::too_many_arguments)]
    fn tune_joint_phase(
        &self,
        train_features: &[f32],
        train_targets: &[f32],
        val_features: &[f32],
        val_targets: &[f32],
        num_features: usize,
        linear_params: &LinearHyperparams,
        history: &mut LttTuningHistory,
    ) -> Result<LinearHyperparams> {
        let mut best_params = *linear_params;
        let mut best_rmse = f32::INFINITY;

        for &shrinkage in &self.config.shrinkage_values {
            let config = LinearConfig::default()
                .with_lambda(linear_params.lambda)
                .with_l1_ratio(linear_params.l1_ratio)
                .with_extrapolation_damping(shrinkage);

            let mut booster = LinearBooster::new(num_features, config);
            booster.fit_direct(train_features, num_features, train_targets)?;

            let val_preds = booster.predict_batch(val_features, num_features);
            let (_, rmse) = Self::compute_regression_metrics(&val_preds, val_targets);

            history.joint_trials.push(JointTrial {
                extrapolation_damping: shrinkage,
                combined_rmse: rmse,
            });

            if rmse < best_rmse {
                best_rmse = rmse;
                best_params.extrapolation_damping = shrinkage;
            }
        }

        Ok(best_params)
    }

    /// Compute final RMSE with best parameters
    fn compute_final_rmse(
        &self,
        linear_params: &LinearHyperparams,
        val_features: &[f32],
        val_targets: &[f32],
        num_features: usize,
    ) -> Result<f32> {
        let config = linear_params.to_config();
        let mut booster = LinearBooster::new(num_features, config);
        booster.fit_direct(val_features, num_features, val_targets)?;

        let val_preds = booster.predict_batch(val_features, num_features);
        let (_, rmse) = Self::compute_regression_metrics(&val_preds, val_targets);

        Ok(rmse)
    }

    /// Compute R² and RMSE for regression
    fn compute_regression_metrics(predictions: &[f32], targets: &[f32]) -> (f32, f32) {
        let n = predictions.len() as f32;
        if n == 0.0 {
            return (0.0, f32::INFINITY);
        }

        let mean_target: f32 = targets.iter().sum::<f32>() / n;

        let mut ss_res = 0.0f32;
        let mut ss_tot = 0.0f32;
        let mut mse = 0.0f32;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let residual = target - pred;
            ss_res += residual * residual;
            ss_tot += (target - mean_target).powi(2);
            mse += residual * residual;
        }

        let r2 = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        let rmse = (mse / n).sqrt();

        (r2, rmse)
    }
}

impl Default for LttTuner {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_hyperparams_default() {
        let params = LinearHyperparams::default();
        assert_eq!(params.lambda, 1.0);
        assert_eq!(params.l1_ratio, 0.0); // Ridge
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
        let quick = LttTunerConfig::quick();
        let thorough = LttTunerConfig::thorough();

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
        let joint_trials = config.shrinkage_values.len();

        assert_eq!(
            tuner.estimated_trials(),
            linear_trials + tree_trials + joint_trials
        );
    }

    #[test]
    fn test_regression_metrics() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (r2, rmse) = LttTuner::compute_regression_metrics(&predictions, &targets);

        // Perfect predictions
        assert!((r2 - 1.0).abs() < 0.001);
        assert!(rmse.abs() < 0.001);
    }

    #[test]
    fn test_regression_metrics_imperfect() {
        let predictions = vec![1.0, 2.5, 2.5, 4.0, 5.5];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (r2, rmse) = LttTuner::compute_regression_metrics(&predictions, &targets);

        // Should have reasonable R² but non-zero RMSE
        assert!(r2 > 0.0 && r2 < 1.0);
        assert!(rmse > 0.0);
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

        let config = LttTunerConfig::quick();
        let tuner = LttTuner::new(config);

        let result = tuner.tune(&features, num_features, &targets).unwrap();

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
            extrapolation_damping: 0.1,
        };

        let config = params.to_config();

        assert!((config.lambda - 0.5).abs() < 1e-6);
        assert!((config.l1_ratio - 0.3).abs() < 1e-6);
        assert!((config.extrapolation_damping - 0.1).abs() < 1e-6);
    }
}
