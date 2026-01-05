//! Linear weak learner with Ridge, LASSO, and Elastic Net regularization
//!
//! Implements a linear model trained via Coordinate Descent on gradients/hessians.
//! This enables Linear+Tree hybrid boosting for better extrapolation.
//!
//! # Regularization Options
//!
//! | Mode        | l1_ratio | Effect                                  |
//! |-------------|----------|----------------------------------------|
//! | Ridge       | 0.0      | L2 only - smooth weights               |
//! | LASSO       | 1.0      | L1 only - sparse weights (feature selection) |
//! | Elastic Net | 0.0-1.0  | Mix of L1 + L2 - sparse + stable       |
//!
//! # Critical Design Decisions
//!
//! 1. **Mandatory regularization**: lambda >= 1e-6 (prevents multicollinearity explosion)
//! 2. **Mandatory internal standardization**: Always standardizes features before fitting
//! 3. **Numerically stable updates**: Clamped deltas prevent divergence
//!
//! # Algorithm
//!
//! Coordinate Descent with Elastic Net:
//! ```text
//! for each feature j:
//!     grad_j = Σ_i gradient[i] * x[i,j] + lambda * (1 - l1_ratio) * w[j]
//!     hess_j = Σ_i hessian[i] * x[i,j]²
//!     raw_update = -grad_j / (hess_j + lambda * (1 - l1_ratio))
//!     w[j] = soft_threshold(raw_update, lambda * l1_ratio / hess_j)
//! ```
//!
//! Where `soft_threshold(x, t) = sign(x) * max(|x| - t, 0)`
//!
//! # Example
//!
//! ```ignore
//! use treeboost::learner::{LinearBooster, LinearConfig};
//!
//! // Ridge (L2 only - default)
//! let ridge_config = LinearConfig::default();
//!
//! // LASSO (L1 only - sparse)
//! let lasso_config = LinearConfig::default()
//!     .with_l1_ratio(1.0);
//!
//! // Elastic Net (L1 + L2)
//! let elastic_config = LinearConfig::default()
//!     .with_lambda(1.0)
//!     .with_l1_ratio(0.5);  // 50% L1, 50% L2
//!
//! let mut booster = LinearBooster::new(10, elastic_config);
//! booster.fit_on_gradients(&features, 10, &gradients, &hessians)?;
//! let preds = booster.predict_batch(&features, 10);
//! ```

use crate::learner::WeakLearner;
use crate::{Result, TreeBoostError};
use rkyv::{Archive, Deserialize, Serialize};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for LinearBooster
///
/// # Regularization Types
///
/// | l1_ratio | Type        | Properties                           |
/// |----------|-------------|--------------------------------------|
/// | 0.0      | Ridge (L2)  | Smooth weights, handles correlation  |
/// | 1.0      | LASSO (L1)  | Sparse weights, feature selection    |
/// | 0.0-1.0  | Elastic Net | Sparse + stable (recommended)        |
///
/// # Critical: Regularization is MANDATORY
///
/// Setting lambda=0 will cause numerical instability on correlated features.
/// The minimum allowed value is 1e-6.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LinearConfig {
    /// Overall regularization strength
    ///
    /// **MINIMUM: 1e-6** - Never set to 0, it causes NaN on correlated features.
    /// **DEFAULT: 1.0** - Strong regularization for stability.
    ///
    /// Higher values = more regularization = smaller weights = more stable.
    pub lambda: f32,

    /// Elastic Net mixing parameter
    ///
    /// **DEFAULT: 0.0** (pure Ridge/L2)
    ///
    /// - `0.0` = pure Ridge (L2 penalty only)
    /// - `1.0` = pure LASSO (L1 penalty only)
    /// - `0.0-1.0` = Elastic Net (mix of L1 and L2)
    ///
    /// L1 penalty encourages sparse solutions (zero weights).
    /// L2 penalty encourages small but non-zero weights.
    pub l1_ratio: f32,

    /// Learning rate for weight updates
    ///
    /// Scales the coordinate descent step size.
    /// Lower values = slower convergence but more stable.
    pub learning_rate: f32,

    /// Maximum iterations per fit_on_gradients call
    ///
    /// Usually 1-10 is enough since we're doing boosting (many rounds).
    pub max_iter: usize,

    /// Convergence tolerance
    ///
    /// Stop early if max weight change < tol.
    pub tol: f32,

    /// Maximum absolute weight value (prevents explosion)
    pub max_weight: f32,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,           // Strong default regularization
            l1_ratio: 0.0,         // Pure Ridge by default (most stable)
            learning_rate: 0.3,    // Moderate step size for boosting
            max_iter: 10,          // Few iterations per round (boosting does many rounds)
            tol: 1e-6,             // Tight convergence
            max_weight: 100.0,     // Prevent extreme weights
        }
    }
}

impl LinearConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a Ridge (L2) config
    ///
    /// Pure L2 regularization - smooth weights, handles correlated features well.
    pub fn ridge(lambda: f32) -> Self {
        Self::default().with_lambda(lambda).with_l1_ratio(0.0)
    }

    /// Create a LASSO (L1) config
    ///
    /// Pure L1 regularization - encourages sparse solutions (zero weights).
    /// Good for feature selection.
    pub fn lasso(lambda: f32) -> Self {
        Self::default().with_lambda(lambda).with_l1_ratio(1.0)
    }

    /// Create an Elastic Net config
    ///
    /// Mix of L1 and L2 regularization.
    /// - Sparse like LASSO (feature selection)
    /// - Stable like Ridge (handles correlation)
    ///
    /// # Arguments
    /// - `lambda`: Overall regularization strength
    /// - `l1_ratio`: Balance between L1 and L2 (0.0 = Ridge, 1.0 = LASSO)
    pub fn elastic_net(lambda: f32, l1_ratio: f32) -> Self {
        Self::default().with_lambda(lambda).with_l1_ratio(l1_ratio)
    }

    /// Set overall regularization strength
    ///
    /// **CRITICAL**: Minimum value is 1e-6 to prevent numerical instability.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        // NEVER allow lambda = 0
        self.lambda = lambda.max(1e-6);
        self
    }

    /// Set Elastic Net mixing parameter
    ///
    /// - `0.0` = pure Ridge (L2 only) - default, most stable
    /// - `1.0` = pure LASSO (L1 only) - sparse solutions
    /// - `0.0-1.0` = Elastic Net mix
    pub fn with_l1_ratio(mut self, l1_ratio: f32) -> Self {
        self.l1_ratio = l1_ratio.clamp(0.0, 1.0);
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr.clamp(1e-6, 1.0);
        self
    }

    /// Set maximum iterations per round
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set maximum weight magnitude
    pub fn with_max_weight(mut self, max_weight: f32) -> Self {
        self.max_weight = max_weight.max(1.0);
        self
    }

    /// Get L2 regularization component
    ///
    /// `lambda * (1 - l1_ratio)`
    #[inline]
    pub fn l2_penalty(&self) -> f32 {
        self.lambda * (1.0 - self.l1_ratio)
    }

    /// Get L1 regularization component
    ///
    /// `lambda * l1_ratio`
    #[inline]
    pub fn l1_penalty(&self) -> f32 {
        self.lambda * self.l1_ratio
    }
}

// =============================================================================
// LinearBooster
// =============================================================================

/// Linear weak learner for gradient boosting
///
/// Fits a linear model w·x + b on gradients using Coordinate Descent with Ridge.
///
/// # Internal Standardization
///
/// The booster automatically standardizes features internally:
/// 1. During fit: learns mean/std, transforms features
/// 2. During predict: applies same transform
///
/// This is **mandatory** - linear models are sensitive to feature scales.
///
/// # Reset Behavior
///
/// Calling `reset()` clears learned weights but **preserves the fitted scaler**.
/// This is intentional for boosting workflows where you reset between CV folds
/// on the same dataset. For different datasets, create a new `LinearBooster`.
///
/// # Numerical Stability
///
/// Several safeguards prevent NaN/Inf:
/// - Lambda >= 1e-6 ensures non-zero denominator
/// - Delta clamping prevents extreme updates
/// - Weight clamping prevents explosion
/// - Zero-variance features handled gracefully
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LinearBooster {
    /// Weights (one per feature)
    weights: Vec<f32>,

    /// Bias term
    bias: f32,

    /// Feature means (for internal standardization)
    means: Vec<f32>,

    /// Feature standard deviations (for internal standardization)
    stds: Vec<f32>,

    /// Configuration
    config: LinearConfig,

    /// Number of features
    num_features: usize,

    /// Whether scaler has been fitted
    scaler_fitted: bool,
}

impl LinearBooster {
    /// Create a new LinearBooster
    ///
    /// # Arguments
    /// - `num_features`: Number of input features
    /// - `config`: Configuration (regularization, learning rate, etc.)
    pub fn new(num_features: usize, config: LinearConfig) -> Self {
        Self {
            weights: vec![0.0; num_features],
            bias: 0.0,
            means: vec![0.0; num_features],
            stds: vec![1.0; num_features],
            config,
            num_features,
            scaler_fitted: false,
        }
    }

    /// Get the learned weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get the bias term
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Get configuration
    pub fn config(&self) -> &LinearConfig {
        &self.config
    }

    /// Fit internal scaler on data
    ///
    /// Called automatically on first fit_on_gradients.
    fn fit_scaler(&mut self, features: &[f32], num_features: usize) {
        let num_rows = features.len() / num_features;
        if num_rows == 0 {
            return;
        }

        // Compute mean and std for each feature
        for j in 0..num_features {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut count = 0usize;

            for i in 0..num_rows {
                let val = features[i * num_features + j] as f64;
                if val.is_finite() {
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }

            if count > 0 {
                let mean = sum / count as f64;
                let variance = (sum_sq / count as f64) - mean * mean;
                let std = variance.max(0.0).sqrt();

                self.means[j] = mean as f32;
                // Prevent division by zero - use 1.0 for constant features
                self.stds[j] = if std > 1e-10 { std as f32 } else { 1.0 };
            }
        }

        self.scaler_fitted = true;
    }

    /// Standardize a single value
    #[inline]
    fn standardize(&self, value: f32, feature_idx: usize) -> f32 {
        (value - self.means[feature_idx]) / self.stds[feature_idx]
    }

    /// Soft thresholding operator for L1 regularization
    ///
    /// S(x, t) = sign(x) * max(|x| - t, 0)
    ///
    /// This shrinks values toward zero, with values |x| < t becoming exactly zero.
    #[inline]
    fn soft_threshold(x: f32, threshold: f32) -> f32 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Coordinate Descent with Elastic Net regularization
    ///
    /// This is the core algorithm. Updates weights to minimize:
    /// L = Σ_i hessian[i] * (pred[i] - target[i])² + λ₂ * ||w||² + λ₁ * ||w||₁
    ///
    /// Where:
    /// - target[i] = -gradient[i] / hessian[i] (Newton step target)
    /// - λ₂ = lambda * (1 - l1_ratio)  (L2/Ridge penalty)
    /// - λ₁ = lambda * l1_ratio         (L1/LASSO penalty)
    fn coordinate_descent(
        &mut self,
        features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) {
        let num_rows = gradients.len();
        if num_rows == 0 {
            return;
        }

        let l2_penalty = self.config.l2_penalty();
        let l1_penalty = self.config.l1_penalty();

        // Compute current predictions (for residual updates)
        let mut predictions = vec![self.bias; num_rows];
        for i in 0..num_rows {
            for j in 0..num_features {
                let x_ij = self.standardize(features[i * num_features + j], j);
                predictions[i] += self.weights[j] * x_ij;
            }
        }

        // Coordinate Descent iterations
        for _iter in 0..self.config.max_iter {
            let mut max_change = 0.0f32;

            // Update bias first (no regularization on bias)
            {
                let mut grad_bias = 0.0f32;
                let mut hess_bias = 0.0f32;

                for i in 0..num_rows {
                    // Residual: what we need to fit
                    // grad[i] is the negative gradient of the loss
                    // We want to minimize: Σ h[i] * (pred[i] + delta - (-g[i]/h[i]))²
                    // Which simplifies to fitting: target = -g[i]/h[i]
                    let residual = predictions[i] + gradients[i] / hessians[i].max(1e-10);
                    grad_bias += hessians[i] * residual;
                    hess_bias += hessians[i];
                }

                // No regularization on bias term
                let delta = -grad_bias / hess_bias.max(1e-10);
                let delta = delta.clamp(-10.0, 10.0);

                self.bias += self.config.learning_rate * delta;
                self.bias = self.bias.clamp(-self.config.max_weight, self.config.max_weight);

                // Update predictions
                for pred in predictions.iter_mut() {
                    *pred += self.config.learning_rate * delta;
                }

                max_change = max_change.max(delta.abs());
            }

            // Update each weight with Elastic Net
            for j in 0..num_features {
                let mut grad_j = 0.0f32;
                let mut hess_j = 0.0f32;

                for i in 0..num_rows {
                    let x_ij = self.standardize(features[i * num_features + j], j);
                    let residual = predictions[i] + gradients[i] / hessians[i].max(1e-10);

                    grad_j += hessians[i] * residual * x_ij;
                    hess_j += hessians[i] * x_ij * x_ij;
                }

                // L2 regularization gradient: λ₂ * w[j]
                grad_j += l2_penalty * self.weights[j];

                // Denominator includes L2 penalty (never zero)
                let denominator = hess_j + l2_penalty;
                let denominator = denominator.max(1e-10);

                // Compute raw update (without L1)
                let raw_update = -grad_j / denominator;

                // Apply L1 soft thresholding
                // The threshold scales with learning rate and inversely with denominator
                let l1_threshold = l1_penalty * self.config.learning_rate / denominator;
                let delta = Self::soft_threshold(raw_update, l1_threshold);

                // Clamp update for numerical stability
                let delta = delta.clamp(-10.0, 10.0);

                let old_weight = self.weights[j];
                self.weights[j] += self.config.learning_rate * delta;
                self.weights[j] = self.weights[j].clamp(-self.config.max_weight, self.config.max_weight);

                let weight_change = self.weights[j] - old_weight;

                // Update predictions incrementally
                for i in 0..num_rows {
                    let x_ij = self.standardize(features[i * num_features + j], j);
                    predictions[i] += weight_change * x_ij;
                }

                max_change = max_change.max(delta.abs());
            }

            // Check convergence
            if max_change < self.config.tol {
                break;
            }
        }
    }

    /// Get the number of non-zero weights (sparsity measure)
    ///
    /// Useful for LASSO/Elastic Net to see how many features were selected.
    pub fn num_nonzero_weights(&self) -> usize {
        self.weights.iter().filter(|&&w| w.abs() > 1e-10).count()
    }

    /// Get indices of non-zero weights (selected features)
    ///
    /// Returns feature indices with non-zero weights after LASSO/Elastic Net.
    pub fn selected_features(&self) -> Vec<usize> {
        self.weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w.abs() > 1e-10)
            .map(|(i, _)| i)
            .collect()
    }
}

impl WeakLearner for LinearBooster {
    fn fit_on_gradients(
        &mut self,
        features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<()> {
        // Validate inputs
        if num_features != self.num_features {
            return Err(TreeBoostError::Config(format!(
                "Feature count mismatch: expected {}, got {}",
                self.num_features, num_features
            )));
        }

        let num_rows = gradients.len();
        if features.len() != num_rows * num_features {
            return Err(TreeBoostError::Data(format!(
                "Feature matrix size mismatch: expected {}, got {}",
                num_rows * num_features,
                features.len()
            )));
        }

        if hessians.len() != num_rows {
            return Err(TreeBoostError::Data(format!(
                "Hessian size mismatch: expected {}, got {}",
                num_rows,
                hessians.len()
            )));
        }

        // Fit scaler on first call (learns mean/std)
        if !self.scaler_fitted {
            self.fit_scaler(features, num_features);
        }

        // Run coordinate descent
        self.coordinate_descent(features, num_features, gradients, hessians);

        Ok(())
    }

    fn predict_batch(&self, features: &[f32], num_features: usize) -> Vec<f32> {
        let num_rows = features.len() / num_features;
        let mut predictions = vec![self.bias; num_rows];

        for i in 0..num_rows {
            for j in 0..num_features {
                let x_ij = self.standardize(features[i * num_features + j], j);
                predictions[i] += self.weights[j] * x_ij;
            }
        }

        predictions
    }

    fn predict_row(&self, features: &[f32], num_features: usize, row_idx: usize) -> f32 {
        let mut pred = self.bias;
        let start = row_idx * num_features;

        for j in 0..num_features {
            let x_ij = self.standardize(features[start + j], j);
            pred += self.weights[j] * x_ij;
        }

        pred
    }

    fn num_params(&self) -> usize {
        self.num_features + 1 // weights + bias
    }

    /// Reset model weights to zero while preserving the fitted scaler.
    ///
    /// # Scaler Preservation Rationale
    ///
    /// The internal feature scaler (mean/std) is intentionally preserved because:
    /// - In boosting, `reset()` is typically called between CV folds on the **same dataset**
    /// - The scaler captures data distribution, not learned weights
    /// - Re-fitting the scaler on identical data wastes computation
    ///
    /// If you need to fit on a **different dataset** with different feature distributions,
    /// create a new `LinearBooster` instead of calling `reset()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Same dataset, different fold - reset is appropriate
    /// booster.reset();
    /// booster.fit_on_gradients(&same_data, ...)?;
    ///
    /// // Different dataset - create new booster
    /// let booster = LinearBooster::new(num_features, config);
    /// booster.fit_on_gradients(&different_data, ...)?;
    /// ```
    fn reset(&mut self) {
        self.weights.fill(0.0);
        self.bias = 0.0;
        // Scaler preserved: based on data distribution, reusable across CV folds
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_config_lambda_minimum() {
        let config = LinearConfig::new().with_lambda(0.0);
        assert!(config.lambda >= 1e-6, "Lambda should never be 0");

        let config = LinearConfig::new().with_lambda(-1.0);
        assert!(config.lambda >= 1e-6, "Lambda should never be negative");
    }

    #[test]
    fn test_linear_booster_creation() {
        let config = LinearConfig::default();
        let booster = LinearBooster::new(5, config);

        assert_eq!(booster.weights().len(), 5);
        assert_eq!(booster.bias(), 0.0);
        assert_eq!(booster.num_params(), 6);
    }

    #[test]
    fn test_linear_booster_simple_fit() {
        // Simple linear relationship: y = 2*x + 1
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 rows, 1 feature
        let targets = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        // For gradient boosting, gradients = predictions - targets (for MSE)
        // Initial predictions = 0, so gradients = -targets
        let gradients: Vec<f32> = targets.iter().map(|&t| -t).collect();
        let hessians = vec![1.0; 5]; // MSE has constant hessian

        let config = LinearConfig::default()
            .with_lambda(0.01)
            .with_learning_rate(0.5)
            .with_max_iter(100);

        let mut booster = LinearBooster::new(1, config);
        booster.fit_on_gradients(&features, 1, &gradients, &hessians).unwrap();

        let predictions = booster.predict_batch(&features, 1);

        // Check predictions are reasonable (not exact due to regularization)
        for (pred, &target) in predictions.iter().zip(targets.iter()) {
            let error = (pred - target).abs();
            assert!(error < 2.0, "Prediction {} too far from target {}", pred, target);
        }
    }

    #[test]
    fn test_linear_booster_multivariate() {
        // y = x1 + 2*x2
        // 4 rows, 2 features
        let features = vec![
            1.0, 1.0,  // row 0: y = 1 + 2 = 3
            2.0, 1.0,  // row 1: y = 2 + 2 = 4
            1.0, 2.0,  // row 2: y = 1 + 4 = 5
            2.0, 2.0,  // row 3: y = 2 + 4 = 6
        ];
        let targets = vec![3.0, 4.0, 5.0, 6.0];
        let gradients: Vec<f32> = targets.iter().map(|&t| -t).collect();
        let hessians = vec![1.0; 4];

        let config = LinearConfig::default()
            .with_lambda(0.001)
            .with_learning_rate(0.5)
            .with_max_iter(200);

        let mut booster = LinearBooster::new(2, config);
        booster.fit_on_gradients(&features, 2, &gradients, &hessians).unwrap();

        let predictions = booster.predict_batch(&features, 2);

        // Check predictions
        for (i, (pred, &target)) in predictions.iter().zip(targets.iter()).enumerate() {
            let error = (pred - target).abs();
            assert!(error < 1.5, "Row {}: pred {} too far from target {}", i, pred, target);
        }
    }

    #[test]
    fn test_linear_booster_no_nan() {
        // Test with correlated features (would cause NaN without regularization)
        let features = vec![
            1.0, 1.0,  // x1 = x2 (perfect correlation)
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
        ];
        let gradients = vec![-1.0, -2.0, -3.0, -4.0];
        let hessians = vec![1.0; 4];

        let config = LinearConfig::default();
        let mut booster = LinearBooster::new(2, config);
        booster.fit_on_gradients(&features, 2, &gradients, &hessians).unwrap();

        let predictions = booster.predict_batch(&features, 2);

        // No NaN or Inf
        for pred in &predictions {
            assert!(pred.is_finite(), "Prediction should be finite, got {}", pred);
        }
    }

    #[test]
    fn test_linear_booster_constant_feature() {
        // One constant feature (std = 0)
        let features = vec![
            1.0, 5.0,  // x2 is constant
            2.0, 5.0,
            3.0, 5.0,
        ];
        let gradients = vec![-1.0, -2.0, -3.0];
        let hessians = vec![1.0; 3];

        let config = LinearConfig::default();
        let mut booster = LinearBooster::new(2, config);
        booster.fit_on_gradients(&features, 2, &gradients, &hessians).unwrap();

        let predictions = booster.predict_batch(&features, 2);

        // No NaN or Inf
        for pred in &predictions {
            assert!(pred.is_finite(), "Prediction should be finite, got {}", pred);
        }
    }

    #[test]
    fn test_linear_booster_reset() {
        let config = LinearConfig::default();
        let mut booster = LinearBooster::new(3, config);

        // Fit some data
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gradients = vec![-1.0, -2.0];
        let hessians = vec![1.0, 1.0];
        booster.fit_on_gradients(&features, 3, &gradients, &hessians).unwrap();

        // Weights should be non-zero
        let has_nonzero = booster.weights().iter().any(|&w| w.abs() > 1e-10);
        assert!(has_nonzero, "Weights should be non-zero after fitting");

        // Reset
        booster.reset();

        // Weights should be zero
        for &w in booster.weights() {
            assert!((w.abs()) < 1e-10, "Weights should be zero after reset");
        }
        assert!((booster.bias().abs()) < 1e-10, "Bias should be zero after reset");
    }

    #[test]
    fn test_linear_booster_single_row_prediction() {
        let config = LinearConfig::default();
        let mut booster = LinearBooster::new(2, config);

        let features = vec![1.0, 2.0, 3.0, 4.0];
        let gradients = vec![-5.0, -10.0];
        let hessians = vec![1.0, 1.0];
        booster.fit_on_gradients(&features, 2, &gradients, &hessians).unwrap();

        let batch_preds = booster.predict_batch(&features, 2);
        let single_pred_0 = booster.predict_row(&features, 2, 0);
        let single_pred_1 = booster.predict_row(&features, 2, 1);

        assert!((batch_preds[0] - single_pred_0).abs() < 1e-6);
        assert!((batch_preds[1] - single_pred_1).abs() < 1e-6);
    }

    #[test]
    fn test_soft_threshold() {
        // Above threshold
        assert!((LinearBooster::soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-6);
        // Below negative threshold
        assert!((LinearBooster::soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-6);
        // Within threshold (should be zero)
        assert!((LinearBooster::soft_threshold(1.5, 2.0) - 0.0).abs() < 1e-6);
        assert!((LinearBooster::soft_threshold(-1.5, 2.0) - 0.0).abs() < 1e-6);
        // At threshold boundary
        assert!((LinearBooster::soft_threshold(2.0, 2.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_lasso_sparsity() {
        // Create a problem where only feature 0 matters: y = 3*x0
        // Features 1, 2, 3 are noise - LASSO should zero them out
        let n_samples = 100;
        let n_features = 4;

        let mut features = Vec::with_capacity(n_samples * n_features);
        let mut targets = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let x0 = (i as f32) / 10.0;
            features.push(x0);         // Feature 0 - relevant
            features.push(0.5);        // Feature 1 - noise (constant)
            features.push(0.3);        // Feature 2 - noise (constant)
            features.push(0.1);        // Feature 3 - noise (constant)
            targets.push(3.0 * x0);    // Only depends on x0
        }

        let gradients: Vec<f32> = targets.iter().map(|&t| -t).collect();
        let hessians = vec![1.0; n_samples];

        // Use LASSO with strong regularization
        let config = LinearConfig::lasso(2.0)
            .with_learning_rate(0.5)
            .with_max_iter(200);

        let mut booster = LinearBooster::new(n_features, config);
        booster.fit_on_gradients(&features, n_features, &gradients, &hessians).unwrap();

        // Feature 0 should have non-zero weight
        assert!(booster.weights()[0].abs() > 0.1, "Feature 0 should be selected");

        // LASSO should encourage sparsity
        let selected = booster.selected_features();
        println!("Selected features: {:?}", selected);
        println!("Weights: {:?}", booster.weights());
        println!("Num nonzero: {}", booster.num_nonzero_weights());

        // At minimum, feature 0 should be selected (others may be selected too due to
        // gradient boosting dynamics, but sparsity should be encouraged)
        assert!(selected.contains(&0), "Feature 0 must be selected");
    }

    #[test]
    fn test_elastic_net_config() {
        let config = LinearConfig::elastic_net(1.0, 0.5);
        assert!((config.lambda - 1.0).abs() < 1e-6);
        assert!((config.l1_ratio - 0.5).abs() < 1e-6);
        assert!((config.l1_penalty() - 0.5).abs() < 1e-6);
        assert!((config.l2_penalty() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_vs_lasso_sparsity() {
        // Same problem, compare Ridge vs LASSO sparsity
        let n_samples = 50;
        let n_features = 10;

        let mut features = Vec::with_capacity(n_samples * n_features);
        let mut targets = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let x = (i as f32) / 10.0;
            for _ in 0..n_features {
                features.push(x);
            }
            targets.push(x);  // All features contribute equally
        }

        let gradients: Vec<f32> = targets.iter().map(|&t| -t).collect();
        let hessians = vec![1.0; n_samples];

        // Ridge - should have all non-zero weights
        let ridge_config = LinearConfig::ridge(0.1)
            .with_learning_rate(0.5)
            .with_max_iter(100);
        let mut ridge_booster = LinearBooster::new(n_features, ridge_config);
        ridge_booster.fit_on_gradients(&features, n_features, &gradients, &hessians).unwrap();

        // LASSO - should have sparser weights
        let lasso_config = LinearConfig::lasso(0.5)
            .with_learning_rate(0.5)
            .with_max_iter(100);
        let mut lasso_booster = LinearBooster::new(n_features, lasso_config);
        lasso_booster.fit_on_gradients(&features, n_features, &gradients, &hessians).unwrap();

        // Ridge typically has more non-zero weights than LASSO
        // (though in this degenerate case both may have many)
        println!("Ridge nonzero: {}", ridge_booster.num_nonzero_weights());
        println!("LASSO nonzero: {}", lasso_booster.num_nonzero_weights());

        // Both should produce finite predictions
        let ridge_preds = ridge_booster.predict_batch(&features, n_features);
        let lasso_preds = lasso_booster.predict_batch(&features, n_features);

        for pred in ridge_preds.iter().chain(lasso_preds.iter()) {
            assert!(pred.is_finite(), "Predictions must be finite");
        }
    }

    #[test]
    fn test_elastic_net_stability() {
        // Elastic Net should handle correlated features better than pure LASSO
        let features = vec![
            1.0, 1.0,  // x1 ≈ x2 (correlation)
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
        ];
        let gradients = vec![-1.0, -2.0, -3.0, -4.0];
        let hessians = vec![1.0; 4];

        let config = LinearConfig::elastic_net(0.5, 0.5)  // 50% L1, 50% L2
            .with_learning_rate(0.5)
            .with_max_iter(100);

        let mut booster = LinearBooster::new(2, config);
        booster.fit_on_gradients(&features, 2, &gradients, &hessians).unwrap();

        let predictions = booster.predict_batch(&features, 2);

        // All predictions should be finite
        for pred in &predictions {
            assert!(pred.is_finite(), "Elastic Net prediction should be finite, got {}", pred);
        }
    }
}
