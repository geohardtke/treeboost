//! Stacking strategies for ensemble combination
//!
//! Provides Ridge regression and simple average stackers for combining
//! predictions from multiple base models.

use super::traits::Stacker;
use crate::defaults::ensemble as ensemble_defaults;

/// Configuration for Ridge stacking
#[derive(Debug, Clone)]
pub struct StackingConfig {
    /// Ridge regularization parameter (alpha)
    pub alpha: f32,
    /// Whether to apply rank transformation before stacking
    pub rank_transform: bool,
    /// Whether to fit an intercept term
    pub fit_intercept: bool,
    /// Minimum weight to keep (clip smaller weights to zero)
    pub min_weight: f32,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            alpha: ensemble_defaults::DEFAULT_STACKING_ALPHA,
            rank_transform: ensemble_defaults::DEFAULT_RANK_TRANSFORM,
            fit_intercept: ensemble_defaults::DEFAULT_FIT_INTERCEPT,
            min_weight: ensemble_defaults::DEFAULT_MIN_WEIGHT,
        }
    }
}

impl StackingConfig {
    /// Create a new stacking config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Ridge alpha parameter
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Enable or disable rank transformation
    pub fn with_rank_transform(mut self, enabled: bool) -> Self {
        self.rank_transform = enabled;
        self
    }

    /// Enable or disable intercept fitting
    pub fn with_intercept(mut self, fit: bool) -> Self {
        self.fit_intercept = fit;
        self
    }

    /// Set minimum weight threshold
    pub fn with_min_weight(mut self, min: f32) -> Self {
        self.min_weight = min;
        self
    }
}

/// Ridge regression stacker
///
/// Combines model predictions using Ridge regression:
/// `y = X * w + b` where `w` is regularized by `||w||^2 * alpha`
///
/// Optionally applies rank transformation to predictions before fitting,
/// which can improve robustness when models have different prediction scales.
pub struct RidgeStacker {
    config: StackingConfig,
    weights: Vec<f32>,
    intercept: f32,
    fitted: bool,
}

impl RidgeStacker {
    /// Create a new Ridge stacker
    pub fn new(config: StackingConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            intercept: 0.0,
            fitted: false,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(StackingConfig::default())
    }

    /// Check if the stacker has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the fitted intercept term
    ///
    /// Returns the intercept from the Ridge regression model trained on out-of-fold
    /// predictions from ensemble members.
    ///
    /// # Returns
    ///
    /// - If `fit_intercept` was true during fitting: the computed intercept value
    /// - If `fit_intercept` was false during fitting: 0.0
    /// - If the stacker has never been fitted: 0.0
    ///
    /// # Ensemble Prediction Formula
    ///
    /// When using this intercept with weights from `weights()`, the prediction is:
    /// ```text
    /// prediction = sum(weights[i] * member_predictions[i]) + intercept()
    /// ```
    pub fn intercept(&self) -> f32 {
        self.intercept
    }
}

impl Stacker for RidgeStacker {
    fn fit(&mut self, oof_preds: &[Vec<f32>], targets: &[f32]) {
        if oof_preds.is_empty() || targets.is_empty() {
            self.weights = Vec::new();
            self.intercept = 0.0;
            self.fitted = true;
            return;
        }

        let n_samples = targets.len();
        let n_models = oof_preds.len();

        // Optional rank transformation
        let transformed: Vec<Vec<f32>> = if self.config.rank_transform {
            oof_preds.iter().map(|p| rank_transform(p)).collect()
        } else {
            oof_preds.to_vec()
        };

        // Center targets if fitting intercept
        let y_mean = if self.config.fit_intercept {
            targets.iter().sum::<f32>() / n_samples as f32
        } else {
            0.0
        };
        let y_centered: Vec<f32> = targets.iter().map(|&y| y - y_mean).collect();

        // Center features if fitting intercept
        let x_means: Vec<f32> = if self.config.fit_intercept {
            transformed
                .iter()
                .map(|col| col.iter().sum::<f32>() / n_samples as f32)
                .collect()
        } else {
            vec![0.0; n_models]
        };

        let x_centered: Vec<Vec<f32>> = transformed
            .iter()
            .zip(x_means.iter())
            .map(|(col, &mean)| col.iter().map(|&x| x - mean).collect())
            .collect();

        // Build X'X + alpha*I (n_models x n_models)
        let mut xtx = vec![vec![0.0f64; n_models]; n_models];
        for i in 0..n_models {
            for j in 0..n_models {
                let dot: f64 = x_centered[i]
                    .iter()
                    .zip(x_centered[j].iter())
                    .map(|(&a, &b)| (a as f64) * (b as f64))
                    .sum();
                xtx[i][j] = dot;
            }
            // Add regularization to diagonal
            xtx[i][i] += self.config.alpha as f64;
        }

        // Build X'y (n_models)
        let xty: Vec<f64> = x_centered
            .iter()
            .map(|col| {
                col.iter()
                    .zip(y_centered.iter())
                    .map(|(&x, &y)| (x as f64) * (y as f64))
                    .sum()
            })
            .collect();

        // Solve (X'X + alpha*I)w = X'y using Cholesky decomposition
        let w = solve_positive_definite(&xtx, &xty);

        // Apply minimum weight threshold
        self.weights = w
            .into_iter()
            .map(|w| {
                let w = w as f32;
                if w.abs() < self.config.min_weight {
                    0.0
                } else {
                    w
                }
            })
            .collect();

        // Compute intercept
        if self.config.fit_intercept {
            self.intercept = y_mean
                - self
                    .weights
                    .iter()
                    .zip(x_means.iter())
                    .map(|(&w, &m)| w * m)
                    .sum::<f32>();
        } else {
            self.intercept = 0.0;
        }

        self.fitted = true;
    }

    fn combine(&self, predictions: &[Vec<f32>]) -> Vec<f32> {
        if !self.fitted || predictions.is_empty() || self.weights.is_empty() {
            return Vec::new();
        }

        let n_samples = predictions.first().map(|p| p.len()).unwrap_or(0);

        // Optional rank transformation (must match training)
        let transformed: Vec<Vec<f32>> = if self.config.rank_transform {
            predictions.iter().map(|p| rank_transform(p)).collect()
        } else {
            predictions.to_vec()
        };

        (0..n_samples)
            .map(|i| {
                let weighted_sum: f32 = transformed
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(preds, &w)| preds[i] * w)
                    .sum();
                weighted_sum + self.intercept
            })
            .collect()
    }

    fn weights(&self) -> Option<&[f32]> {
        if self.fitted && !self.weights.is_empty() {
            Some(&self.weights)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "ridge"
    }
}

/// Simple average stacker (baseline)
///
/// Combines predictions by taking the mean across all models.
/// No fitting required - always gives equal weight to all models.
pub struct SimpleAverageStacker {
    n_models: usize,
}

impl SimpleAverageStacker {
    /// Create a new simple average stacker
    pub fn new() -> Self {
        Self { n_models: 0 }
    }
}

impl Default for SimpleAverageStacker {
    fn default() -> Self {
        Self::new()
    }
}

impl Stacker for SimpleAverageStacker {
    fn fit(&mut self, oof_preds: &[Vec<f32>], _targets: &[f32]) {
        self.n_models = oof_preds.len();
    }

    fn combine(&self, predictions: &[Vec<f32>]) -> Vec<f32> {
        if predictions.is_empty() {
            return Vec::new();
        }

        let n_samples = predictions[0].len();
        let n_models = predictions.len() as f32;

        (0..n_samples)
            .map(|i| predictions.iter().map(|p| p[i]).sum::<f32>() / n_models)
            .collect()
    }

    fn weights(&self) -> Option<&[f32]> {
        None // Equal weights, not explicitly stored
    }

    fn name(&self) -> &'static str {
        "simple_average"
    }
}

/// Transform predictions to ranks in [0, 1]
///
/// Rank transformation normalizes predictions to a uniform distribution,
/// which can improve stacking when models have different prediction scales.
pub fn rank_transform(predictions: &[f32]) -> Vec<f32> {
    let n = predictions.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.5];
    }

    // Get sorted indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predictions[a]
            .partial_cmp(&predictions[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign ranks (handling ties with average rank)
    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let value = predictions[indices[i]];
        let mut j = i + 1;

        // Find all elements with the same value (ties)
        while j < n && predictions[indices[j]] == value {
            j += 1;
        }

        // Average rank for ties
        let avg_rank = (i + j - 1) as f32 / 2.0;
        for k in i..j {
            ranks[indices[k]] = avg_rank / (n - 1) as f32;
        }

        i = j;
    }

    ranks
}

/// Solve positive definite system Ax = b using Cholesky decomposition
fn solve_positive_definite(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    // Cholesky decomposition: A = L * L^T
    let mut l = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for (li_k, lj_k) in l[i][..j].iter().zip(l[j][..j].iter()) {
                sum += li_k * lj_k;
            }

            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 {
                    // Not positive definite, fall back to regularization
                    l[i][j] = 1e-6;
                } else {
                    l[i][j] = diag.sqrt();
                }
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i][i];
    }

    // Backward substitution: L^T * x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / l[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stacking_config_default() {
        let config = StackingConfig::default();
        assert!((config.alpha - 10.0).abs() < 1e-6);
        assert!(!config.rank_transform);
        assert!(config.fit_intercept);
    }

    #[test]
    fn test_stacking_config_builder() {
        let config = StackingConfig::new()
            .with_alpha(5.0)
            .with_rank_transform(true)
            .with_intercept(false);

        assert!((config.alpha - 5.0).abs() < 1e-6);
        assert!(config.rank_transform);
        assert!(!config.fit_intercept);
    }

    #[test]
    fn test_rank_transform_basic() {
        let preds = vec![1.0, 3.0, 2.0, 4.0];
        let ranks = rank_transform(&preds);

        // 1.0 is smallest (rank 0), 4.0 is largest (rank 1)
        assert!((ranks[0] - 0.0).abs() < 1e-6); // 1.0 -> rank 0
        assert!((ranks[1] - 2.0 / 3.0).abs() < 1e-6); // 3.0 -> rank 2
        assert!((ranks[2] - 1.0 / 3.0).abs() < 1e-6); // 2.0 -> rank 1
        assert!((ranks[3] - 1.0).abs() < 1e-6); // 4.0 -> rank 3
    }

    #[test]
    fn test_rank_transform_ties() {
        let preds = vec![1.0, 2.0, 2.0, 3.0];
        let ranks = rank_transform(&preds);

        // Ties at 2.0 should have same rank (average of positions 1 and 2)
        assert!((ranks[1] - ranks[2]).abs() < 1e-6);
    }

    #[test]
    fn test_rank_transform_empty() {
        let empty: Vec<f32> = vec![];
        let ranks = rank_transform(&empty);
        assert!(ranks.is_empty());
    }

    #[test]
    fn test_simple_average_stacker() {
        let mut stacker = SimpleAverageStacker::new();

        let oof = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let targets = vec![1.5, 2.5, 3.5];

        stacker.fit(&oof, &targets);

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let combined = stacker.combine(&predictions);

        assert_eq!(combined.len(), 3);
        assert!((combined[0] - 1.5).abs() < 1e-6);
        assert!((combined[1] - 2.5).abs() < 1e-6);
        assert!((combined[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_stacker_basic() {
        let config = StackingConfig::new().with_alpha(0.1);
        let mut stacker = RidgeStacker::new(config);

        // Two models with slightly different predictions
        let oof = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.1, 2.1, 3.1, 4.1, 5.1]];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        stacker.fit(&oof, &targets);

        assert!(stacker.is_fitted());
        assert!(stacker.weights().is_some());
        assert_eq!(stacker.weights().unwrap().len(), 2);
    }

    #[test]
    fn test_solve_positive_definite() {
        // Simple 2x2 system: [[2, 1], [1, 2]] * x = [1, 2]
        // Solution: x = [0, 1]
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let b = vec![1.0, 2.0];
        let x = solve_positive_definite(&a, &b);

        assert_eq!(x.len(), 2);
        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - 1.0).abs() < 1e-6);
    }
}
