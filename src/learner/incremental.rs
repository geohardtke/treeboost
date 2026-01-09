//! Incremental learning support for weak learners
//!
//! Provides traits and utilities for incrementally updating learners
//! with new data batches, enabling online learning and model updates.
//!
//! # Supported Learners
//!
//! | Learner | Incremental Support | Notes |
//! |---------|---------------------|-------|
//! | LinearBooster | Full | Warm start continues CD from current weights |
//! | TreeBooster | Partial | Trees are additive; new trees trained on residuals |
//! | LinearTreeBooster | Not Supported | Complex leaf structure requires refit |
//!
//! # Example
//!
//! ```ignore
//! use treeboost::learner::{LinearBooster, LinearConfig};
//! use treeboost::learner::incremental::IncrementalLearner;
//!
//! let mut booster = LinearBooster::new(5, LinearConfig::default());
//!
//! // Initial training
//! booster.fit_on_gradients(&features_a, 5, &grads_a, &hess_a)?;
//!
//! // Warm start with new data (continues from current weights)
//! booster.warm_fit(&features_b, 5, &grads_b, &hess_b)?;
//!
//! // Check total iterations across all fits
//! println!("Total iterations: {}", booster.iterations_completed());
//! ```

use crate::Result;

/// Trait for learners that support incremental fitting (warm start)
///
/// Learners implementing this trait can continue training from their current
/// state instead of starting from scratch. This enables:
/// - Online learning (streaming data)
/// - Model updates (new batches without full retrain)
/// - Transfer learning (start from pretrained weights)
pub trait IncrementalLearner {
    /// Continue training from current state (warm start)
    ///
    /// Unlike `fit_on_gradients()` which resets internal state,
    /// `warm_fit()` continues optimization from current weights.
    ///
    /// # Arguments
    /// * `features` - Row-major flat array (row0_feat0, row0_feat1, ..., row1_feat0, ...)
    /// * `num_features` - Number of features per row
    /// * `gradients` - Negative gradient of loss
    /// * `hessians` - Second derivative of loss
    ///
    /// # Notes
    /// - Internal scaler parameters (mean/std) are frozen from first fit
    /// - Weights continue from current values
    /// - Convergence may be faster than cold start if data is similar
    fn warm_fit(
        &mut self,
        features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<()>;

    /// Get total number of optimization iterations completed
    ///
    /// Accumulates across all `fit_on_gradients` and `warm_fit` calls.
    /// Useful for tracking convergence and learning curves.
    fn iterations_completed(&self) -> usize;

    /// Reset iteration counter
    ///
    /// Call this when starting a completely new training run.
    fn reset_iterations(&mut self);
}

/// Trait for models that support appending new components
///
/// This is the tree-level incremental training interface.
/// Trees are trained on residuals from the existing ensemble,
/// then appended to grow the model.
pub trait AppendableLearner {
    /// Type of component that can be appended (e.g., Tree)
    type Component;

    /// Append new components to the ensemble
    ///
    /// # Arguments
    /// * `components` - New components to add (e.g., trees trained on residuals)
    ///
    /// # Notes
    /// - For multi-class models, components must follow the expected ordering
    /// - Existing components are preserved (append-only)
    fn append(&mut self, components: Vec<Self::Component>);

    /// Get number of components in the ensemble
    fn num_components(&self) -> usize;

    /// Compute residuals from current ensemble predictions
    ///
    /// These residuals become the targets for training new components.
    ///
    /// # Arguments
    /// * `predictions` - Current ensemble predictions
    /// * `targets` - Original target values
    ///
    /// # Returns
    /// Residuals (target - prediction) for each sample
    fn compute_residuals(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
        predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| t - p)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    /// Helper to compute residuals for testing (since trait method can't be called directly)
    fn compute_residuals_test(predictions: &[f32], targets: &[f32]) -> Vec<f32> {
        predictions
            .iter()
            .zip(targets)
            .map(|(p, t)| t - p)
            .collect()
    }

    #[test]
    fn test_compute_residuals() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 2.0];

        let residuals = compute_residuals_test(&predictions, &targets);

        assert_eq!(residuals.len(), 3);
        assert!((residuals[0] - 0.5).abs() < 1e-6);
        assert!((residuals[1] - 0.5).abs() < 1e-6);
        assert!((residuals[2] - (-1.0)).abs() < 1e-6);
    }
}
