//! WeakLearner trait definition
//!
//! The core abstraction that enables gradient boosting with different base learners.

use crate::Result;

/// Trait for weak learners in gradient boosting
///
/// A weak learner fits to the negative gradient of the loss function.
/// This trait abstracts over trees (GBDTs), linear models, and hybrids.
///
/// # Design Notes
///
/// - Uses raw `&[f32]` arrays instead of `BinnedDataset` for flexibility
/// - Linear models need raw values, not binned data
/// - Trees can work with either (binned is faster)
///
/// # Example Implementation
///
/// ```ignore
/// impl WeakLearner for LinearBooster {
///     fn fit_on_gradients(
///         &mut self,
///         features: &[f32],
///         num_features: usize,
///         gradients: &[f32],
///         hessians: &[f32],
///     ) -> Result<()> {
///         // Coordinate descent on gradients
///         self.coordinate_descent(features, num_features, gradients, hessians)
///     }
///
///     fn predict_batch(&self, features: &[f32], num_features: usize) -> Vec<f32> {
///         // w · x + b for each row
///     }
/// }
/// ```
pub trait WeakLearner: Send + Sync {
    /// Fit the learner on gradients and hessians
    ///
    /// # Arguments
    /// - `features`: Row-major feature matrix (num_rows × num_features)
    /// - `num_features`: Number of features per row
    /// - `gradients`: Negative gradient of loss (one per row)
    /// - `hessians`: Second derivative of loss (one per row)
    fn fit_on_gradients(
        &mut self,
        features: &[f32],
        num_features: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<()>;

    /// Predict for all rows
    ///
    /// # Arguments
    /// - `features`: Row-major feature matrix
    /// - `num_features`: Number of features per row
    ///
    /// # Returns
    /// Vector of predictions (one per row)
    fn predict_batch(&self, features: &[f32], num_features: usize) -> Vec<f32>;

    /// Predict for a single row
    ///
    /// Default implementation extracts row and calls predict_batch.
    /// Override for better performance.
    fn predict_row(&self, features: &[f32], num_features: usize, row_idx: usize) -> f32 {
        let start = row_idx * num_features;
        let row = &features[start..start + num_features];
        // Create a single-row slice for prediction
        self.predict_batch(row, num_features)[0]
    }

    /// Number of learnable parameters
    ///
    /// Used for regularization scaling and model complexity estimation.
    fn num_params(&self) -> usize;

    /// Reset the learner to initial state
    fn reset(&mut self);
}
