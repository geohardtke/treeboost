//! Core traits for ensemble learning

use crate::dataset::BinnedDataset;

/// Trait for models that can participate in an ensemble
///
/// Any model implementing this trait can be used with the ensemble selection
/// and stacking infrastructure. The key requirement is the ability to provide
/// out-of-fold predictions for stacking.
pub trait EnsembleMember: Send + Sync {
    /// Get out-of-fold predictions and corresponding row indices
    ///
    /// Returns `None` if OOF predictions are not available (e.g., model
    /// was trained without K-fold cross-validation).
    ///
    /// # Returns
    /// - `Some((predictions, indices))` where predictions[i] corresponds to row indices[i]
    /// - `None` if OOF predictions are unavailable
    fn oof_predictions(&self) -> Option<(&[f32], &[usize])>;

    /// Predict on binned dataset
    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32>;

    /// Get unique model identifier (typically config hash + seed)
    fn model_id(&self) -> u64;

    /// Get the random seed used for training
    fn seed(&self) -> u64;

    /// Clone into a boxed trait object
    fn clone_boxed(&self) -> Box<dyn EnsembleMember>;
}

/// Trait for stacking/blending strategies
///
/// Stackers combine predictions from multiple base models into a single
/// prediction. The `fit` method learns the combination weights from
/// out-of-fold predictions, and `combine` applies those weights to new data.
pub trait Stacker: Send + Sync {
    /// Fit the stacker on out-of-fold predictions
    ///
    /// # Arguments
    /// * `oof_preds` - Vector of OOF predictions, one per model.
    ///   Each inner vector has length = n_samples.
    /// * `targets` - Ground truth target values
    fn fit(&mut self, oof_preds: &[Vec<f32>], targets: &[f32]);

    /// Combine predictions from member models
    ///
    /// # Arguments
    /// * `predictions` - Vector of predictions, one per model.
    ///   Each inner vector has length = n_samples.
    ///
    /// # Returns
    /// Combined predictions with length = n_samples
    fn combine(&self, predictions: &[Vec<f32>]) -> Vec<f32>;

    /// Get blend weights if applicable
    ///
    /// Returns `None` for stackers that don't use explicit weights
    /// (e.g., median blending).
    fn weights(&self) -> Option<&[f32]>;

    /// Name of the stacking strategy
    fn name(&self) -> &'static str;
}
