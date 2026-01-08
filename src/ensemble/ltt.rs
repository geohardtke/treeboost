//! LTT (LinearThenTree) Ensemble
//!
//! Combines a single linear model with an ensemble of GBDTs trained on residuals.
//! The linear model captures global trends while the GBDT ensemble captures
//! complex residual patterns with variance reduction via multi-seed + stacking.

use crate::dataset::feature_extractor::FeatureExtractor;
use crate::dataset::BinnedDataset;
use crate::ensemble::StackedEnsemble;
use crate::learner::{LinearBooster, LinearConfig, WeakLearner};
use crate::utils::features::extract_selected_features;

/// Statistics about an LTT ensemble
#[derive(Debug, Clone)]
pub struct LttEnsembleStats {
    /// Number of GBDT members in the ensemble
    pub n_gbdt_members: usize,
    /// Stacking weights for GBDT ensemble
    pub gbdt_weights: Option<Vec<f32>>,
    /// GBDT ensemble OOF metric (on residuals)
    pub gbdt_oof_metric: f32,
    /// Linear model shrinkage factor
    pub linear_shrinkage: f32,
}

/// LTT Ensemble: Linear model + stacked GBDT ensemble on residuals
///
/// Architecture:
/// - Single LinearBooster trained once (deterministic)
/// - StackedEnsemble of GBDTs trained on residuals (multi-seed + ridge stacking)
///
/// Prediction:
/// ```text
/// pred = base_prediction + shrinkage * linear_pred + gbdt_ensemble_pred
/// ```
pub struct LttEnsemble {
    /// Trained linear model
    linear_booster: LinearBooster,
    /// Ensemble of GBDTs trained on residuals
    gbdt_ensemble: StackedEnsemble,
    /// Base prediction (mean of targets)
    base_prediction: f32,
    /// Linear config (for shrinkage_factor)
    linear_config: LinearConfig,
    /// Number of features for linear model
    num_linear_features: usize,
    /// Feature indices for linear model (if subset)
    linear_feature_indices: Option<Vec<usize>>,
    /// Feature extractor for inference
    feature_extractor: Option<FeatureExtractor>,
}

impl LttEnsemble {
    /// Create a new LTT ensemble
    pub fn new(
        linear_booster: LinearBooster,
        gbdt_ensemble: StackedEnsemble,
        base_prediction: f32,
        linear_config: LinearConfig,
        num_linear_features: usize,
        linear_feature_indices: Option<Vec<usize>>,
        feature_extractor: Option<FeatureExtractor>,
    ) -> Self {
        Self {
            linear_booster,
            gbdt_ensemble,
            base_prediction,
            linear_config,
            num_linear_features,
            linear_feature_indices,
            feature_extractor,
        }
    }

    /// Predict using binned features only (lossy for linear component)
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        // Extract raw features from bins (lossy approximation)
        let raw_features = Self::extract_raw_features_from_bins(dataset);

        self.predict_internal(dataset, &raw_features, num_rows, num_features)
    }

    /// Predict with raw features (accurate for linear component)
    pub fn predict_with_raw_features(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
    ) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_raw_features = if num_rows > 0 {
            raw_features.len() / num_rows
        } else {
            dataset.num_features()
        };

        self.predict_internal(dataset, raw_features, num_rows, num_raw_features)
    }

    fn predict_internal(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
        num_rows: usize,
        num_raw_features: usize,
    ) -> Vec<f32> {
        // Start with base prediction
        let mut predictions = vec![self.base_prediction; num_rows];

        // Add linear predictions (with shrinkage)
        let linear_features = extract_selected_features(
            raw_features,
            num_rows,
            num_raw_features,
            self.linear_feature_indices.as_deref(),
        );

        let shrinkage = self.linear_config.shrinkage_factor;
        for i in 0..num_rows {
            let linear_pred = self.linear_booster.predict_row(
                &linear_features,
                self.num_linear_features,
                i,
            );
            predictions[i] += shrinkage * linear_pred;
        }

        // Add GBDT ensemble predictions
        let gbdt_preds = self.gbdt_ensemble.predict(dataset);
        for i in 0..num_rows {
            predictions[i] += gbdt_preds[i];
        }

        predictions
    }

    /// Extract raw features from binned data (lossy approximation)
    ///
    /// This is a simple wrapper around `BinnedDataset::extract_raw_features_from_bins()`
    /// for convenience. Uses bin-center approximation - see BinnedDataset docs for details.
    fn extract_raw_features_from_bins(dataset: &BinnedDataset) -> Vec<f32> {
        dataset.extract_raw_features_from_bins()
    }

    /// Get ensemble statistics
    pub fn stats(&self) -> LttEnsembleStats {
        LttEnsembleStats {
            n_gbdt_members: self.gbdt_ensemble.n_members(),
            gbdt_weights: self.gbdt_ensemble.weights().map(|w| w.to_vec()),
            gbdt_oof_metric: self.gbdt_ensemble.oof_metric(),
            linear_shrinkage: self.linear_config.shrinkage_factor,
        }
    }

    /// Get the feature extractor (if set)
    pub fn feature_extractor(&self) -> Option<&FeatureExtractor> {
        self.feature_extractor.as_ref()
    }

    /// Get reference to the linear booster
    pub fn linear_booster(&self) -> &LinearBooster {
        &self.linear_booster
    }

    /// Get reference to the GBDT ensemble
    pub fn gbdt_ensemble(&self) -> &StackedEnsemble {
        &self.gbdt_ensemble
    }

    /// Get the base prediction
    pub fn base_prediction(&self) -> f32 {
        self.base_prediction
    }

    /// Predict using only linear component (base + shrinkage * linear)
    ///
    /// This is useful for comparing the contribution of the linear model
    /// vs the GBDT ensemble. Does NOT include GBDT predictions.
    pub fn predict_linear_only(&self, dataset: &BinnedDataset, raw_features: &[f32]) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_raw_features = if num_rows > 0 {
            raw_features.len() / num_rows
        } else {
            dataset.num_features()
        };

        let linear_features = extract_selected_features(
            raw_features,
            num_rows,
            num_raw_features,
            self.linear_feature_indices.as_deref(),
        );

        let mut predictions = vec![self.base_prediction; num_rows];
        let shrinkage = self.linear_config.shrinkage_factor;

        for i in 0..num_rows {
            let linear_pred = self.linear_booster.predict_row(
                &linear_features,
                self.num_linear_features,
                i,
            );
            predictions[i] += shrinkage * linear_pred;
        }

        predictions
    }
}
