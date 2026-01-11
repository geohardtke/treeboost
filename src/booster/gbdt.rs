//! GBDT model and core types
//!
//! Provides the main `GBDTModel` struct and basic accessor methods.
//! Training, prediction, and analysis are delegated to submodules:
//! - `training`: Training implementations (high-level and low-level APIs)
//! - `prediction`: Prediction implementations (inference methods)
//! - `analysis`: Feature importance and model analysis
//! - `conformal`: Conformal prediction intervals

use std::collections::HashMap;
use std::path::Path;

use crate::booster::GBDTConfig;
use crate::dataset::{BinnedDataset, ColumnPermutation, FeatureInfo};
use crate::tree::Tree;
use crate::tuner::{ParamValue, TunableModel};
use crate::Result;
use rkyv::{Archive, Deserialize, Serialize};

use crate::booster::OutputType;

/// Trained GBDT model
///
/// This struct contains the trained ensemble of trees and associated metadata.
/// Training and prediction methods are implemented via impl blocks in separate modules:
/// - Training: see `training` module
/// - Prediction: see `prediction` module
/// - Analysis: see `analysis` module
/// - Conformal intervals: see `conformal` module
#[derive(Debug, Clone, Archive, Serialize, Deserialize, serde::Serialize, serde::Deserialize)]
pub struct GBDTModel {
    /// Training configuration
    pub(super) config: GBDTConfig,

    /// Base predictions (initial values) per output dimension
    ///
    /// - Regression/Binary: `[base_value]` (length 1)
    /// - Multi-class: `[base_class0, base_class1, ..., base_classK]` (length K)
    /// - Multi-label: `[base_label0, base_label1, ..., base_labelN]` (length N)
    pub(super) base_predictions: Vec<f32>,

    /// Ensemble of trees
    ///
    /// ## Storage Order
    ///
    /// **Regression/Binary**: One tree per round, stored sequentially.
    /// - `trees[round]` = tree for round `round`
    ///
    /// **Multi-class (K classes)**: K trees per round (one per class), stored round-major.
    /// - `trees[round * K + class_idx]` = tree for round `round`, class `class_idx`
    /// - Example with 3 classes, 2 rounds: `[r0_c0, r0_c1, r0_c2, r1_c0, r1_c1, r1_c2]`
    ///
    /// **Multi-label**: One tree per round with vector leaf values.
    /// - `trees[round]` = tree for round `round` (leaf values are vectors of length N)
    ///
    /// Total trees = `num_rounds` (regression/binary/multi-label) or `num_rounds * K` (multi-class)
    pub(super) trees: Vec<Tree>,

    /// Number of output dimensions
    ///
    /// - Regression/Binary: 1
    /// - Multi-class: number of classes (K)
    /// - Multi-label: number of labels (N)
    pub(super) num_outputs: usize,

    /// Output type (determines prediction transformation)
    pub(super) output_type: OutputType,

    /// Conformal quantile for prediction intervals (if calibrated)
    pub(super) conformal_q: Option<f32>,
    /// Feature info from training (bin boundaries for consistent prediction)
    pub(super) feature_info: Vec<FeatureInfo>,
    /// Column permutation for cache-optimized prediction (if enabled)
    pub(super) column_permutation: Option<ColumnPermutation>,

    // --- Legacy fields for backward compatibility (deprecated, will be removed in v2.0) ---
    /// Legacy: Base prediction for regression/binary (now use base_predictions[0])
    #[deprecated(note = "Use base_predictions instead")]
    pub(super) base_prediction: f32,
    /// Legacy: Base predictions for multi-class (now use base_predictions)
    #[deprecated(note = "Use base_predictions instead")]
    pub(super) base_predictions_multiclass: Vec<f32>,
    /// Legacy: Number of classes (now use num_outputs with output_type == MultiClass)
    #[deprecated(note = "Use num_outputs and output_type instead")]
    pub(super) num_classes: usize,
}

impl GBDTModel {
    // =============================================================================
    // Basic Accessors
    // =============================================================================

    /// Get number of trees
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get configuration
    pub fn config(&self) -> &GBDTConfig {
        &self.config
    }

    /// Get base predictions for all output dimensions
    pub fn base_predictions(&self) -> &[f32] {
        &self.base_predictions
    }

    /// Get base prediction (for scalar outputs: regression/binary)
    ///
    /// For multi-output models, use `base_predictions()` instead.
    #[allow(deprecated)]
    pub fn base_prediction(&self) -> f32 {
        // Prefer unified field, fall back to legacy
        if !self.base_predictions.is_empty() {
            self.base_predictions[0]
        } else {
            self.base_prediction
        }
    }

    /// Get trees
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Get feature info (for consistent binning during prediction)
    pub fn feature_info(&self) -> &[FeatureInfo] {
        &self.feature_info
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.feature_info.len()
    }

    /// Get column permutation (if optimized layout was applied)
    pub fn column_permutation(&self) -> Option<&ColumnPermutation> {
        self.column_permutation.as_ref()
    }

    /// Get number of output dimensions
    ///
    /// - Regression/Binary: 1
    /// - Multi-class: number of classes
    /// - Multi-label: number of labels
    #[allow(deprecated)]
    pub fn num_outputs(&self) -> usize {
        if self.num_outputs > 0 {
            self.num_outputs
        } else if self.num_classes > 0 {
            // Legacy multi-class
            self.num_classes
        } else {
            // Legacy scalar
            1
        }
    }

    /// Get output type
    #[allow(deprecated)]
    pub fn output_type(&self) -> OutputType {
        // If new field is set, use it
        if self.num_outputs > 0 {
            self.output_type
        } else if self.num_classes > 0 {
            // Legacy multi-class detection
            OutputType::MultiClass
        } else if matches!(self.config.loss_type, crate::booster::LossType::BinaryLogLoss) {
            OutputType::Binary
        } else {
            OutputType::Regression
        }
    }

    /// Create a GBDTModel from pre-trained components
    ///
    /// This is useful for combining separately trained models (e.g., per-label
    /// models in LinearThenTree multi-output mode).
    ///
    /// # Arguments
    /// * `config` - Training configuration
    /// * `base_predictions` - Base predictions per output dimension
    /// * `trees` - Pre-trained trees
    /// * `num_outputs` - Number of output dimensions
    /// * `output_type` - Type of output (Regression, Binary, MultiClass, MultiLabel)
    /// * `feature_info` - Feature metadata from training
    #[allow(deprecated)]
    pub fn from_components(
        config: GBDTConfig,
        base_predictions: Vec<f32>,
        trees: Vec<crate::tree::Tree>,
        num_outputs: usize,
        output_type: OutputType,
        feature_info: Vec<FeatureInfo>,
    ) -> Self {
        let base_prediction = base_predictions.first().copied().unwrap_or(0.0);

        Self {
            config,
            base_predictions,
            trees,
            num_outputs,
            output_type,
            conformal_q: None,
            feature_info,
            column_permutation: None,
            // Legacy fields
            base_prediction,
            base_predictions_multiclass: Vec::new(),
            num_classes: 0,
        }
    }
}

// =============================================================================
// TunableModel Trait Implementation
// =============================================================================

impl TunableModel for GBDTModel {
    type Config = GBDTConfig;

    fn train(dataset: &BinnedDataset, config: &Self::Config) -> Result<Self> {
        Self::train_binned(dataset, config.clone())
    }

    fn train_with_validation(
        train_data: &BinnedDataset,
        val_data: &BinnedDataset,
        val_targets: &[f32],
        config: &Self::Config,
    ) -> Result<Self> {
        Self::train_binned_with_validation(train_data, val_data, val_targets, config.clone())
    }

    fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        self.predict(dataset)
    }

    fn num_trees(&self) -> usize {
        self.num_trees()
    }

    fn apply_params(config: &mut Self::Config, params: &HashMap<String, ParamValue>) {
        for (key, value) in params {
            // Only apply numeric parameters
            if !value.is_numeric() {
                continue;
            }
            let v = value.as_numeric();

            match key.as_str() {
                "num_rounds" => config.num_rounds = v as usize,
                "max_depth" => config.max_depth = v as usize,
                "max_leaves" => config.max_leaves = v as usize,
                "learning_rate" => config.learning_rate = v,
                "lambda" => config.lambda = v,
                "min_samples_leaf" => config.min_samples_leaf = v as usize,
                "min_hessian_leaf" => config.min_hessian_leaf = v,
                "subsample" => config.subsample = v,
                "colsample" => config.colsample = v,
                "entropy_weight" => config.entropy_weight = v,
                "min_gain" => config.min_gain = v,
                "validation_ratio" => config.validation_ratio = v,
                "early_stopping_rounds" => config.early_stopping_rounds = v as usize,
                "calibration_ratio" => config.calibration_ratio = v,
                "conformal_quantile" => config.conformal_quantile = v,
                "goss_top_rate" => config.goss_top_rate = v,
                "goss_other_rate" => config.goss_other_rate = v,
                _ => {} // Ignore unknown parameters
            }
        }
    }

    fn valid_params() -> &'static [&'static str] {
        &[
            "num_rounds",
            "max_depth",
            "max_leaves",
            "learning_rate",
            "lambda",
            "min_samples_leaf",
            "min_hessian_leaf",
            "subsample",
            "colsample",
            "entropy_weight",
            "min_gain",
            "validation_ratio",
            "early_stopping_rounds",
            "calibration_ratio",
            "conformal_quantile",
            "goss_top_rate",
            "goss_other_rate",
        ]
    }

    fn default_config() -> Self::Config {
        GBDTConfig::new()
    }

    fn is_gpu_config(config: &Self::Config) -> bool {
        // Conservative: treat Auto as GPU since it might resolve to CUDA/WGPU
        matches!(
            config.backend_type,
            crate::backend::BackendType::Cuda
                | crate::backend::BackendType::Wgpu
                | crate::backend::BackendType::Auto
        )
    }

    fn get_learning_rate(config: &Self::Config) -> f32 {
        config.learning_rate
    }

    fn configure_validation(
        config: &mut Self::Config,
        validation_ratio: f32,
        early_stopping_rounds: usize,
    ) {
        config.validation_ratio = validation_ratio;
        config.early_stopping_rounds = early_stopping_rounds;
    }

    fn set_num_rounds(config: &mut Self::Config, num_rounds: usize) {
        config.num_rounds = num_rounds;
    }

    fn save_rkyv(&self, path: &Path) -> Result<()> {
        crate::serialize::save_model(self, path)
    }

    fn save_bincode(&self, path: &Path) -> Result<()> {
        crate::serialize::save_model_bincode(self, path)
    }

    fn supports_conformal() -> bool {
        true
    }

    fn conformal_quantile(&self) -> Option<f32> {
        self.conformal_q
    }

    fn configure_conformal(config: &mut Self::Config, calibration_ratio: f32, quantile: f32) {
        config.calibration_ratio = calibration_ratio;
        config.conformal_quantile = quantile;
    }
}
