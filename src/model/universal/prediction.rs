//! Prediction methods for UniversalModel
//!
//! This module contains all prediction-related methods:
//! - predict() and predict_with_raw_features()
//! - predict_row(), predict_linear_only()
//! - Classification: predict_proba(), predict_class(), multiclass variants
//! - Raw (unbinned) predictions: predict_raw(), predict_proba_raw(), etc.
//! - Conformal prediction: predict_with_intervals(), conformal_quantile()
//! - Feature importance

use crate::dataset::BinnedDataset;
use crate::learner::WeakLearner;
use crate::utils::features::extract_selected_features;
use crate::Result;

use super::{BoostingMode, UniversalModel};

impl UniversalModel {
    // =========================================================================
    // Batch Prediction
    // =========================================================================

    /// Predict for all rows in dataset
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Check for ensemble first, then single model
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    self.predict_ensemble(dataset, ensemble)
                } else {
                    // Delegate to single GBDTModel
                    self.gbdt_model
                        .as_ref()
                        .map(|m| m.predict(dataset))
                        .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
                }
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution + GBDTModel (trained on residuals)
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution with shrinkage
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features(dataset);
                    let linear_preds = linear.predict_batch(&raw_features, self.num_features);
                    self.apply_linear_shrinkage(&mut predictions, &linear_preds);
                }

                // Add tree contribution (either ensemble or single GBDT, trained on residuals)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                // gbdt.predict() returns (gbdt_base + tree_sum), but we already have ltt_base
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    let tree_preds = self.predict_ensemble(dataset, ensemble);
                    // Ensemble already accounts for base predictions internally
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i];
                    }
                } else if let Some(ref gbdt) = self.gbdt_model {
                    let tree_preds = gbdt.predict(dataset);
                    let gbdt_base = gbdt.base_prediction();
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i] - gbdt_base;
                    }
                }

                predictions
            }
            BoostingMode::RandomForest => {
                // RandomForest: Each tree is independent, predictions averaged
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                if !self.trees.is_empty() {
                    let mut tree_sum = vec![0.0f32; num_rows];
                    for tree in &self.trees {
                        tree.predict_batch_add(dataset, &mut tree_sum);
                    }
                    let scale = 1.0 / self.trees.len() as f32;
                    for i in 0..num_rows {
                        predictions[i] += tree_sum[i] * scale;
                    }
                }

                predictions
            }
        }
    }

    /// Predict for all rows using raw features (recommended for LinearThenTree)
    ///
    /// For LinearThenTree mode, using raw (unbinned) features for the linear model
    /// gives significantly better accuracy than the bin-center approximation.
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset (used for tree predictions)
    /// * `raw_features` - Original features, row-major f32 array (num_rows * num_features)
    ///
    /// # Note
    /// For PureTree and RandomForest, `raw_features` is ignored (trees use binned data).
    pub fn predict_with_raw_features(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
    ) -> Vec<f32> {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Check for ensemble first, then single model (raw features not used for trees)
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    self.predict_ensemble(dataset, ensemble)
                } else {
                    self.gbdt_model
                        .as_ref()
                        .map(|m| m.predict(dataset))
                        .unwrap_or_else(|| vec![0.0; dataset.num_rows()])
                }
            }
            BoostingMode::LinearThenTree => {
                // Linear contribution uses raw features + GBDTModel uses binned
                let num_rows = dataset.num_rows();
                let mut predictions = vec![self.base_prediction; num_rows];

                // Add linear contribution using raw features (possibly selected subset)
                if let Some(ref linear) = self.linear_booster {
                    // Calculate actual number of features in raw_features array
                    // (may differ from self.num_features if FeatureExtractor was used)
                    let num_raw_features = if num_rows > 0 {
                        raw_features.len() / num_rows
                    } else {
                        self.num_features
                    };

                    let num_lin_feats = self.num_linear_features.unwrap_or(num_raw_features);

                    // Extract selected features if indices are specified
                    let linear_features = extract_selected_features(
                        raw_features,
                        num_rows,
                        num_raw_features,
                        self.linear_feature_indices.as_deref(),
                    );

                    let linear_preds = linear.predict_batch(&linear_features, num_lin_feats);

                    // Apply shrinkage factor (ensemble weighting)
                    self.apply_linear_shrinkage(&mut predictions, &linear_preds);
                }

                // Add tree contribution (either ensemble or single GBDT, trees use binned data)
                // IMPORTANT: Subtract gbdt's base_prediction to avoid double-counting
                if let Some(ref ensemble) = self.gbdt_ensemble {
                    let tree_preds = self.predict_ensemble(dataset, ensemble);
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i];
                    }
                } else if let Some(ref gbdt) = self.gbdt_model {
                    let tree_preds = gbdt.predict(dataset);
                    let gbdt_base = gbdt.base_prediction();
                    for i in 0..num_rows {
                        predictions[i] += tree_preds[i] - gbdt_base;
                    }
                }

                predictions
            }
            BoostingMode::RandomForest => {
                // RandomForest: trees don't use raw features
                self.predict(dataset)
            }
        }
    }

    /// Predict using only linear component (LinearThenTree mode only)
    ///
    /// Returns predictions from base + linear model, without tree contribution.
    /// Useful for comparing linear-only vs full LinearThenTree performance.
    pub fn predict_linear_only(
        &self,
        dataset: &BinnedDataset,
        raw_features: &[f32],
    ) -> Result<Vec<f32>> {
        if !matches!(self.config.mode, BoostingMode::LinearThenTree) {
            return Err(crate::TreeBoostError::Config(
                "predict_linear_only() only available for LinearThenTree mode".to_string(),
            ));
        }

        let num_rows = dataset.num_rows();
        let mut predictions = vec![self.base_prediction; num_rows];

        // Add only linear contribution (no trees)
        if let Some(ref linear) = self.linear_booster {
            let num_raw_features = if num_rows > 0 {
                raw_features.len() / num_rows
            } else {
                self.num_features
            };

            let num_lin_feats = self.num_linear_features.unwrap_or(num_raw_features);

            // Extract selected features if indices are specified
            let linear_features = extract_selected_features(
                raw_features,
                num_rows,
                num_raw_features,
                self.linear_feature_indices.as_deref(),
            );

            let linear_preds = linear.predict_batch(&linear_features, num_lin_feats);

            // NOTE: For linear-only, we return the full linear model prediction
            // WITHOUT the boosting shrinkage (shrinkage_factor). This shows the
            // standalone linear model's performance for fair comparison.
            for i in 0..num_rows.min(linear_preds.len()) {
                predictions[i] += linear_preds[i];
            }
        }

        Ok(predictions)
    }

    // =========================================================================
    // Single-Row Prediction
    // =========================================================================

    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        match self.config.mode {
            BoostingMode::PureTree => {
                // Delegate to GBDTModel
                self.gbdt_model
                    .as_ref()
                    .map(|m| m.predict(dataset)[row_idx]) // GBDTModel doesn't have predict_row
                    .unwrap_or(0.0)
            }
            BoostingMode::LinearThenTree => {
                let mut pred = self.base_prediction;

                // Add linear contribution with shrinkage
                if let Some(ref linear) = self.linear_booster {
                    let raw_features = Self::extract_raw_features_row(dataset, row_idx);
                    let linear_pred = linear.predict_row(&raw_features, self.num_features, 0);
                    pred += self.apply_linear_shrinkage_scalar(linear_pred);
                }

                // Add tree contribution from GBDTModel
                // Subtract gbdt's base_prediction to avoid double-counting
                if let Some(ref gbdt) = self.gbdt_model {
                    pred += gbdt.predict(dataset)[row_idx] - gbdt.base_prediction();
                }

                pred
            }
            BoostingMode::RandomForest => {
                let mut pred = self.base_prediction;

                if !self.trees.is_empty() {
                    let tree_sum: f32 = self
                        .trees
                        .iter()
                        .map(|t| t.predict_row(dataset, row_idx))
                        .sum();
                    pred += tree_sum / self.trees.len() as f32;
                }

                pred
            }
        }
    }

    // =========================================================================
    // Conformal Prediction (PureTree only)
    // =========================================================================

    /// Predict with conformal prediction intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds) for uncertainty quantification.
    ///
    /// # Note
    /// Only supported in PureTree mode (delegates to GBDTModel).
    /// LinearThenTree and RandomForest modes return an error.
    pub fn predict_with_intervals(
        &self,
        dataset: &BinnedDataset,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_with_intervals(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            BoostingMode::LinearThenTree | BoostingMode::RandomForest => {
                Err(crate::TreeBoostError::Config(
                    "Conformal prediction only supported in PureTree mode".to_string(),
                ))
            }
        }
    }

    /// Get the calibrated conformal quantile (if available)
    pub fn conformal_quantile(&self) -> Option<f32> {
        self.gbdt_model
            .as_ref()
            .and_then(|m| m.conformal_quantile())
    }

    // =========================================================================
    // Classification (PureTree only)
    // =========================================================================

    /// Binary classification: predict probabilities
    ///
    /// Returns probabilities in [0, 1] for binary classification.
    /// Requires model trained with binary log loss.
    pub fn predict_proba(&self, dataset: &BinnedDataset) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Classification only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification: predict classes
    ///
    /// Returns 0 or 1 based on threshold (default: 0.5).
    pub fn predict_class(&self, dataset: &BinnedDataset, threshold: f32) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class(dataset, threshold)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Classification only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class classification: predict probabilities for each class
    ///
    /// Returns Vec<Vec<f32>> where each inner vec contains probabilities for all classes.
    pub fn predict_proba_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class classification: predict class labels
    pub fn predict_class_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class classification: predict raw logits
    pub fn predict_raw_multiclass(&self, dataset: &BinnedDataset) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_multiclass(dataset)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Multi-class only supported in PureTree mode".to_string(),
            )),
        }
    }

    // =========================================================================
    // Feature Importance
    // =========================================================================

    /// Get feature importance scores
    ///
    /// Returns importance scores for each feature based on gain/split frequency.
    /// For PureTree/LinearThenTree, delegates to GBDTModel.
    /// For RandomForest, computes importance from split frequencies across all trees.
    pub fn feature_importance(&self) -> Vec<f32> {
        use crate::tree::NodeType;

        if let Some(ref gbdt) = self.gbdt_model {
            gbdt.feature_importance()
        } else if !self.trees.is_empty() {
            // RandomForest: count split frequencies per feature across all trees
            let mut importances = vec![0.0f32; self.num_features];
            for tree in &self.trees {
                for node in tree.nodes() {
                    if let NodeType::Internal { feature_idx, .. } = node.node_type {
                        if feature_idx < importances.len() {
                            // Count splits per feature (weighted by samples in node)
                            importances[feature_idx] += node.num_samples as f32;
                        }
                    }
                }
            }
            // Normalize
            let total: f32 = importances.iter().sum();
            if total > 0.0 {
                for imp in &mut importances {
                    *imp /= total;
                }
            }
            importances
        } else {
            vec![0.0; self.num_features]
        }
    }

    // =========================================================================
    // Raw Prediction (from unbinned features)
    // =========================================================================

    /// Predict from raw (unbinned) feature values
    ///
    /// Useful when you have raw feature values and don't want to create a BinnedDataset.
    /// Only supported in PureTree mode.
    ///
    /// # Arguments
    /// * `features` - Raw feature values for one or more rows, flattened row-major
    pub fn predict_raw(&self, features: &[f64]) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Predict with intervals from raw (unbinned) feature values
    pub fn predict_raw_with_intervals(
        &self,
        features: &[f64],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_with_intervals(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification probability from raw features
    pub fn predict_proba_raw(&self, features: &[f64]) -> Result<Vec<f32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Binary classification class from raw features
    pub fn predict_class_raw(&self, features: &[f64], threshold: f32) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_raw(features, threshold)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class probabilities from raw features
    pub fn predict_proba_multiclass_raw(&self, features: &[f64]) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_proba_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class class labels from raw features
    pub fn predict_class_multiclass_raw(&self, features: &[f64]) -> Result<Vec<u32>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_class_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }

    /// Multi-class raw logits from raw features
    pub fn predict_raw_multiclass_raw(&self, features: &[f64]) -> Result<Vec<Vec<f32>>> {
        match self.config.mode {
            BoostingMode::PureTree => self
                .gbdt_model
                .as_ref()
                .map(|m| Ok(m.predict_raw_multiclass_raw(features)))
                .unwrap_or_else(|| {
                    Err(crate::TreeBoostError::Config(
                        "No model trained".to_string(),
                    ))
                }),
            _ => Err(crate::TreeBoostError::Config(
                "Raw prediction only supported in PureTree mode".to_string(),
            )),
        }
    }
}
