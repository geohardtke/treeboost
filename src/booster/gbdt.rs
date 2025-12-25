//! GBDT model and training

use crate::booster::GBDTConfig;
use crate::dataset::BinnedDataset;
use crate::tree::{Tree, TreeGrower};
use crate::{Result, TreeBoostError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

use crate::dataset::FeatureInfo;

/// Trained GBDT model
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct GBDTModel {
    /// Training configuration
    config: GBDTConfig,
    /// Base prediction (initial value)
    base_prediction: f32,
    /// Ensemble of trees
    trees: Vec<Tree>,
    /// Conformal quantile for prediction intervals (if calibrated)
    conformal_q: Option<f32>,
    /// Feature info from training (bin boundaries for consistent prediction)
    feature_info: Vec<FeatureInfo>,
}

impl GBDTModel {
    /// Train a GBDT model
    pub fn train(dataset: &BinnedDataset, config: GBDTConfig) -> Result<Self> {
        config.validate().map_err(TreeBoostError::Config)?;

        let loss_fn = config.loss_type.create();
        let targets = dataset.targets();

        // Split data for conformal prediction if enabled
        let (train_indices, calibration_indices) = if config.calibration_ratio > 0.0 {
            Self::split_for_calibration(dataset.num_rows(), config.calibration_ratio)
        } else {
            ((0..dataset.num_rows()).collect(), Vec::new())
        };

        // Compute base prediction
        let train_targets: Vec<f32> = train_indices.iter().map(|&i| targets[i]).collect();
        let base_prediction = loss_fn.initial_prediction(&train_targets);

        // Initialize predictions
        let mut predictions = vec![base_prediction; dataset.num_rows()];

        // Gradient and hessian buffers
        let mut gradients = vec![0.0f32; dataset.num_rows()];
        let mut hessians = vec![0.0f32; dataset.num_rows()];

        // Create tree grower
        let tree_grower = TreeGrower::new()
            .with_max_depth(config.max_depth)
            .with_max_leaves(config.max_leaves)
            .with_lambda(config.lambda)
            .with_min_samples_leaf(config.min_samples_leaf)
            .with_min_hessian_leaf(config.min_hessian_leaf)
            .with_entropy_weight(config.entropy_weight)
            .with_min_gain(config.min_gain)
            .with_learning_rate(config.learning_rate);

        let mut trees = Vec::with_capacity(config.num_rounds);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _round in 0..config.num_rounds {
            // Compute gradients and hessians for training data
            for &idx in &train_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }

            // Subsample rows if configured
            let sample_indices: Vec<usize> = if config.subsample < 1.0 {
                let n_samples = ((train_indices.len() as f32) * config.subsample).ceil() as usize;
                let mut indices = train_indices.clone();
                indices.shuffle(&mut rng);
                indices.truncate(n_samples);
                indices
            } else {
                train_indices.clone()
            };

            // Grow tree using all indices (subsampling prepared but not currently used)
            // Note: sample_indices are prepared via config.subsample but tree_grower.grow()
            // currently processes all rows. Full row subsampling would require changes to
            // TreeGrower to accept and filter index lists during histogram construction.
            let _ = &sample_indices;
            let tree = tree_grower.grow(dataset, &gradients, &hessians);

            // Update predictions for all rows
            for row_idx in 0..dataset.num_rows() {
                predictions[row_idx] += tree.predict_row(dataset, row_idx);
            }

            trees.push(tree);
        }

        // Compute conformal quantile if calibration set exists
        let conformal_q = if !calibration_indices.is_empty() {
            let calib_residuals: Vec<f32> = calibration_indices
                .iter()
                .map(|&idx| (targets[idx] - predictions[idx]).abs())
                .collect();

            Some(Self::compute_quantile(&calib_residuals, config.conformal_quantile))
        } else {
            None
        };

        Ok(Self {
            config,
            base_prediction,
            trees,
            conformal_q,
            feature_info: dataset.all_feature_info().to_vec(),
        })
    }

    /// Split data indices for conformal calibration
    fn split_for_calibration(num_rows: usize, ratio: f32) -> (Vec<usize>, Vec<usize>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut indices: Vec<usize> = (0..num_rows).collect();
        indices.shuffle(&mut rng);

        let n_calibration = ((num_rows as f32) * ratio).ceil() as usize;
        let calibration: Vec<usize> = indices.drain(..n_calibration).collect();

        (indices, calibration)
    }

    /// Compute quantile of a sorted slice
    fn compute_quantile(values: &[f32], q: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((sorted.len() as f32) * q).ceil() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Predict for a single row
    pub fn predict_row(&self, dataset: &BinnedDataset, row_idx: usize) -> f32 {
        let mut pred = self.base_prediction;
        for tree in &self.trees {
            pred += tree.predict_row(dataset, row_idx);
        }
        pred
    }

    /// Predict for all rows (optimized with row-wise bin caching)
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        self.predict_parallel(dataset)
    }

    /// Single-threaded prediction (for small datasets or when parallelism not desired)
    pub fn predict_sequential(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();
        let num_features = dataset.num_features();

        let mut predictions = Vec::with_capacity(num_rows);
        let mut row_bins = vec![0u8; num_features];

        for row_idx in 0..num_rows {
            // Cache all bins for this row
            for f in 0..num_features {
                row_bins[f] = dataset.get_bin(row_idx, f);
            }

            // Traverse all trees with cached bins
            let mut pred = self.base_prediction;
            for tree in &self.trees {
                pred += tree.predict(|f| row_bins[f]);
            }
            predictions.push(pred);
        }

        predictions
    }

    /// Parallel prediction using Rayon
    pub fn predict_parallel(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_features = dataset.num_features();
        let base = self.base_prediction;

        (0..dataset.num_rows())
            .into_par_iter()
            .map(|row_idx| {
                // Cache bins for this row
                let row_bins: Vec<u8> = (0..num_features)
                    .map(|f| dataset.get_bin(row_idx, f))
                    .collect();

                // Traverse all trees
                let mut pred = base;
                for tree in &self.trees {
                    pred += tree.predict(|f| row_bins[f]);
                }
                pred
            })
            .collect()
    }

    /// Predict with conformal intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds)
    pub fn predict_with_intervals(&self, dataset: &BinnedDataset) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let predictions = self.predict(dataset);

        let q = self.conformal_q.unwrap_or(0.0);
        let lower: Vec<f32> = predictions.iter().map(|&p| p - q).collect();
        let upper: Vec<f32> = predictions.iter().map(|&p| p + q).collect();

        (predictions, lower, upper)
    }

    /// Get number of trees
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get configuration
    pub fn config(&self) -> &GBDTConfig {
        &self.config
    }

    /// Get base prediction
    pub fn base_prediction(&self) -> f32 {
        self.base_prediction
    }

    /// Get conformal quantile (if calibrated)
    pub fn conformal_quantile(&self) -> Option<f32> {
        self.conformal_q
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

    /// Compute feature importances (gain-based)
    pub fn feature_importances(&self, num_features: usize) -> Vec<f32> {
        let mut importances = vec![0.0f32; num_features];

        for tree in &self.trees {
            for (_, node) in tree.internal_nodes() {
                // Safe to unwrap: internal_nodes() filters to only internal nodes
                let (feature_idx, _, _, _) = node.split_info().unwrap();
                // Use hessian as importance weight (proxy for sample weight)
                importances[feature_idx] += node.sum_hessians;
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_regression_dataset(num_rows: usize, noise: f32) -> BinnedDataset {
        let num_features = 3;

        // Generate features
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * (f + 1) * 17) % 256) as u8);
            }
        }

        // Generate targets with some pattern
        let targets: Vec<f32> = (0..num_rows)
            .map(|i| {
                let f0 = features[i] as f32 / 255.0;
                let f1 = features[num_rows + i] as f32 / 255.0;
                f0 * 10.0 + f1 * 5.0 + noise * (i as f32 % 10.0 - 5.0) / 5.0
            })
            .collect();

        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("feature_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_train_basic() {
        let dataset = create_regression_dataset(500, 0.1);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(3)
            .with_learning_rate(0.1);

        let model = GBDTModel::train(&dataset, config).unwrap();

        assert_eq!(model.num_trees(), 10);

        // Test prediction
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 500);
    }

    #[test]
    fn test_train_with_pseudo_huber() {
        let dataset = create_regression_dataset(500, 1.0);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_pseudo_huber_loss(1.0);

        let model = GBDTModel::train(&dataset, config).unwrap();
        assert_eq!(model.num_trees(), 10);
    }

    #[test]
    fn test_train_with_conformal() {
        let dataset = create_regression_dataset(500, 0.5);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_conformal(0.2, 0.9);

        let model = GBDTModel::train(&dataset, config).unwrap();

        assert!(model.conformal_quantile().is_some());
        assert!(model.conformal_quantile().unwrap() > 0.0);

        // Test interval prediction
        let (preds, lower, upper) = model.predict_with_intervals(&dataset);
        assert_eq!(preds.len(), dataset.num_rows());
        assert_eq!(lower.len(), dataset.num_rows());
        assert_eq!(upper.len(), dataset.num_rows());

        // Intervals should be symmetric
        for i in 0..preds.len() {
            assert!((preds[i] - lower[i] - (upper[i] - preds[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn test_feature_importances() {
        let dataset = create_regression_dataset(500, 0.1);

        let config = GBDTConfig::new()
            .with_num_rounds(20)
            .with_max_depth(4);

        let model = GBDTModel::train(&dataset, config).unwrap();
        let importances = model.feature_importances(3);

        assert_eq!(importances.len(), 3);

        // Importances should sum to ~1 (normalized)
        let total: f32 = importances.iter().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}
