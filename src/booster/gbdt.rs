//! GBDT model and training

use crate::booster::GBDTConfig;
use crate::dataset::{BinnedDataset, ColumnPermutation, FeatureInfo};
use crate::tree::{InteractionConstraints, Tree, TreeGrower};
use crate::{Result, TreeBoostError};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

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
    /// Column permutation for cache-optimized prediction (if enabled)
    column_permutation: Option<ColumnPermutation>,
}

impl GBDTModel {
    /// Train a GBDT model
    pub fn train(dataset: &BinnedDataset, config: GBDTConfig) -> Result<Self> {
        config.validate().map_err(TreeBoostError::Config)?;

        let loss_fn = config.loss_type.create();
        let targets = dataset.targets();

        // Split data for validation (early stopping) and conformal calibration
        let (train_indices, validation_indices, calibration_indices) =
            Self::split_for_training(
                dataset.num_rows(),
                config.validation_ratio,
                config.calibration_ratio,
            );

        // Compute base prediction from training data only
        let train_targets: Vec<f32> = train_indices.iter().map(|&i| targets[i]).collect();
        let base_prediction = loss_fn.initial_prediction(&train_targets);

        // Initialize predictions for all rows
        let mut predictions = vec![base_prediction; dataset.num_rows()];

        // Gradient and hessian buffers
        let mut gradients = vec![0.0f32; dataset.num_rows()];
        let mut hessians = vec![0.0f32; dataset.num_rows()];

        // Build interaction constraints from groups
        let interaction_constraints = if config.interaction_groups.is_empty() {
            InteractionConstraints::new()
        } else {
            InteractionConstraints::from_groups(
                config.interaction_groups.clone(),
                dataset.num_features(),
            )
        };

        // Create tree grower
        let tree_grower = TreeGrower::new()
            .with_max_depth(config.max_depth)
            .with_max_leaves(config.max_leaves)
            .with_lambda(config.lambda)
            .with_min_samples_leaf(config.min_samples_leaf)
            .with_min_hessian_leaf(config.min_hessian_leaf)
            .with_entropy_weight(config.entropy_weight)
            .with_min_gain(config.min_gain)
            .with_learning_rate(config.learning_rate)
            .with_colsample(config.colsample)
            .with_monotonic_constraints(config.monotonic_constraints.clone())
            .with_interaction_constraints(interaction_constraints);

        let mut trees = Vec::with_capacity(config.num_rounds);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Early stopping state
        let early_stopping_enabled = config.early_stopping_rounds > 0 && !validation_indices.is_empty();
        let mut best_val_loss = f32::MAX;
        let mut rounds_without_improvement = 0;
        let mut best_num_trees = 0;

        for _round in 0..config.num_rounds {
            // Compute gradients and hessians for training data only
            for &idx in &train_indices {
                let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                gradients[idx] = g;
                hessians[idx] = h;
            }

            // Subsample rows if configured (Stochastic Gradient Boosting)
            let sample_indices: Vec<usize> = if config.subsample < 1.0 {
                let n_samples = ((train_indices.len() as f32) * config.subsample).ceil() as usize;
                let mut indices = train_indices.clone();
                indices.shuffle(&mut rng);
                indices.truncate(n_samples);
                indices
            } else {
                train_indices.clone()
            };

            // Grow tree using subsampled training indices
            let tree = tree_grower.grow_with_indices(dataset, &gradients, &hessians, &sample_indices);

            // Update predictions for all rows (including validation for loss computation)
            for row_idx in 0..dataset.num_rows() {
                predictions[row_idx] += tree.predict_row(dataset, row_idx);
            }

            trees.push(tree);

            // Check for early stopping on validation set
            if early_stopping_enabled {
                // Compute validation loss (MSE for simplicity, works with any loss)
                let val_loss: f32 = validation_indices
                    .iter()
                    .map(|&idx| {
                        let residual = targets[idx] - predictions[idx];
                        residual * residual
                    })
                    .sum::<f32>()
                    / validation_indices.len() as f32;

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_num_trees = trees.len();
                    rounds_without_improvement = 0;
                } else {
                    rounds_without_improvement += 1;
                    if rounds_without_improvement >= config.early_stopping_rounds {
                        // Truncate to best number of trees
                        trees.truncate(best_num_trees);
                        break;
                    }
                }
            }
        }

        // If early stopping was used but we finished all rounds, still check if we should truncate
        if early_stopping_enabled && best_num_trees > 0 && best_num_trees < trees.len() {
            trees.truncate(best_num_trees);
        }

        // Auto-apply column reordering by feature importance if enabled
        let column_permutation = if config.column_reordering && !trees.is_empty() {
            let importances = Self::compute_importances_from_trees(&trees, dataset.num_features());
            Some(ColumnPermutation::from_importances(&importances))
        } else {
            None
        };

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
            column_permutation,
        })
    }

    /// Compute feature importances from a collection of trees (internal helper)
    fn compute_importances_from_trees(trees: &[Tree], num_features: usize) -> Vec<f32> {
        let mut importances = vec![0.0f32; num_features];

        for tree in trees {
            for (_, node) in tree.internal_nodes() {
                if let Some((feature_idx, _, _, _)) = node.split_info() {
                    importances[feature_idx] += node.sum_hessians;
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
    }

    /// Split data indices for training, validation, and calibration
    ///
    /// Returns (train_indices, validation_indices, calibration_indices)
    fn split_for_training(
        num_rows: usize,
        validation_ratio: f32,
        calibration_ratio: f32,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut indices: Vec<usize> = (0..num_rows).collect();
        indices.shuffle(&mut rng);

        // First split off calibration set
        let n_calibration = if calibration_ratio > 0.0 {
            ((num_rows as f32) * calibration_ratio).ceil() as usize
        } else {
            0
        };
        let calibration: Vec<usize> = indices.drain(..n_calibration).collect();

        // Then split off validation set from remaining
        let n_validation = if validation_ratio > 0.0 {
            ((indices.len() as f32) * validation_ratio / (1.0 - calibration_ratio)).ceil() as usize
        } else {
            0
        };
        let validation: Vec<usize> = indices.drain(..n_validation).collect();

        // Remaining is training set
        (indices, validation, calibration)
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
    ///
    /// Routes to parallel or sequential prediction based on config.parallel_prediction
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        if self.config.parallel_prediction {
            self.predict_parallel(dataset)
        } else {
            self.predict_sequential(dataset)
        }
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

    /// Get column permutation (if optimized layout was applied)
    pub fn column_permutation(&self) -> Option<&ColumnPermutation> {
        self.column_permutation.as_ref()
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

    /// Create a cache-optimized dataset by reordering columns based on feature importance
    ///
    /// More frequently used features are placed at the beginning of the dataset
    /// for better CPU cache locality during tree traversal.
    ///
    /// Returns the reordered dataset and the permutation mapping (new_idx -> original_idx)
    pub fn optimize_dataset_layout(
        &self,
        dataset: &BinnedDataset,
    ) -> (BinnedDataset, crate::dataset::ColumnPermutation) {
        let importances = self.feature_importances(dataset.num_features());
        let permutation = crate::dataset::ColumnPermutation::from_importances(&importances);
        let optimized = crate::dataset::reorder_dataset(dataset, &permutation);
        (optimized, permutation)
    }

    /// Create a memory-optimized packed dataset from a BinnedDataset
    ///
    /// Uses 4-bit packing for features with ≤16 unique bins,
    /// providing up to 50% memory savings for low-cardinality features.
    pub fn create_packed_dataset(
        &self,
        dataset: &BinnedDataset,
    ) -> crate::dataset::PackedDataset {
        crate::dataset::PackedDataset::from_binned(dataset)
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
    fn test_train_with_early_stopping() {
        let dataset = create_regression_dataset(1000, 0.1);

        let config = GBDTConfig::new()
            .with_num_rounds(100) // Max rounds
            .with_max_depth(4)
            .with_early_stopping(5, 0.2); // Stop after 5 rounds without improvement, 20% validation

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Should have stopped early (fewer than 100 trees)
        // With deterministic data, early stopping should trigger
        assert!(model.num_trees() < 100);
        assert!(model.num_trees() > 0);
    }

    #[test]
    fn test_train_with_subsampling() {
        let dataset = create_regression_dataset(1000, 0.1);

        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_subsample(0.8)  // 80% row subsampling
            .with_colsample(0.8); // 80% column subsampling

        let model = GBDTModel::train(&dataset, config).unwrap();

        assert_eq!(model.num_trees(), 10);

        // Predictions should still be reasonable
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 1000);
    }

    #[test]
    fn test_auto_column_reordering() {
        let dataset = create_regression_dataset(500, 0.1);

        // With column reordering enabled (default)
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4);

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Should have computed column permutation
        assert!(model.column_permutation().is_some());
        let permutation = model.column_permutation().unwrap();
        assert_eq!(permutation.new_to_original().len(), 3); // 3 features
    }

    #[test]
    fn test_column_reordering_disabled() {
        let dataset = create_regression_dataset(500, 0.1);

        // With column reordering disabled
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_column_reordering(false);

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Should not have computed column permutation
        assert!(model.column_permutation().is_none());
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

    #[test]
    fn test_train_with_monotonic_constraints() {
        use crate::tree::MonotonicConstraint;

        let dataset = create_regression_dataset(500, 0.1);

        // Set monotonic increasing constraint on feature 0
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_monotonic_constraints(vec![
                MonotonicConstraint::Increasing,
                MonotonicConstraint::None,
                MonotonicConstraint::None,
            ]);

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Model should train successfully with constraints
        assert!(model.num_trees() > 0);

        // Predictions should still work
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 500);
    }

    #[test]
    fn test_train_with_interaction_constraints() {
        let dataset = create_regression_dataset(500, 0.1);

        // Features 0, 1 can interact; feature 2 is unconstrained
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_interaction_groups(vec![vec![0, 1]]);

        let model = GBDTModel::train(&dataset, config).unwrap();

        // Model should train successfully with constraints
        assert!(model.num_trees() > 0);

        // Predictions should still work
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 500);
    }
}
