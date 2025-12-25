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
            // Compute gradients and hessians
            // Parallel mode: enable for large datasets (100k+ rows)
            if config.parallel_gradient {
                train_indices.par_iter().for_each(|&idx| {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                    // SAFETY: Each idx is unique within train_indices, so no data races
                    unsafe {
                        let grad_ptr = gradients.as_ptr() as *mut f32;
                        let hess_ptr = hessians.as_ptr() as *mut f32;
                        *grad_ptr.add(idx) = g;
                        *hess_ptr.add(idx) = h;
                    }
                });
            } else {
                for &idx in &train_indices {
                    let (g, h) = loss_fn.gradient_hessian(targets[idx], predictions[idx]);
                    gradients[idx] = g;
                    hessians[idx] = h;
                }
            }

            // GOSS or random subsampling
            let sample_indices: Vec<usize> = if config.goss_enabled {
                // GOSS: Gradient-based One-Side Sampling
                // Returns sampled indices and applies weight correction in-place
                Self::goss_sample(
                    &train_indices,
                    &mut gradients,
                    &mut hessians,
                    config.goss_top_rate,
                    config.goss_other_rate,
                    &mut rng,
                )
            } else if config.subsample < 1.0 {
                // Random subsampling (Stochastic Gradient Boosting)
                let n_samples =
                    ((train_indices.len() as f32) * config.subsample).ceil() as usize;
                let mut indices = train_indices.clone();
                indices.shuffle(&mut rng);
                indices.truncate(n_samples);
                indices
            } else {
                train_indices.clone()
            };

            // Grow tree using subsampled training indices
            let tree = tree_grower.grow_with_indices(dataset, &gradients, &hessians, &sample_indices);

            // Update predictions using tree-wise batch prediction
            // This is more cache-friendly than row-wise and avoids intermediate allocation
            tree.predict_batch_add(dataset, &mut predictions);

            trees.push(tree);

            // Check for early stopping on validation set
            if early_stopping_enabled {
                // Compute validation loss (MSE for simplicity, works with any loss)
                // Use parallel for large validation sets, sequential for small ones
                let val_loss: f32 = if validation_indices.len() >= 10000 {
                    validation_indices
                        .par_iter()
                        .map(|&idx| {
                            let residual = targets[idx] - predictions[idx];
                            residual * residual
                        })
                        .sum::<f32>()
                } else {
                    validation_indices
                        .iter()
                        .map(|&idx| {
                            let residual = targets[idx] - predictions[idx];
                            residual * residual
                        })
                        .sum::<f32>()
                } / validation_indices.len() as f32;

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
            let calib_residuals: Vec<f32> = if calibration_indices.len() >= 10000 {
                calibration_indices
                    .par_iter()
                    .map(|&idx| (targets[idx] - predictions[idx]).abs())
                    .collect()
            } else {
                calibration_indices
                    .iter()
                    .map(|&idx| (targets[idx] - predictions[idx]).abs())
                    .collect()
            };

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
                if let Some((feature_idx, _, _, _, _)) = node.split_info() {
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

    /// GOSS (Gradient-based One-Side Sampling)
    ///
    /// Selects samples based on gradient magnitude:
    /// 1. Keep all top `top_rate` samples with largest |gradient|
    /// 2. Randomly sample `other_rate` from the remaining samples
    /// 3. Apply weight correction (1 - top_rate) / other_rate to sampled small-gradient samples
    ///
    /// Weight correction is applied in-place to gradients and hessians.
    /// Uses partial sorting (select_nth_unstable) for O(n) instead of O(n log n).
    fn goss_sample(
        train_indices: &[usize],
        gradients: &mut [f32],
        hessians: &mut [f32],
        top_rate: f32,
        other_rate: f32,
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<usize> {
        let n = train_indices.len();
        if n == 0 {
            return Vec::new();
        }

        // Number of top-gradient samples to keep
        let n_top = ((n as f32) * top_rate).ceil() as usize;
        let n_top = n_top.min(n);
        // Number of other samples to randomly select
        let n_other = ((n as f32) * other_rate).ceil() as usize;

        // Use partial sort to find the n_top largest gradients in O(n) time
        let mut indexed: Vec<(usize, f32)> = train_indices
            .iter()
            .map(|&idx| (idx, gradients[idx].abs()))
            .collect();

        // Partition around the n_top-th largest element (descending order)
        if n_top < n {
            indexed.select_nth_unstable_by(n_top, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Top n_top samples (large gradients) - no weight modification needed
        let top_indices: Vec<usize> = indexed[..n_top].iter().map(|(idx, _)| *idx).collect();

        // Randomly sample n_other from the rest (small gradients)
        let mut rest: Vec<usize> = indexed[n_top..].iter().map(|(idx, _)| *idx).collect();
        rest.shuffle(rng);
        rest.truncate(n_other);

        // Weight correction factor for small-gradient samples
        let weight = (1.0 - top_rate) / other_rate;

        // Apply weight correction to the sampled small-gradient samples
        for &idx in &rest {
            gradients[idx] *= weight;
            hessians[idx] *= weight;
        }

        // Combine top samples (weight = 1.0) and weighted small samples
        let mut result = top_indices;
        result.extend(rest);
        result
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

    /// Predict for all rows using tree-wise batch prediction
    ///
    /// This approach traverses one tree for ALL rows before moving to the next tree,
    /// which is more cache-friendly than row-wise traversal.
    ///
    /// Routes to parallel or sequential based on config.parallel_prediction
    pub fn predict(&self, dataset: &BinnedDataset) -> Vec<f32> {
        if self.config.parallel_prediction {
            self.predict_parallel(dataset)
        } else {
            self.predict_sequential(dataset)
        }
    }

    /// Single-threaded tree-wise batch prediction
    ///
    /// Traverses each tree for all rows before moving to the next tree.
    /// More cache-friendly than row-wise traversal.
    pub fn predict_sequential(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // Initialize predictions with base value
        let mut predictions = vec![self.base_prediction; num_rows];

        // Tree-wise: traverse each tree for all rows
        for tree in &self.trees {
            tree.predict_batch_add(dataset, &mut predictions);
        }

        predictions
    }

    /// Parallel tree-wise batch prediction
    ///
    /// Splits rows into chunks and processes each chunk in parallel.
    /// Each chunk uses tree-wise traversal internally.
    pub fn predict_parallel(&self, dataset: &BinnedDataset) -> Vec<f32> {
        let num_rows = dataset.num_rows();

        // For small datasets, use sequential
        if num_rows < 1000 || self.trees.is_empty() {
            return self.predict_sequential(dataset);
        }

        // Initialize predictions with base value
        let mut predictions = vec![self.base_prediction; num_rows];

        // Determine chunk size for parallelism (target ~4 chunks per thread)
        let num_threads = rayon::current_num_threads();
        let chunk_size = (num_rows / (num_threads * 4)).max(256);

        // Process chunks in parallel, each chunk does tree-wise traversal
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_row = chunk_idx * chunk_size;

                // For each tree, process this chunk of rows
                for tree in &self.trees {
                    for (i, pred) in chunk.iter_mut().enumerate() {
                        let row_idx = start_row + i;
                        *pred += tree.predict(|f| dataset.get_bin(row_idx, f));
                    }
                }
            });

        predictions
    }

    /// Legacy row-wise prediction (kept for comparison/testing)
    #[doc(hidden)]
    pub fn predict_row_wise(&self, dataset: &BinnedDataset) -> Vec<f32> {
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

    // ============================================================================
    // Raw prediction methods (no binning required)
    // ============================================================================

    /// Predict using raw feature values (no binning needed)
    ///
    /// This is the primary prediction method for external use (e.g., Python bindings).
    /// Uses the split_value stored in tree nodes to compare directly against raw values,
    /// avoiding the overhead of binning on every prediction call.
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix: features[row * num_features + feature]
    ///                Shape: (num_rows, num_features)
    ///
    /// # Returns
    /// Vector of predictions for each row
    pub fn predict_raw(&self, features: &[f64]) -> Vec<f32> {
        let num_features = self.num_features();
        if num_features == 0 {
            return vec![];
        }

        let num_rows = features.len() / num_features;
        debug_assert_eq!(features.len(), num_rows * num_features);

        if self.config.parallel_prediction && num_rows >= 1000 {
            self.predict_raw_parallel(features, num_features)
        } else {
            self.predict_raw_sequential(features, num_features)
        }
    }

    /// Single-threaded raw prediction using tree-wise traversal
    fn predict_raw_sequential(&self, features: &[f64], num_features: usize) -> Vec<f32> {
        let num_rows = features.len() / num_features;

        // Initialize predictions with base value
        let mut predictions = vec![self.base_prediction; num_rows];

        // Tree-wise: traverse each tree for all rows
        for tree in &self.trees {
            tree.predict_batch_add_raw(features, num_features, &mut predictions);
        }

        predictions
    }

    /// Parallel raw prediction using tree-wise traversal
    fn predict_raw_parallel(&self, features: &[f64], num_features: usize) -> Vec<f32> {
        let num_rows = features.len() / num_features;

        // For small datasets, use sequential
        if num_rows < 1000 || self.trees.is_empty() {
            return self.predict_raw_sequential(features, num_features);
        }

        // Initialize predictions with base value
        let mut predictions = vec![self.base_prediction; num_rows];

        // Determine chunk size for parallelism
        let num_threads = rayon::current_num_threads();
        let chunk_size = (num_rows / (num_threads * 4)).max(256);

        // Process chunks in parallel
        predictions
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_row = chunk_idx * chunk_size;
                let chunk_features_start = start_row * num_features;

                // Each thread processes its chunk through all trees
                for tree in &self.trees {
                    for (i, pred) in chunk.iter_mut().enumerate() {
                        let row_offset = chunk_features_start + i * num_features;
                        *pred += tree.predict_raw(|f| features[row_offset + f]);
                    }
                }
            });

        predictions
    }

    /// Predict raw with conformal intervals
    ///
    /// Returns (predictions, lower_bounds, upper_bounds)
    pub fn predict_raw_with_intervals(&self, features: &[f64]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let predictions = self.predict_raw(features);

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
                let (feature_idx, _, _, _, _) = node.split_info().unwrap();
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
    fn test_train_with_goss() {
        let dataset = create_regression_dataset(1000, 0.1);

        // GOSS enabled with default rates (top 20%, sample 10% of rest)
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_goss(true);

        let model = GBDTModel::train(&dataset, config).unwrap();

        assert_eq!(model.num_trees(), 10);

        // Predictions should still be reasonable
        let predictions = model.predict(&dataset);
        assert_eq!(predictions.len(), 1000);
    }

    #[test]
    fn test_train_with_goss_custom_rates() {
        let dataset = create_regression_dataset(1000, 0.1);

        // Custom GOSS rates
        let config = GBDTConfig::new()
            .with_num_rounds(10)
            .with_max_depth(4)
            .with_goss_rates(0.3, 0.15); // top 30%, sample 15% of rest

        let model = GBDTModel::train(&dataset, config).unwrap();

        assert_eq!(model.num_trees(), 10);

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
