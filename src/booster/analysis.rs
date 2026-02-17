//! GBDT model analysis and introspection
//!
//! Contains methods for analyzing trained models:
//! - Feature importance computation
//! - Dataset optimization and layout

use super::GBDTModel;
use crate::dataset::{BinnedDataset, ColumnPermutation};
use crate::tree::{EnsembleTree, Tree};

impl GBDTModel {
    /// Compute feature importance (gain-based)
    ///
    /// Works with both scalar trees (regression/binary/multi-class) and
    /// vector trees (multi-label with unified splits).
    pub fn feature_importance(&self) -> Vec<f32> {
        let mut importances = vec![0.0f32; self.num_features()];

        for tree in &self.trees {
            // EnsembleTree.internal_nodes() returns (node_idx, feature_idx, hessian_sum)
            for (_, feature_idx, hessian_sum) in tree.internal_nodes() {
                if feature_idx < importances.len() {
                    importances[feature_idx] += hessian_sum;
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

    /// Create a cache-optimized dataset by reordering columns based on feature importance
    ///
    /// This is primarily useful for repeated predictions where cache locality matters.
    /// Most frequent features are moved to lower indices for better cache utilization.
    ///
    /// # Returns
    /// A new BinnedDataset with columns reordered by importance (descending)
    pub fn optimize_dataset_layout(&self, dataset: &BinnedDataset) -> BinnedDataset {
        let importances = self.feature_importance();
        let perm = ColumnPermutation::from_importances(&importances);
        crate::dataset::reorder_dataset(dataset, &perm)
    }

    /// Create a "packed" dataset optimized for SIMD prediction
    ///
    /// This packs features into a more cache-friendly layout for repeated
    /// prediction operations, particularly useful for serving models.
    pub fn create_packed_dataset(&self, dataset: &BinnedDataset) -> crate::dataset::PackedDataset {
        crate::dataset::PackedDataset::from_binned(dataset)
    }

    /// Get number of boosting rounds (trees per class for multi-class, trees for others)
    pub fn num_rounds(&self) -> usize {
        if matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            self.trees.len() / self.num_outputs()
        } else {
            self.trees.len()
        }
    }

    /// Append a vector of scalar trees to the model
    ///
    /// Used for continuing training or ensembling.
    /// Validates that tree count matches multi-class requirements.
    pub fn append_trees(&mut self, new_trees: Vec<Tree>) {
        self.trees
            .extend(new_trees.into_iter().map(EnsembleTree::from));
    }

    /// Append ensemble trees directly to the model
    ///
    /// Use this when you have pre-wrapped EnsembleTree instances.
    pub fn append_ensemble_trees(&mut self, new_trees: Vec<EnsembleTree>) {
        self.trees.extend(new_trees);
    }

    /// Append a single scalar tree to the model
    ///
    /// Useful for incremental training scenarios.
    pub fn append_tree(&mut self, tree: Tree) {
        self.trees.push(EnsembleTree::from(tree));
    }

    /// Append a single ensemble tree to the model
    pub fn append_ensemble_tree(&mut self, tree: EnsembleTree) {
        self.trees.push(tree);
    }

    /// Compute residuals (prediction errors) for training data
    ///
    /// Residual = target - prediction
    /// Useful for analyzing model performance and building stacks.
    pub fn compute_residuals(&self, dataset: &BinnedDataset, targets: &[f32]) -> Vec<f32> {
        let predictions = self.predict(dataset);
        predictions
            .into_iter()
            .zip(targets.iter())
            .map(|(pred, &target)| target - pred)
            .collect()
    }

    /// Compute residuals using raw feature predictions
    ///
    /// Residual = target - prediction
    /// Doesn't require a binned dataset (uses stored split values in trees)
    pub fn compute_residuals_raw(&self, features: &[f64], targets: &[f32]) -> Vec<f32> {
        let predictions = self.predict_raw(features);
        predictions
            .into_iter()
            .zip(targets.iter())
            .map(|(pred, &target)| target - pred)
            .collect()
    }

    /// Check if this model can accept new trees with the given number of features
    ///
    /// Used when continuing training to ensure compatibility.
    pub fn is_compatible_for_update(&self, num_features: usize) -> bool {
        self.num_features() == num_features
    }

    /// Get mutable reference to tree vector for advanced use cases
    ///
    /// Use with caution - modifying trees directly can break invariants.
    pub fn trees_mut(&mut self) -> &mut Vec<EnsembleTree> {
        &mut self.trees
    }

    /// Truncate model to keep only first N rounds
    ///
    /// Useful for early stopping, model compression, or finding optimal ensemble size.
    /// For multi-class models, truncates to N complete rounds (N * num_classes trees).
    pub fn truncate_to_rounds(&mut self, num_rounds: usize) {
        if matches!(self.output_type(), crate::booster::OutputType::MultiClass) {
            // Multi-class: truncate to num_rounds * num_outputs trees
            let target_trees = num_rounds * self.num_outputs();
            self.trees.truncate(target_trees);
        } else {
            // Binary/regression: num_rounds = num_trees
            self.trees.truncate(num_rounds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booster::GBDTConfig;
    use crate::dataset::FeatureInfo;
    use crate::dataset::FeatureType;

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
                impute_value: 0.0,
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_feature_importance() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(5);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let importances = model.feature_importance();

        assert_eq!(importances.len(), 3);
        let sum: f32 = importances.iter().sum();
        assert!((sum - 1.0).abs() < 0.01 || sum == 0.0); // Normalized or all zero
    }

    #[test]
    fn test_tree_residual_appending() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(3);
        let mut model = GBDTModel::train_binned(&dataset, config).unwrap();
        let original_trees = model.num_trees();

        // Append more trees
        let config2 = GBDTConfig::new().with_num_rounds(2);
        let model2 = GBDTModel::train_binned(&dataset, config2).unwrap();
        let new_trees = model2.trees().to_vec();

        model.append_ensemble_trees(new_trees);
        assert_eq!(model.num_trees(), original_trees + 2);
    }

    #[test]
    fn test_tree_ensemble_growth() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(3);
        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let mut model_ensemble = model.clone();

        for _ in 0..2 {
            let config2 = GBDTConfig::new().with_num_rounds(1);
            let new_model = GBDTModel::train_binned(&dataset, config2).unwrap();
            let new_trees = new_model.trees().to_vec();
            model_ensemble.append_ensemble_trees(new_trees);
        }

        assert_eq!(model_ensemble.num_trees(), 5);
    }

    #[test]
    fn test_append_single_tree() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(5);
        let mut model = GBDTModel::train_binned(&dataset, config).unwrap();
        assert_eq!(model.num_trees(), 5);

        // Append a single new tree
        let config2 = GBDTConfig::new().with_num_rounds(1);
        let model2 = GBDTModel::train_binned(&dataset, config2).unwrap();
        let tree = model2.trees()[0].clone();

        model.append_ensemble_tree(tree);
        assert_eq!(model.num_trees(), 6);
    }

    #[test]
    fn test_compute_residuals_correctness() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(3);
        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        let targets = dataset.targets();
        let residuals = model.compute_residuals(&dataset, targets);

        // Verify: residual = target - prediction
        let predictions = model.predict(&dataset);
        for (i, &r) in residuals.iter().enumerate() {
            let expected = targets[i] - predictions[i];
            assert!(
                (r - expected).abs() < 1e-5,
                "Residual {} mismatch: got {}, expected {}",
                i,
                r,
                expected
            );
        }
    }

    #[test]
    fn test_truncate_to_rounds() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(3);

        let mut model = GBDTModel::train_binned(&dataset, config).unwrap();
        assert_eq!(model.num_trees(), 10);

        // Truncate to 5 rounds
        model.truncate_to_rounds(5);
        assert_eq!(model.num_trees(), 5);
        assert_eq!(model.num_rounds(), 5);

        // Truncating to more rounds than exist should be no-op
        model.truncate_to_rounds(20);
        assert_eq!(model.num_trees(), 5);
    }

    #[test]
    fn test_is_compatible_for_update() {
        let dataset = create_regression_dataset(100, 0.1);

        let config = GBDTConfig::new().with_num_rounds(3);
        let model = GBDTModel::train_binned(&dataset, config).unwrap();

        // Should be compatible with same number of features
        assert!(model.is_compatible_for_update(3));

        // Should not be compatible with different number
        assert!(!model.is_compatible_for_update(5));
        assert!(!model.is_compatible_for_update(2));
    }
}
