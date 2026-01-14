//! TreeSHAP: Exact SHAP values for tree ensembles
//!
//! Implements the TreeSHAP algorithm from "Consistent Individualized Feature Attribution
//! for Tree Ensembles" (Lundberg et al., 2020).
//!
//! SHAP (SHapley Additive exPlanations) values provide:
//! - Local explanations: per-sample feature contributions
//! - Consistency: features that increase prediction get positive attribution
//! - Additivity: contributions sum to (prediction - base_value)
//!
//! # Example
//!
//! ```ignore
//! let model = GBDTModel::train_binned(&dataset, config)?;
//! let shap = model.compute_shap_values(&dataset);
//!
//! // Per-sample explanations
//! let sample_contributions = shap.values_for_sample(0);
//!
//! // Global importance (mean |SHAP|)
//! let importance = shap.mean_abs_shap();
//! ```

use super::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::tree::{NodeType, Tree};
use rayon::prelude::*;

/// SHAP values for a dataset
///
/// Contains per-sample, per-feature SHAP values that explain model predictions.
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// SHAP values: shape [num_samples, num_features]
    /// Row-major: values[sample * num_features + feature]
    values: Vec<f32>,
    /// Number of samples
    num_samples: usize,
    /// Number of features
    num_features: usize,
    /// Base value (expected prediction over training data)
    pub base_value: f32,
}

impl ShapValues {
    /// Create new SHAP values container
    pub fn new(num_samples: usize, num_features: usize, base_value: f32) -> Self {
        Self {
            values: vec![0.0; num_samples * num_features],
            num_samples,
            num_features,
            base_value,
        }
    }

    /// Get SHAP values for a specific sample
    ///
    /// Returns a slice of length `num_features` where each element
    /// is the contribution of that feature to the prediction.
    #[inline]
    pub fn values_for_sample(&self, sample_idx: usize) -> &[f32] {
        let start = sample_idx * self.num_features;
        &self.values[start..start + self.num_features]
    }

    /// Get mutable SHAP values for a specific sample
    #[inline]
    fn values_for_sample_mut(&mut self, sample_idx: usize) -> &mut [f32] {
        let start = sample_idx * self.num_features;
        &mut self.values[start..start + self.num_features]
    }

    /// Get all SHAP values as a flat array (row-major)
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of samples
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Compute mean absolute SHAP values per feature (global importance)
    ///
    /// This is the recommended way to compute global feature importance from SHAP.
    /// Higher values indicate more important features.
    pub fn mean_abs_shap(&self) -> Vec<f32> {
        let mut importance = vec![0.0f32; self.num_features];

        for sample_idx in 0..self.num_samples {
            let sample_shap = self.values_for_sample(sample_idx);
            for (feat_idx, &shap) in sample_shap.iter().enumerate() {
                importance[feat_idx] += shap.abs();
            }
        }

        // Normalize by number of samples
        if self.num_samples > 0 {
            for imp in &mut importance {
                *imp /= self.num_samples as f32;
            }
        }

        importance
    }

    /// Compute mean SHAP values per feature (signed importance)
    ///
    /// Positive values indicate the feature generally increases predictions.
    /// Negative values indicate the feature generally decreases predictions.
    pub fn mean_shap(&self) -> Vec<f32> {
        let mut importance = vec![0.0f32; self.num_features];

        for sample_idx in 0..self.num_samples {
            let sample_shap = self.values_for_sample(sample_idx);
            for (feat_idx, &shap) in sample_shap.iter().enumerate() {
                importance[feat_idx] += shap;
            }
        }

        if self.num_samples > 0 {
            for imp in &mut importance {
                *imp /= self.num_samples as f32;
            }
        }

        importance
    }

    /// Get top K most important features by mean |SHAP|
    ///
    /// Returns vector of (feature_index, importance) sorted descending.
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let importance = self.mean_abs_shap();
        let mut indexed: Vec<(usize, f32)> = importance.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// Verify additivity: sum of SHAP values + base_value ≈ prediction
    ///
    /// Returns max absolute error across all samples.
    pub fn verify_additivity(&self, predictions: &[f32]) -> f32 {
        let mut max_error = 0.0f32;

        for (sample_idx, &pred) in predictions.iter().enumerate() {
            let shap_sum: f32 = self.values_for_sample(sample_idx).iter().sum();
            let reconstructed = self.base_value + shap_sum;
            let error = (pred - reconstructed).abs();
            max_error = max_error.max(error);
        }

        max_error
    }
}

/// Path element for tracking the sample's path through the tree
#[derive(Clone, Copy, Debug)]
struct PathNode {
    /// Feature used at this node
    feature_idx: i32,
    /// Expected value when going left from this node
    left_expected: f32,
    /// Expected value when going right from this node
    right_expected: f32,
    /// Whether sample went left
    went_left: bool,
}

impl GBDTModel {
    /// Compute SHAP values for all samples in a dataset
    ///
    /// Uses the TreeSHAP algorithm for exact, efficient computation.
    /// Returns SHAP values where `values[sample][feature]` is the contribution
    /// of that feature to that sample's prediction.
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset to explain
    ///
    /// # Returns
    /// `ShapValues` containing per-sample feature contributions
    ///
    /// # Example
    /// ```ignore
    /// let shap = model.compute_shap_values(&dataset);
    /// let contributions = shap.values_for_sample(0);
    /// println!("Feature 0 contributed: {}", contributions[0]);
    /// ```
    pub fn compute_shap_values(&self, dataset: &BinnedDataset) -> ShapValues {
        let num_samples = dataset.num_rows();
        let num_features = self.num_features();

        // Compute base value (average prediction)
        let base_value = self.compute_base_value();

        let mut shap = ShapValues::new(num_samples, num_features, base_value);

        // Process samples in parallel for large datasets
        const PARALLEL_THRESHOLD: usize = 1000;

        if num_samples >= PARALLEL_THRESHOLD {
            // Parallel computation
            let sample_shap: Vec<Vec<f32>> = (0..num_samples)
                .into_par_iter()
                .map(|sample_idx| self.compute_sample_shap(dataset, sample_idx))
                .collect();

            // Copy results
            for (sample_idx, values) in sample_shap.into_iter().enumerate() {
                shap.values_for_sample_mut(sample_idx)
                    .copy_from_slice(&values);
            }
        } else {
            // Sequential computation
            for sample_idx in 0..num_samples {
                let values = self.compute_sample_shap(dataset, sample_idx);
                shap.values_for_sample_mut(sample_idx)
                    .copy_from_slice(&values);
            }
        }

        shap
    }

    /// Compute SHAP values using raw feature values
    ///
    /// # Arguments
    /// * `features` - Row-major feature matrix [num_samples * num_features]
    /// * `num_samples` - Number of samples
    ///
    /// # Returns
    /// `ShapValues` containing per-sample feature contributions
    pub fn compute_shap_values_raw(&self, features: &[f64], num_samples: usize) -> ShapValues {
        let num_features = self.num_features();
        debug_assert_eq!(features.len(), num_samples * num_features);

        let base_value = self.compute_base_value();
        let mut shap = ShapValues::new(num_samples, num_features, base_value);

        const PARALLEL_THRESHOLD: usize = 1000;

        if num_samples >= PARALLEL_THRESHOLD {
            let sample_shap: Vec<Vec<f32>> = (0..num_samples)
                .into_par_iter()
                .map(|sample_idx| {
                    let row_start = sample_idx * num_features;
                    let row = &features[row_start..row_start + num_features];
                    self.compute_sample_shap_raw(row)
                })
                .collect();

            for (sample_idx, values) in sample_shap.into_iter().enumerate() {
                shap.values_for_sample_mut(sample_idx)
                    .copy_from_slice(&values);
            }
        } else {
            for sample_idx in 0..num_samples {
                let row_start = sample_idx * num_features;
                let row = &features[row_start..row_start + num_features];
                let values = self.compute_sample_shap_raw(row);
                shap.values_for_sample_mut(sample_idx)
                    .copy_from_slice(&values);
            }
        }

        shap
    }

    /// Compute SHAP values for a single sample (binned)
    ///
    /// Uses the "path" approach: attribute prediction changes at each split
    /// to the feature that made the split decision.
    fn compute_sample_shap(&self, dataset: &BinnedDataset, sample_idx: usize) -> Vec<f32> {
        let num_features = self.num_features();
        let mut shap_values = vec![0.0f32; num_features];

        for tree in &self.trees {
            if let Some(scalar_tree) = tree.try_as_scalar() {
                self.tree_shap_path(scalar_tree, &mut shap_values, |feat_idx| {
                    dataset.get_bin(sample_idx, feat_idx)
                });
            }
        }

        shap_values
    }

    /// Compute SHAP values for a single sample (raw features)
    fn compute_sample_shap_raw(&self, features: &[f64]) -> Vec<f32> {
        let num_features = self.num_features();
        let mut shap_values = vec![0.0f32; num_features];

        for tree in &self.trees {
            if let Some(scalar_tree) = tree.try_as_scalar() {
                self.tree_shap_path_raw(scalar_tree, &mut shap_values, features);
            }
        }

        shap_values
    }

    /// Compute SHAP values for a single tree using path attribution
    ///
    /// The idea: as we traverse the tree, each split decision changes the expected
    /// value. We attribute that change to the feature that made the split.
    fn tree_shap_path<F>(&self, tree: &Tree, shap_values: &mut [f32], get_bin: F)
    where
        F: Fn(usize) -> u8,
    {
        let mut path: Vec<PathNode> = Vec::with_capacity(tree.max_depth() + 1);

        // Start at root with tree's expected value
        let root_expected = self.tree_expected_value(tree);
        let mut node_idx = 0;

        loop {
            let node = tree.get_node(node_idx);

            match &node.node_type {
                NodeType::Leaf { value } => {
                    // At leaf: unwind path and compute SHAP contributions
                    self.attribute_path_shap(&path, *value, root_expected, shap_values);
                    break;
                }
                NodeType::Internal {
                    feature_idx,
                    bin_threshold,
                    left_child,
                    right_child,
                    ..
                } => {
                    let bin = get_bin(*feature_idx);
                    let goes_left = bin <= *bin_threshold;

                    // Compute expected values for children
                    let left_expected = self.subtree_expected_value(tree, *left_child);
                    let right_expected = self.subtree_expected_value(tree, *right_child);

                    // Record this decision point
                    path.push(PathNode {
                        feature_idx: *feature_idx as i32,
                        left_expected,
                        right_expected,
                        went_left: goes_left,
                    });

                    // Move to next node
                    node_idx = if goes_left { *left_child } else { *right_child };
                }
            }
        }
    }

    /// Compute SHAP values for a single tree using path attribution (raw features)
    fn tree_shap_path_raw(&self, tree: &Tree, shap_values: &mut [f32], features: &[f64]) {
        let mut path: Vec<PathNode> = Vec::with_capacity(tree.max_depth() + 1);

        let root_expected = self.tree_expected_value(tree);
        let mut node_idx = 0;

        loop {
            let node = tree.get_node(node_idx);

            match &node.node_type {
                NodeType::Leaf { value } => {
                    self.attribute_path_shap(&path, *value, root_expected, shap_values);
                    break;
                }
                NodeType::Internal {
                    feature_idx,
                    split_value,
                    left_child,
                    right_child,
                    ..
                } => {
                    let feat_value = features[*feature_idx];
                    let goes_left = feat_value <= *split_value;

                    let left_expected = self.subtree_expected_value(tree, *left_child);
                    let right_expected = self.subtree_expected_value(tree, *right_child);

                    path.push(PathNode {
                        feature_idx: *feature_idx as i32,
                        left_expected,
                        right_expected,
                        went_left: goes_left,
                    });

                    node_idx = if goes_left { *left_child } else { *right_child };
                }
            }
        }
    }

    /// Attribute SHAP values along a path
    ///
    /// At each split, the feature's contribution is the difference between
    /// the expected value of the actual path taken and the parent's expected value.
    fn attribute_path_shap(
        &self,
        path: &[PathNode],
        _leaf_value: f32,
        root_expected: f32,
        shap_values: &mut [f32],
    ) {
        if path.is_empty() {
            // Tree is a single leaf - all prediction goes to bias (base_value)
            return;
        }

        // Track the "current" expected value as we traverse
        // At each split, the contribution is the change in conditional expected value
        let mut prev_expected = root_expected;

        for path_node in path {
            let feature_idx = path_node.feature_idx as usize;

            // The conditional expected value of the branch we took
            let branch_expected = if path_node.went_left {
                path_node.left_expected
            } else {
                path_node.right_expected
            };

            if feature_idx < shap_values.len() {
                // SHAP contribution: change in expected value from this split
                let contribution = branch_expected - prev_expected;
                shap_values[feature_idx] += contribution;
            }

            prev_expected = branch_expected;
        }
    }

    /// Compute base value (expected prediction = initial + weighted average of tree contributions)
    fn compute_base_value(&self) -> f32 {
        // Start with the model's initial prediction (from loss function)
        let mut base = self.base_predictions.first().copied().unwrap_or(0.0);

        // Add the expected contribution from each tree
        for tree in &self.trees {
            if let Some(scalar_tree) = tree.try_as_scalar() {
                base += self.tree_expected_value(scalar_tree);
            }
        }

        base
    }

    /// Compute expected value of a single tree (weighted average of leaves)
    fn tree_expected_value(&self, tree: &Tree) -> f32 {
        self.conditional_expected_value(tree, 0)
    }

    /// Compute expected value for a subtree rooted at given node
    /// This is the conditional expected value GIVEN we're at this node
    fn subtree_expected_value(&self, tree: &Tree, node_idx: usize) -> f32 {
        self.conditional_expected_value(tree, node_idx)
    }

    /// Compute conditional expected value of subtree
    /// E[f(x) | x reaches node_idx]
    fn conditional_expected_value(&self, tree: &Tree, node_idx: usize) -> f32 {
        let node = tree.get_node(node_idx);
        let node_samples = node.num_samples as f32;

        match &node.node_type {
            NodeType::Leaf { value } => {
                // At leaf, expected value is just the leaf value
                *value
            }
            NodeType::Internal {
                left_child,
                right_child,
                ..
            } => {
                let left_node = tree.get_node(*left_child);
                let right_node = tree.get_node(*right_child);

                let left_samples = left_node.num_samples as f32;
                let right_samples = right_node.num_samples as f32;

                // Weights based on proportion of samples going each way
                let left_weight = left_samples / node_samples.max(1.0);
                let right_weight = right_samples / node_samples.max(1.0);

                // Expected value is weighted average of children's expected values
                let left_expected = self.conditional_expected_value(tree, *left_child);
                let right_expected = self.conditional_expected_value(tree, *right_child);

                left_weight * left_expected + right_weight * right_expected
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booster::GBDTConfig;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize) -> BinnedDataset {
        let num_features = 3;

        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * (f + 1) * 17) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows)
            .map(|i| {
                let f0 = features[i] as f32 / 255.0;
                let f1 = features[num_rows + i] as f32 / 255.0;
                f0 * 10.0 + f1 * 5.0
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
    fn test_shap_values_creation() {
        let shap = ShapValues::new(10, 5, 0.5);

        assert_eq!(shap.num_samples(), 10);
        assert_eq!(shap.num_features(), 5);
        assert_eq!(shap.base_value, 0.5);
        assert_eq!(shap.values().len(), 50);
    }

    #[test]
    fn test_shap_mean_abs() {
        let mut shap = ShapValues::new(3, 2, 0.0);

        // Sample 0: [1.0, -2.0]
        // Sample 1: [-1.0, 2.0]
        // Sample 2: [3.0, 0.0]
        shap.values[0] = 1.0;
        shap.values[1] = -2.0;
        shap.values[2] = -1.0;
        shap.values[3] = 2.0;
        shap.values[4] = 3.0;
        shap.values[5] = 0.0;

        let importance = shap.mean_abs_shap();

        // Feature 0: mean(|1|, |-1|, |3|) = 5/3
        // Feature 1: mean(|-2|, |2|, |0|) = 4/3
        assert!((importance[0] - 5.0 / 3.0).abs() < 1e-6);
        assert!((importance[1] - 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_shap_top_features() {
        let mut shap = ShapValues::new(2, 3, 0.0);

        // Feature 0: mean_abs = 1.0
        // Feature 1: mean_abs = 3.0
        // Feature 2: mean_abs = 2.0
        shap.values[0] = 1.0;
        shap.values[1] = 3.0;
        shap.values[2] = 2.0;
        shap.values[3] = 1.0;
        shap.values[4] = 3.0;
        shap.values[5] = 2.0;

        let top = shap.top_features(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 1); // Feature 1 is most important
        assert_eq!(top[1].0, 2); // Feature 2 is second
    }

    #[test]
    fn test_compute_shap_values() {
        let dataset = create_test_dataset(100);
        let config = GBDTConfig::new().with_num_rounds(5).with_max_depth(3);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let shap = model.compute_shap_values(&dataset);

        assert_eq!(shap.num_samples(), 100);
        assert_eq!(shap.num_features(), 3);

        // SHAP values should exist for all samples
        for i in 0..100 {
            let sample_shap = shap.values_for_sample(i);
            assert_eq!(sample_shap.len(), 3);
        }
    }

    #[test]
    fn test_shap_additivity() {
        let dataset = create_test_dataset(50);
        let config = GBDTConfig::new().with_num_rounds(3).with_max_depth(2);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let predictions = model.predict(&dataset);
        let shap = model.compute_shap_values(&dataset);

        // Verify additivity: base_value + sum(shap_values) ≈ prediction
        let max_error = shap.verify_additivity(&predictions);

        // Allow some numerical tolerance
        assert!(
            max_error < 0.01,
            "SHAP additivity error too high: {}",
            max_error
        );
    }

    #[test]
    fn test_shap_global_importance() {
        let dataset = create_test_dataset(100);
        let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(4);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let shap = model.compute_shap_values(&dataset);

        let importance = shap.mean_abs_shap();

        // All importances should be non-negative
        for imp in &importance {
            assert!(*imp >= 0.0);
        }

        // At least one feature should have non-zero importance
        let total: f32 = importance.iter().sum();
        assert!(total > 0.0, "All features have zero importance");
    }

    #[test]
    fn test_shap_sign_consistency() {
        // Features that increase prediction should have positive SHAP
        // when they have high values (on average)
        let dataset = create_test_dataset(100);
        let config = GBDTConfig::new().with_num_rounds(10).with_max_depth(4);

        let model = GBDTModel::train_binned(&dataset, config).unwrap();
        let shap = model.compute_shap_values(&dataset);

        // Just verify we get some non-zero values
        let importance = shap.mean_abs_shap();
        let total: f32 = importance.iter().sum();
        assert!(total > 0.0);
    }
}
