//! Tree growing with Best-First (Leaf-wise) strategy

use crate::dataset::BinnedDataset;
use crate::histogram::{HistogramBuilder, NodeHistograms};
use crate::tree::{Node, SplitFinder, SplitInfo, Tree};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Candidate node for splitting
#[derive(Debug)]
struct SplitCandidate {
    /// Node index in tree
    node_idx: usize,
    /// Row indices belonging to this node
    row_indices: Vec<usize>,
    /// Precomputed histograms (if available)
    histograms: Option<NodeHistograms>,
    /// Best split info (if computed)
    split_info: Option<SplitInfo>,
    /// Gradient sum (for debugging)
    #[allow(dead_code)]
    sum_gradients: f32,
    /// Hessian sum (for debugging)
    #[allow(dead_code)]
    sum_hessians: f32,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain().eq(&other.gain())
    }
}

impl Eq for SplitCandidate {}

impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by gain
        self.gain()
            .partial_cmp(&other.gain())
            .unwrap_or(Ordering::Equal)
    }
}

impl SplitCandidate {
    fn gain(&self) -> f32 {
        self.split_info.as_ref().map(|s| s.gain).unwrap_or(f32::NEG_INFINITY)
    }

    #[allow(dead_code)]
    fn count(&self) -> u32 {
        self.row_indices.len() as u32
    }
}

/// Tree grower configuration
#[derive(Debug, Clone)]
pub struct TreeGrower {
    /// Maximum depth of tree
    max_depth: usize,
    /// Maximum number of leaves
    max_leaves: usize,
    /// L2 regularization (lambda)
    lambda: f32,
    /// Minimum samples per leaf
    min_samples_leaf: usize,
    /// Minimum hessian sum per leaf
    min_hessian_leaf: f32,
    /// Shannon Entropy regularization weight
    entropy_weight: f32,
    /// Minimum gain to make a split
    min_gain: f32,
    /// Learning rate (shrinkage)
    learning_rate: f32,
}

impl Default for TreeGrower {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_leaves: 31, // 2^5 - 1
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            entropy_weight: 0.0,
            min_gain: 0.0,
            learning_rate: 0.1,
        }
    }
}

impl TreeGrower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Create a split finder with current configuration
    fn create_split_finder(&self) -> SplitFinder {
        SplitFinder::new()
            .with_lambda(self.lambda)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_hessian_leaf(self.min_hessian_leaf)
            .with_entropy_weight(self.entropy_weight)
            .with_min_gain(self.min_gain)
    }

    /// Grow a tree using Best-First (Leaf-wise) strategy
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Gradient for each sample
    /// * `hessians` - Hessian for each sample
    pub fn grow(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Tree {
        let num_rows = dataset.num_rows();
        let histogram_builder = HistogramBuilder::new();
        let split_finder = self.create_split_finder();

        // Compute initial sums
        let total_gradient: f32 = gradients.iter().sum();
        let total_hessian: f32 = hessians.iter().sum();
        let initial_weight = Node::compute_leaf_weight(total_gradient, total_hessian, self.lambda);

        // Initialize tree with root leaf
        let mut tree = Tree::new(
            initial_weight * self.learning_rate,
            num_rows,
            total_gradient,
            total_hessian,
        );

        // Initialize priority queue with root candidate
        let mut candidates: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Build root histograms
        let all_rows: Vec<usize> = (0..num_rows).collect();
        let root_histograms = histogram_builder.build(dataset, &all_rows, gradients, hessians);

        // Find best split for root
        let root_split = split_finder.find_best_split(
            &root_histograms,
            total_gradient,
            total_hessian,
            num_rows as u32,
        );

        candidates.push(SplitCandidate {
            node_idx: 0,
            row_indices: all_rows,
            histograms: Some(root_histograms),
            split_info: root_split,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
        });

        let mut num_leaves = 1;

        // Best-first growth loop
        while let Some(candidate) = candidates.pop() {
            // Check stopping conditions
            if num_leaves >= self.max_leaves {
                break;
            }

            let split_info = match &candidate.split_info {
                Some(info) if info.is_valid() => info,
                _ => continue, // No valid split
            };

            let current_node = tree.get_node(candidate.node_idx);
            if current_node.depth >= self.max_depth {
                continue;
            }

            // Perform the split: partition row indices
            let (left_rows, right_rows) = self.partition_rows(
                dataset,
                &candidate.row_indices,
                split_info.feature_idx,
                split_info.bin_threshold,
            );

            // Create child leaf nodes
            let left_weight = Node::compute_leaf_weight(
                split_info.left_gradient,
                split_info.left_hessian,
                self.lambda,
            );
            let right_weight = Node::compute_leaf_weight(
                split_info.right_gradient,
                split_info.right_hessian,
                self.lambda,
            );

            let child_depth = current_node.depth + 1;

            let left_node = Node::leaf(
                left_weight * self.learning_rate,
                child_depth,
                left_rows.len(),
                split_info.left_gradient,
                split_info.left_hessian,
            );
            let right_node = Node::leaf(
                right_weight * self.learning_rate,
                child_depth,
                right_rows.len(),
                split_info.right_gradient,
                split_info.right_hessian,
            );

            let left_idx = tree.add_node(left_node);
            let right_idx = tree.add_node(right_node);

            // Convert current leaf to internal node
            let current_node = tree.get_node_mut(candidate.node_idx);
            current_node.node_type = crate::tree::NodeType::Internal {
                feature_idx: split_info.feature_idx,
                bin_threshold: split_info.bin_threshold,
                left_child: left_idx,
                right_child: right_idx,
            };

            num_leaves += 1; // One leaf becomes two (net +1)

            // Skip further splits if we've reached the limit
            if num_leaves >= self.max_leaves {
                break;
            }

            // Build histograms for children using subtraction trick
            let parent_histograms = candidate.histograms.unwrap();

            // Determine smaller child
            let (smaller_rows, smaller_idx, larger_rows, larger_idx, smaller_g, smaller_h, larger_g, larger_h) =
                if left_rows.len() <= right_rows.len() {
                    (
                        left_rows,
                        left_idx,
                        right_rows,
                        right_idx,
                        split_info.left_gradient,
                        split_info.left_hessian,
                        split_info.right_gradient,
                        split_info.right_hessian,
                    )
                } else {
                    (
                        right_rows,
                        right_idx,
                        left_rows,
                        left_idx,
                        split_info.right_gradient,
                        split_info.right_hessian,
                        split_info.left_gradient,
                        split_info.left_hessian,
                    )
                };

            // Build histogram for smaller child directly
            let smaller_histograms =
                histogram_builder.build(dataset, &smaller_rows, gradients, hessians);

            // Compute larger child histogram using subtraction
            let larger_histograms =
                HistogramBuilder::build_sibling(&parent_histograms, &smaller_histograms);

            // Find splits and add to queue
            let smaller_split = split_finder.find_best_split(
                &smaller_histograms,
                smaller_g,
                smaller_h,
                smaller_rows.len() as u32,
            );
            let larger_split = split_finder.find_best_split(
                &larger_histograms,
                larger_g,
                larger_h,
                larger_rows.len() as u32,
            );

            candidates.push(SplitCandidate {
                node_idx: smaller_idx,
                row_indices: smaller_rows,
                histograms: Some(smaller_histograms),
                split_info: smaller_split,
                sum_gradients: smaller_g,
                sum_hessians: smaller_h,
            });

            candidates.push(SplitCandidate {
                node_idx: larger_idx,
                row_indices: larger_rows,
                histograms: Some(larger_histograms),
                split_info: larger_split,
                sum_gradients: larger_g,
                sum_hessians: larger_h,
            });
        }

        tree
    }

    /// Partition rows by split
    fn partition_rows(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        feature_idx: usize,
        bin_threshold: u8,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::with_capacity(row_indices.len() / 2);
        let mut right = Vec::with_capacity(row_indices.len() / 2);

        for &row_idx in row_indices {
            let bin = dataset.get_bin(row_idx, feature_idx);
            if bin <= bin_threshold {
                left.push(row_idx);
            } else {
                right.push(row_idx);
            }
        }

        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FeatureInfo, FeatureType};

    fn create_test_dataset(num_rows: usize, num_features: usize) -> BinnedDataset {
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r * 3 + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| (i as f32).sin()).collect();
        let feature_info = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("f{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 255,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_grow_single_leaf() {
        let dataset = create_test_dataset(100, 3);
        let gradients: Vec<f32> = vec![0.0; 100]; // No gradient = no split benefit
        let hessians: Vec<f32> = vec![1.0; 100];

        let grower = TreeGrower::new()
            .with_max_depth(3)
            .with_min_gain(1000.0); // Very high min gain = no splits

        let tree = grower.grow(&dataset, &gradients, &hessians);

        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.num_nodes(), 1);
    }

    #[test]
    fn test_grow_with_splits() {
        let dataset = create_test_dataset(1000, 5);

        // Create gradients that should encourage splitting
        let gradients: Vec<f32> = (0..1000)
            .map(|i| if i < 500 { -1.0 } else { 1.0 })
            .collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new()
            .with_max_depth(4)
            .with_max_leaves(16)
            .with_min_samples_leaf(10)
            .with_learning_rate(0.1);

        let tree = grower.grow(&dataset, &gradients, &hessians);

        // Should have multiple nodes
        assert!(tree.num_nodes() > 1);
        assert!(tree.num_leaves() > 1);
        assert!(tree.max_depth() <= 4);
    }

    #[test]
    fn test_max_depth_constraint() {
        let dataset = create_test_dataset(1000, 3);
        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new()
            .with_max_depth(2)
            .with_max_leaves(100);

        let tree = grower.grow(&dataset, &gradients, &hessians);

        assert!(tree.max_depth() <= 2);
    }

    #[test]
    fn test_max_leaves_constraint() {
        let dataset = create_test_dataset(1000, 3);
        let gradients: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let hessians: Vec<f32> = vec![1.0; 1000];

        let grower = TreeGrower::new()
            .with_max_depth(10)
            .with_max_leaves(5);

        let tree = grower.grow(&dataset, &gradients, &hessians);

        assert!(tree.num_leaves() <= 5);
    }
}
