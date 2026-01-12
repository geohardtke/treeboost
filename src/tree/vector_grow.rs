//! Vector tree growing with Best-First (Leaf-wise) strategy
//!
//! Similar to `TreeGrower` but for multi-output learning where each leaf
//! contains a vector of values (one per output/label).
//!
//! Uses the sum-of-gains criterion: the split gain is the sum of per-output
//! gains, allowing a single tree structure to optimize for all outputs.

use crate::dataset::BinnedDataset;
use crate::histogram::VectorNodeHistograms;
use crate::tree::{VectorNode, VectorNodeType, VectorSplitFinder, VectorSplitInfo, VectorTree};
use crate::Result;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Manages row indices during tree growth with zero-allocation partitioning.
struct RowPartitioner {
    /// All row indices, partitioned by node (contiguous slices)
    rows: Vec<usize>,
}

impl RowPartitioner {
    fn new(initial_rows: Vec<usize>) -> Self {
        Self { rows: initial_rows }
    }

    #[inline]
    fn get_rows(&self, start: usize, end: usize) -> &[usize] {
        &self.rows[start..end]
    }

    /// Partition rows in-place using Dutch National Flag algorithm.
    #[inline]
    fn partition_in_place(
        &mut self,
        dataset: &BinnedDataset,
        start: usize,
        end: usize,
        feature_idx: usize,
        bin_threshold: u8,
    ) -> usize {
        let feature_column = dataset.feature_column(feature_idx);
        let mut left = start;
        let mut right = end;

        unsafe {
            while left < right {
                while left < right {
                    let row_idx = *self.rows.get_unchecked(left);
                    let bin = *feature_column.get_unchecked(row_idx);
                    if bin > bin_threshold {
                        break;
                    }
                    left += 1;
                }

                while left < right {
                    right -= 1;
                    let row_idx = *self.rows.get_unchecked(right);
                    let bin = *feature_column.get_unchecked(row_idx);
                    if bin <= bin_threshold {
                        self.rows.swap(left, right);
                        left += 1;
                        break;
                    }
                }
            }
        }

        left
    }
}

/// Candidate node for splitting
struct VectorSplitCandidate {
    node_idx: usize,
    row_start: usize,
    row_end: usize,
    histograms: Option<VectorNodeHistograms>,
    split_info: Option<VectorSplitInfo>,
    /// Sum of gradients per output (reserved for future use)
    #[allow(dead_code)]
    sum_gradients: Vec<f32>,
    /// Sum of hessians per output (reserved for future use)
    #[allow(dead_code)]
    sum_hessians: Vec<f32>,
}

impl PartialEq for VectorSplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.gain().eq(&other.gain())
    }
}

impl Eq for VectorSplitCandidate {}

impl PartialOrd for VectorSplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VectorSplitCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain()
            .partial_cmp(&other.gain())
            .unwrap_or(Ordering::Equal)
    }
}

impl VectorSplitCandidate {
    fn gain(&self) -> f32 {
        self.split_info
            .as_ref()
            .map(|s| s.gain)
            .unwrap_or(f32::NEG_INFINITY)
    }
}

/// Vector tree grower configuration
#[derive(Debug, Clone)]
pub struct VectorTreeGrower {
    /// Number of outputs
    num_outputs: usize,
    /// Maximum depth of tree
    max_depth: usize,
    /// Maximum number of leaves
    max_leaves: usize,
    /// L2 regularization (lambda)
    lambda: f32,
    /// Minimum samples per leaf
    min_samples_leaf: usize,
    /// Minimum hessian sum per leaf (per output)
    min_hessian_leaf: f32,
    /// Minimum gain to make a split
    min_gain: f32,
    /// Learning rate (shrinkage)
    learning_rate: f32,
}

impl VectorTreeGrower {
    /// Create a new vector tree grower
    pub fn new(num_outputs: usize) -> Self {
        Self {
            num_outputs,
            max_depth: 6,
            max_leaves: 31,
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            min_gain: 0.0,
            learning_rate: 0.1,
        }
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

    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Create split finder with current configuration
    fn create_split_finder(&self) -> VectorSplitFinder {
        VectorSplitFinder::new(self.num_outputs)
            .with_lambda(self.lambda)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_min_hessian_leaf(self.min_hessian_leaf)
            .with_min_gain(self.min_gain)
    }

    /// Build vector histograms for all features at a node
    ///
    /// # Arguments
    /// * `dataset` - Binned dataset
    /// * `gradients` - Flat gradient buffer: `[s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]`
    /// * `hessians` - Flat hessian buffer with same layout
    /// * `row_indices` - Rows in this node
    fn build_vector_histograms(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) -> VectorNodeHistograms {
        let num_features = dataset.num_features();
        let num_outputs = self.num_outputs;

        let mut histograms = VectorNodeHistograms::new(num_features, num_outputs);

        // For each feature, accumulate gradients/hessians into histogram bins
        for feature_idx in 0..num_features {
            let feature_column = dataset.feature_column(feature_idx);
            let hist = histograms.get_mut(feature_idx);

            for &row_idx in row_indices {
                let bin = feature_column[row_idx];
                let offset = row_idx * num_outputs;

                // Get gradient/hessian slices for this sample
                let g_slice = &gradients[offset..offset + num_outputs];
                let h_slice = &hessians[offset..offset + num_outputs];

                hist.accumulate(bin, g_slice, h_slice);
            }
        }

        histograms
    }

    /// Build sibling histograms using subtraction trick
    fn build_vector_histograms_sibling(
        &self,
        parent: &VectorNodeHistograms,
        smaller_child: &VectorNodeHistograms,
    ) -> VectorNodeHistograms {
        VectorNodeHistograms::from_subtraction(parent, smaller_child)
    }

    /// Grow a vector tree
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Flat gradient buffer: `[s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]`
    /// * `hessians` - Flat hessian buffer with same layout
    pub fn grow(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<VectorTree> {
        let all_rows: Vec<usize> = (0..dataset.num_rows()).collect();
        self.grow_with_indices(dataset, gradients, hessians, &all_rows)
    }

    /// Grow a vector tree using only the specified row indices
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Flat gradient buffer: `[s0_o0, s0_o1, ..., s1_o0, s1_o1, ...]`
    /// * `hessians` - Flat hessian buffer with same layout
    /// * `row_indices` - Subset of row indices to use for training
    pub fn grow_with_indices(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) -> Result<VectorTree> {
        let _num_features = dataset.num_features();
        let num_rows = row_indices.len();
        let num_outputs = self.num_outputs;

        let split_finder = self.create_split_finder();

        // Compute initial sums for all outputs
        let mut total_gradients = vec![0.0f32; num_outputs];
        let mut total_hessians = vec![0.0f32; num_outputs];

        for &row_idx in row_indices {
            let offset = row_idx * num_outputs;
            for k in 0..num_outputs {
                total_gradients[k] += gradients[offset + k];
                total_hessians[k] += hessians[offset + k];
            }
        }

        // Compute initial leaf weights
        let initial_weights: Vec<f32> = total_gradients
            .iter()
            .zip(total_hessians.iter())
            .map(|(g, h)| -g / (h + self.lambda) * self.learning_rate)
            .collect();

        // Initialize tree with root leaf
        let mut tree = VectorTree::new(
            initial_weights,
            num_rows,
            total_gradients.clone(),
            total_hessians.clone(),
        );

        // Initialize priority queue with root candidate
        let mut candidates: BinaryHeap<VectorSplitCandidate> = BinaryHeap::new();

        // Initialize row partitioner
        let mut partitioner = RowPartitioner::new(row_indices.to_vec());

        // Build root histograms
        let root_histograms = self.build_vector_histograms(
            dataset,
            gradients,
            hessians,
            partitioner.get_rows(0, num_rows),
        );

        // Find best split for root
        let root_split = split_finder.find_best_split_all_features(&root_histograms);

        candidates.push(VectorSplitCandidate {
            node_idx: 0,
            row_start: 0,
            row_end: num_rows,
            histograms: Some(root_histograms),
            split_info: if root_split.is_valid() {
                Some(root_split)
            } else {
                None
            },
            sum_gradients: total_gradients,
            sum_hessians: total_hessians,
        });

        let mut num_leaves = 1;

        // Best-first growth loop
        while !candidates.is_empty() && num_leaves < self.max_leaves {
            let candidate = candidates.pop().unwrap();

            // Check if candidate has valid split
            let split_info = match &candidate.split_info {
                Some(info) if info.is_valid() => info.clone(),
                _ => continue,
            };

            // Check depth constraint
            let current_node = tree.get_node(candidate.node_idx);
            if current_node.depth >= self.max_depth {
                continue;
            }

            // Check leaf limit
            if num_leaves >= self.max_leaves {
                break;
            }

            // Perform the split: partition rows in-place
            let mid = partitioner.partition_in_place(
                dataset,
                candidate.row_start,
                candidate.row_end,
                split_info.feature_idx,
                split_info.bin_threshold,
            );

            let left_start = candidate.row_start;
            let left_end = mid;
            let right_start = mid;
            let right_end = candidate.row_end;
            let left_count = left_end - left_start;
            let right_count = right_end - right_start;

            // Create child leaf nodes
            let left_gradients = split_info.left_gradients();
            let left_hessians = split_info.left_hessians();
            let right_gradients = split_info.right_gradients();
            let right_hessians = split_info.right_hessians();

            let left_weights: Vec<f32> = left_gradients
                .iter()
                .zip(left_hessians.iter())
                .map(|(g, h)| -g / (h + self.lambda) * self.learning_rate)
                .collect();

            let right_weights: Vec<f32> = right_gradients
                .iter()
                .zip(right_hessians.iter())
                .map(|(g, h)| -g / (h + self.lambda) * self.learning_rate)
                .collect();

            let child_depth = current_node.depth + 1;

            let left_node = VectorNode::leaf(
                left_weights,
                child_depth,
                left_count,
                left_gradients.clone(),
                left_hessians.clone(),
            );
            let right_node = VectorNode::leaf(
                right_weights,
                child_depth,
                right_count,
                right_gradients.clone(),
                right_hessians.clone(),
            );

            let left_idx = tree.add_node(left_node);
            let right_idx = tree.add_node(right_node);

            // Get the actual split value from bin boundaries
            let split_value =
                dataset.get_split_value(split_info.feature_idx, split_info.bin_threshold);

            // Convert current leaf to internal node
            let current_node = tree.get_node_mut(candidate.node_idx);
            current_node.node_type = VectorNodeType::Internal {
                feature_idx: split_info.feature_idx,
                bin_threshold: split_info.bin_threshold,
                split_value,
                left_child: left_idx,
                right_child: right_idx,
            };

            num_leaves += 1;

            // Skip histogram building if we've hit the leaf limit
            if num_leaves >= self.max_leaves {
                break;
            }

            // Determine smaller child for histogram subtraction trick
            let parent_histograms = candidate.histograms.unwrap();
            let (
                smaller_start,
                smaller_end,
                smaller_idx,
                larger_start,
                larger_end,
                larger_idx,
                smaller_g,
                smaller_h,
                larger_g,
                larger_h,
            ) = if left_count <= right_count {
                (
                    left_start,
                    left_end,
                    left_idx,
                    right_start,
                    right_end,
                    right_idx,
                    left_gradients,
                    left_hessians,
                    right_gradients,
                    right_hessians,
                )
            } else {
                (
                    right_start,
                    right_end,
                    right_idx,
                    left_start,
                    left_end,
                    left_idx,
                    right_gradients,
                    right_hessians,
                    left_gradients,
                    left_hessians,
                )
            };

            // Build histograms for smaller child
            let smaller_histograms = self.build_vector_histograms(
                dataset,
                gradients,
                hessians,
                partitioner.get_rows(smaller_start, smaller_end),
            );

            // Compute larger child histogram using subtraction trick
            let larger_histograms =
                self.build_vector_histograms_sibling(&parent_histograms, &smaller_histograms);

            // Find splits for children
            let smaller_split = split_finder.find_best_split_all_features(&smaller_histograms);
            let larger_split = split_finder.find_best_split_all_features(&larger_histograms);

            candidates.push(VectorSplitCandidate {
                node_idx: smaller_idx,
                row_start: smaller_start,
                row_end: smaller_end,
                histograms: Some(smaller_histograms),
                split_info: if smaller_split.is_valid() {
                    Some(smaller_split)
                } else {
                    None
                },
                sum_gradients: smaller_g,
                sum_hessians: smaller_h,
            });

            candidates.push(VectorSplitCandidate {
                node_idx: larger_idx,
                row_start: larger_start,
                row_end: larger_end,
                histograms: Some(larger_histograms),
                split_info: if larger_split.is_valid() {
                    Some(larger_split)
                } else {
                    None
                },
                sum_gradients: larger_g,
                sum_hessians: larger_h,
            });
        }

        Ok(tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FeatureInfo;

    fn create_test_dataset() -> BinnedDataset {
        use crate::dataset::FeatureType;

        // Simple dataset: 10 rows, 2 features
        let num_rows = 10;
        let num_features = 2;

        // Feature 0: bins 0-4 for first half, 5-9 for second half
        // Feature 1: bins 0-2 for first third, 3-6 for second third, 7-9 for last third
        let features: Vec<u8> = vec![
            // Feature 0
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // Feature 1
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        ];

        let targets = vec![0.0f32; num_rows]; // Dummy targets

        let feature_info: Vec<FeatureInfo> = (0..num_features)
            .map(|i| FeatureInfo {
                name: format!("feature_{}", i),
                feature_type: FeatureType::Numeric,
                num_bins: 10,
                bin_boundaries: vec![],
            })
            .collect();

        BinnedDataset::new(num_rows, features, targets, feature_info)
    }

    #[test]
    fn test_vector_tree_grower_creation() {
        let grower = VectorTreeGrower::new(3)
            .with_max_depth(4)
            .with_max_leaves(16)
            .with_lambda(0.5)
            .with_learning_rate(0.05);

        assert_eq!(grower.num_outputs(), 3);
    }

    #[test]
    fn test_grow_simple_tree() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();
        let num_outputs = 2;

        // Create gradients/hessians that encourage a split at feature 0, bin 4
        // First 5 samples: positive gradient for output 0, negative for output 1
        // Last 5 samples: negative gradient for output 0, positive for output 1
        let mut gradients = vec![0.0f32; num_rows * num_outputs];
        let hessians = vec![1.0f32; num_rows * num_outputs]; // Constant hessians

        for i in 0..5 {
            gradients[i * num_outputs] = 1.0; // output 0
            gradients[i * num_outputs + 1] = -1.0; // output 1
        }
        for i in 5..10 {
            gradients[i * num_outputs] = -1.0; // output 0
            gradients[i * num_outputs + 1] = 1.0; // output 1
        }

        let grower = VectorTreeGrower::new(num_outputs)
            .with_max_depth(3)
            .with_min_gain(0.0)
            .with_learning_rate(1.0);

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        // Tree should have at least one split
        assert!(tree.num_nodes() > 1);
        assert_eq!(tree.num_outputs(), num_outputs);
    }

    #[test]
    fn test_prediction() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();
        let num_outputs = 2;

        let mut gradients = vec![0.0f32; num_rows * num_outputs];
        let hessians = vec![1.0f32; num_rows * num_outputs];

        for i in 0..5 {
            gradients[i * num_outputs] = 1.0;
            gradients[i * num_outputs + 1] = -1.0;
        }
        for i in 5..10 {
            gradients[i * num_outputs] = -1.0;
            gradients[i * num_outputs + 1] = 1.0;
        }

        let grower = VectorTreeGrower::new(num_outputs)
            .with_max_depth(3)
            .with_learning_rate(1.0);

        let tree = grower.grow(&dataset, &gradients, &hessians).unwrap();

        // Predictions should be vectors of length num_outputs
        let pred = tree.predict(|f| dataset.get_bin(0, f));
        assert_eq!(pred.len(), num_outputs);

        // Batch prediction
        let mut predictions = vec![0.0f32; num_rows * num_outputs];
        tree.predict_batch_add(
            |sample, feat| dataset.get_bin(sample, feat),
            num_rows,
            &mut predictions,
        );

        // All predictions should be updated
        assert!(predictions.iter().any(|&p| p != 0.0));
    }
}
