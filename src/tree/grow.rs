//! Tree growing with Best-First (Leaf-wise) strategy
//!
//! Uses a sorted row array with position tracking for zero-allocation partitioning.
//! This is the LightGBM approach: a single Vec<usize> contains all row indices,
//! partitioned in-place by node. Each node tracks its range (start, end) into this array.

use crate::dataset::BinnedDataset;
use crate::histogram::{HistogramBuilder, NodeHistograms};
use crate::tree::{InteractionConstraints, MonotonicConstraint, Node, SplitFinder, SplitInfo, Tree};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Manages row indices during tree growth with zero-allocation partitioning.
///
/// Instead of storing row indices per node, we maintain a single sorted array
/// where each node's rows are a contiguous slice. Partitioning is done in-place
/// using the Dutch National Flag algorithm.
struct RowPartitioner {
    /// All row indices, partitioned by node (contiguous slices)
    rows: Vec<usize>,
}

impl RowPartitioner {
    /// Create a new partitioner from initial row indices
    fn new(initial_rows: Vec<usize>) -> Self {
        Self { rows: initial_rows }
    }

    /// Get a slice of rows for the given range
    #[inline]
    fn get_rows(&self, start: usize, end: usize) -> &[usize] {
        &self.rows[start..end]
    }

    /// Partition rows in-place using Dutch National Flag algorithm.
    ///
    /// After partitioning, rows[start..mid] go left, rows[mid..end] go right.
    /// Returns the midpoint index.
    fn partition_in_place(
        &mut self,
        dataset: &BinnedDataset,
        start: usize,
        end: usize,
        feature_idx: usize,
        bin_threshold: u8,
    ) -> usize {
        // Dutch National Flag: partition in-place with single pass
        // Left pointer scans right, right pointer scans left
        let mut left = start;
        let mut right = end;

        while left < right {
            // Find first row from left that should go right
            while left < right {
                let row_idx = self.rows[left];
                let bin = dataset.get_bin(row_idx, feature_idx);
                if bin > bin_threshold {
                    break;
                }
                left += 1;
            }

            // Find first row from right that should go left
            while left < right {
                right -= 1;
                let row_idx = self.rows[right];
                let bin = dataset.get_bin(row_idx, feature_idx);
                if bin <= bin_threshold {
                    // Swap and continue
                    self.rows.swap(left, right);
                    left += 1;
                    break;
                }
            }
        }

        left // Midpoint: rows[start..left] go left, rows[left..end] go right
    }
}

/// Candidate node for splitting (zero-allocation version)
#[derive(Debug)]
struct SplitCandidate {
    /// Node index in tree
    node_idx: usize,
    /// Start index in RowPartitioner.rows (inclusive)
    row_start: usize,
    /// End index in RowPartitioner.rows (exclusive)
    row_end: usize,
    /// Precomputed histograms (if available)
    histograms: Option<NodeHistograms>,
    /// Best split info (if computed)
    split_info: Option<SplitInfo>,
    /// Gradient sum
    #[allow(dead_code)]
    sum_gradients: f32,
    /// Hessian sum
    #[allow(dead_code)]
    sum_hessians: f32,
    /// Features used in ancestors (for interaction constraints)
    ancestor_features: Vec<usize>,
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

    #[inline]
    fn row_count(&self) -> usize {
        self.row_end - self.row_start
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
    /// Column subsampling ratio (0.0-1.0, 1.0 = use all features)
    colsample: f32,
    /// Monotonic constraints per feature
    monotonic_constraints: Vec<MonotonicConstraint>,
    /// Feature interaction constraints
    interaction_constraints: InteractionConstraints,
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
            colsample: 1.0, // Use all features by default
            monotonic_constraints: Vec::new(),
            interaction_constraints: InteractionConstraints::new(),
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

    pub fn with_colsample(mut self, colsample: f32) -> Self {
        self.colsample = colsample;
        self
    }

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Features in the same group can interact (appear together in a tree path).
    /// Features in different groups cannot be used together.
    pub fn with_interaction_constraints(mut self, constraints: InteractionConstraints) -> Self {
        self.interaction_constraints = constraints;
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
            .with_monotonic_constraints(self.monotonic_constraints.clone())
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
        // Use all rows
        let all_rows: Vec<usize> = (0..dataset.num_rows()).collect();
        self.grow_with_indices(dataset, gradients, hessians, &all_rows)
    }

    /// Grow a tree using only the specified row indices (for row subsampling)
    ///
    /// # Arguments
    /// * `dataset` - Binned training data
    /// * `gradients` - Gradient for each sample
    /// * `hessians` - Hessian for each sample
    /// * `row_indices` - Subset of row indices to use for training this tree
    pub fn grow_with_indices(
        &self,
        dataset: &BinnedDataset,
        gradients: &[f32],
        hessians: &[f32],
        row_indices: &[usize],
    ) -> Tree {
        let num_features = dataset.num_features();
        let num_rows = row_indices.len();
        let histogram_builder = HistogramBuilder::new();
        let split_finder = self.create_split_finder();

        // Generate column subsample mask (per tree)
        let feature_mask: Option<Vec<usize>> = if self.colsample < 1.0 {
            let n_features = ((num_features as f32) * self.colsample).ceil() as usize;
            let n_features = n_features.max(1); // At least one feature
            let mut rng = rand::rngs::StdRng::seed_from_u64(
                // Use row count as seed variation per tree
                (num_rows as u64).wrapping_mul(31337)
            );
            let mut all_features: Vec<usize> = (0..num_features).collect();
            all_features.shuffle(&mut rng);
            all_features.truncate(n_features);
            Some(all_features)
        } else {
            None
        };

        // Compute initial sums for the subsampled rows only
        let total_gradient: f32 = row_indices.iter().map(|&i| gradients[i]).sum();
        let total_hessian: f32 = row_indices.iter().map(|&i| hessians[i]).sum();
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

        // Initialize row partitioner with all rows (single allocation for entire tree growth)
        let mut partitioner = RowPartitioner::new(row_indices.to_vec());

        // Build root histograms from subsampled rows (feature-parallel via Rayon)
        let root_histograms = histogram_builder.build(
            dataset,
            partitioner.get_rows(0, num_rows),
            gradients,
            hessians,
        );

        // Compute effective feature mask for root (no ancestors)
        let root_feature_mask = self.compute_effective_feature_mask(
            &[],
            feature_mask.as_deref(),
            num_features,
        );

        // Find best split for root
        let root_split = split_finder.find_best_split_with_features(
            &root_histograms,
            total_gradient,
            total_hessian,
            num_rows as u32,
            root_feature_mask.as_deref(),
        );

        candidates.push(SplitCandidate {
            node_idx: 0,
            row_start: 0,
            row_end: num_rows,
            histograms: Some(root_histograms),
            split_info: root_split,
            sum_gradients: total_gradient,
            sum_hessians: total_hessian,
            ancestor_features: Vec::new(),
        });

        let mut num_leaves = 1;

        // Best-first growth loop
        while let Some(candidate) = candidates.pop() {
            // Check stopping conditions
            if num_leaves >= self.max_leaves {
                break;
            }

            let split_info = match &candidate.split_info {
                Some(info) if info.is_valid() => info.clone(),
                _ => continue, // No valid split
            };

            let current_node = tree.get_node(candidate.node_idx);
            if current_node.depth >= self.max_depth {
                continue;
            }

            // Perform the split: partition rows in-place (zero allocation!)
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
                left_count,
                split_info.left_gradient,
                split_info.left_hessian,
            );
            let right_node = Node::leaf(
                right_weight * self.learning_rate,
                child_depth,
                right_count,
                split_info.right_gradient,
                split_info.right_hessian,
            );

            let left_idx = tree.add_node(left_node);
            let right_idx = tree.add_node(right_node);

            // Get the actual split value from bin boundaries
            let split_value = dataset.get_split_value(
                split_info.feature_idx,
                split_info.bin_threshold,
            );

            // Convert current leaf to internal node
            let current_node = tree.get_node_mut(candidate.node_idx);
            current_node.node_type = crate::tree::NodeType::Internal {
                feature_idx: split_info.feature_idx,
                bin_threshold: split_info.bin_threshold,
                split_value,
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

            // Determine smaller child for histogram subtraction trick
            let (smaller_start, smaller_end, smaller_idx, larger_start, larger_end, larger_idx, smaller_g, smaller_h, larger_g, larger_h) =
                if left_count <= right_count {
                    (
                        left_start,
                        left_end,
                        left_idx,
                        right_start,
                        right_end,
                        right_idx,
                        split_info.left_gradient,
                        split_info.left_hessian,
                        split_info.right_gradient,
                        split_info.right_hessian,
                    )
                } else {
                    (
                        right_start,
                        right_end,
                        right_idx,
                        left_start,
                        left_end,
                        left_idx,
                        split_info.right_gradient,
                        split_info.right_hessian,
                        split_info.left_gradient,
                        split_info.left_hessian,
                    )
                };

            let smaller_count = smaller_end - smaller_start;
            let larger_count = larger_end - larger_start;

            // Build histogram for smaller child directly (feature-parallel via Rayon)
            let smaller_histograms = histogram_builder.build(
                dataset,
                partitioner.get_rows(smaller_start, smaller_end),
                gradients,
                hessians,
            );

            // Compute larger child histogram using subtraction
            let larger_histograms =
                HistogramBuilder::build_sibling(&parent_histograms, &smaller_histograms);

            // Compute child ancestor features (parent's ancestors + current split feature)
            let mut child_ancestors = candidate.ancestor_features.clone();
            child_ancestors.push(split_info.feature_idx);

            // Compute effective feature mask for children (interaction + column subsampling)
            let child_feature_mask = self.compute_effective_feature_mask(
                &child_ancestors,
                feature_mask.as_deref(),
                num_features,
            );

            // Find splits for children
            let smaller_split = split_finder.find_best_split_with_features(
                &smaller_histograms,
                smaller_g,
                smaller_h,
                smaller_count as u32,
                child_feature_mask.as_deref(),
            );
            let larger_split = split_finder.find_best_split_with_features(
                &larger_histograms,
                larger_g,
                larger_h,
                larger_count as u32,
                child_feature_mask.as_deref(),
            );

            candidates.push(SplitCandidate {
                node_idx: smaller_idx,
                row_start: smaller_start,
                row_end: smaller_end,
                histograms: Some(smaller_histograms),
                split_info: smaller_split,
                sum_gradients: smaller_g,
                sum_hessians: smaller_h,
                ancestor_features: child_ancestors.clone(),
            });

            candidates.push(SplitCandidate {
                node_idx: larger_idx,
                row_start: larger_start,
                row_end: larger_end,
                histograms: Some(larger_histograms),
                split_info: larger_split,
                sum_gradients: larger_g,
                sum_hessians: larger_h,
                ancestor_features: child_ancestors,
            });
        }

        tree
    }

    /// Compute effective feature mask combining interaction constraints and column subsampling
    ///
    /// Returns None if all features are allowed, Some(mask) otherwise
    fn compute_effective_feature_mask(
        &self,
        ancestor_features: &[usize],
        colsample_mask: Option<&[usize]>,
        num_features: usize,
    ) -> Option<Vec<usize>> {
        // Get interaction-allowed features
        let interaction_allowed = if self.interaction_constraints.is_empty() {
            None
        } else {
            Some(self.interaction_constraints.allowed_features(ancestor_features, num_features))
        };

        // Combine with column subsampling mask
        match (interaction_allowed, colsample_mask) {
            (None, None) => None, // No constraints
            (Some(allowed), None) => Some(allowed),
            (None, Some(mask)) => Some(mask.to_vec()),
            (Some(allowed), Some(mask)) => {
                // Intersection of both constraints
                let allowed_set: std::collections::HashSet<_> = allowed.into_iter().collect();
                let combined: Vec<usize> = mask
                    .iter()
                    .copied()
                    .filter(|f| allowed_set.contains(f))
                    .collect();
                if combined.is_empty() {
                    // Edge case: no features allowed - return the interaction allowed set
                    // to let the algorithm gracefully stop
                    Some(self.interaction_constraints.allowed_features(ancestor_features, num_features))
                } else {
                    Some(combined)
                }
            }
        }
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
