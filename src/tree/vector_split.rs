//! Vector split finding for multi-output learning
//!
//! Provides split finding types that support multiple output dimensions.
//! The gain formula for vector outputs is:
//!
//! ```text
//! Gain = Σ_k 0.5 * [G_L,k² / (H_L,k + λ) + G_R,k² / (H_R,k + λ) - G_k² / (H_k + λ)]
//! ```
//!
//! This is the sum of individual gains across all outputs, allowing a single
//! tree split to optimize for all labels/targets simultaneously.

use crate::histogram::{VectorHistogram, VectorNodeHistograms, NUM_BINS};
use rkyv::{Archive, Deserialize, Serialize};

/// Information about a potential split for vector outputs
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VectorSplitInfo {
    /// Number of outputs
    num_outputs: usize,

    /// Feature index
    pub feature_idx: usize,

    /// Bin threshold (samples with bin <= threshold go left)
    pub bin_threshold: u8,

    /// Actual split value for raw prediction (samples with value <= split_value go left)
    /// This is populated from bin_boundaries during tree growth.
    pub split_value: f64,

    /// Total split gain (sum of per-output gains)
    pub gain: f32,

    /// Left child statistics: [grad_0, hess_0, grad_1, hess_1, ...]
    left_stats: Vec<f32>,

    /// Right child statistics: [grad_0, hess_0, grad_1, hess_1, ...]
    right_stats: Vec<f32>,

    /// Left child sample count (shared across outputs)
    pub left_count: u32,

    /// Right child sample count (shared across outputs)
    pub right_count: u32,
}

impl VectorSplitInfo {
    /// Create a new empty split info
    pub fn new(num_outputs: usize) -> Self {
        Self {
            num_outputs,
            feature_idx: 0,
            bin_threshold: 0,
            split_value: 0.0,
            gain: f32::NEG_INFINITY,
            left_stats: vec![0.0; num_outputs * 2],
            right_stats: vec![0.0; num_outputs * 2],
            left_count: 0,
            right_count: 0,
        }
    }

    /// Number of outputs
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Check if this split is valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > f32::NEG_INFINITY && self.left_count > 0 && self.right_count > 0
    }

    /// Get left child (gradient, hessian) for a specific output
    #[inline]
    pub fn left_stats(&self, output_idx: usize) -> (f32, f32) {
        debug_assert!(output_idx < self.num_outputs);
        let offset = output_idx * 2;
        (self.left_stats[offset], self.left_stats[offset + 1])
    }

    /// Get right child (gradient, hessian) for a specific output
    #[inline]
    pub fn right_stats(&self, output_idx: usize) -> (f32, f32) {
        debug_assert!(output_idx < self.num_outputs);
        let offset = output_idx * 2;
        (self.right_stats[offset], self.right_stats[offset + 1])
    }

    /// Set left child statistics for a specific output
    #[inline]
    pub fn set_left_stats(&mut self, output_idx: usize, gradient: f32, hessian: f32) {
        debug_assert!(output_idx < self.num_outputs);
        let offset = output_idx * 2;
        self.left_stats[offset] = gradient;
        self.left_stats[offset + 1] = hessian;
    }

    /// Set right child statistics for a specific output
    #[inline]
    pub fn set_right_stats(&mut self, output_idx: usize, gradient: f32, hessian: f32) {
        debug_assert!(output_idx < self.num_outputs);
        let offset = output_idx * 2;
        self.right_stats[offset] = gradient;
        self.right_stats[offset + 1] = hessian;
    }

    /// Get all left gradients
    pub fn left_gradients(&self) -> Vec<f32> {
        (0..self.num_outputs)
            .map(|k| self.left_stats[k * 2])
            .collect()
    }

    /// Get all left hessians
    pub fn left_hessians(&self) -> Vec<f32> {
        (0..self.num_outputs)
            .map(|k| self.left_stats[k * 2 + 1])
            .collect()
    }

    /// Get all right gradients
    pub fn right_gradients(&self) -> Vec<f32> {
        (0..self.num_outputs)
            .map(|k| self.right_stats[k * 2])
            .collect()
    }

    /// Get all right hessians
    pub fn right_hessians(&self) -> Vec<f32> {
        (0..self.num_outputs)
            .map(|k| self.right_stats[k * 2 + 1])
            .collect()
    }

    /// Get raw left stats buffer
    pub fn left_stats_buffer(&self) -> &[f32] {
        &self.left_stats
    }

    /// Get raw right stats buffer
    pub fn right_stats_buffer(&self) -> &[f32] {
        &self.right_stats
    }
}

impl Default for VectorSplitInfo {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Split finder for vector outputs
pub struct VectorSplitFinder {
    /// Number of outputs
    num_outputs: usize,

    /// L2 regularization parameter (lambda)
    lambda: f32,

    /// Minimum samples in a leaf
    min_samples_leaf: usize,

    /// Minimum hessian sum in a leaf (per output, any output must meet this)
    min_hessian_leaf: f32,

    /// Minimum gain to make a split
    min_gain: f32,
}

impl VectorSplitFinder {
    /// Create a new vector split finder
    pub fn new(num_outputs: usize) -> Self {
        Self {
            num_outputs,
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 0.0,
            min_gain: 0.0,
        }
    }

    /// Set L2 regularization (lambda)
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set minimum hessian sum per leaf
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Find the best split for a single feature
    ///
    /// # Arguments
    /// * `histogram` - Vector histogram for this feature
    /// * `feature_idx` - Index of the feature being evaluated
    ///
    /// # Returns
    /// Split information with the best gain, or invalid split if no good split found
    pub fn find_best_split(&self, histogram: &VectorHistogram, feature_idx: usize) -> VectorSplitInfo {
        debug_assert_eq!(histogram.num_outputs(), self.num_outputs);

        let mut best = VectorSplitInfo::new(self.num_outputs);
        best.feature_idx = feature_idx;

        // Compute total statistics for parent node
        let totals = histogram.totals_all_outputs();
        let total_count = histogram.total_count();

        // Early exit if not enough samples
        if total_count < 2 * self.min_samples_leaf as u32 {
            return best;
        }

        // Compute parent gain for each output (for gain calculation)
        let parent_gains: Vec<f32> = totals
            .iter()
            .map(|(g, h)| {
                if *h + self.lambda > 0.0 {
                    g * g / (h + self.lambda)
                } else {
                    0.0
                }
            })
            .collect();

        // Running cumulative sums for left partition
        // Layout: [grad_0, hess_0, grad_1, hess_1, ...]
        let mut left_sums = vec![0.0f32; self.num_outputs * 2];
        let mut left_count: u32 = 0;

        // Scan through bins to find best split
        for bin in 0..NUM_BINS - 1 {
            let bin_u8 = bin as u8;

            // Add this bin to left partition
            left_count += histogram.get_count(bin_u8);

            for k in 0..self.num_outputs {
                let (bin_grad, bin_hess) = histogram.get_output_stats(bin_u8, k);
                left_sums[k * 2] += bin_grad;
                left_sums[k * 2 + 1] += bin_hess;
            }

            // Skip if left partition doesn't have enough samples
            if left_count < self.min_samples_leaf as u32 {
                continue;
            }

            let right_count = total_count - left_count;

            // Skip if right partition doesn't have enough samples
            if right_count < self.min_samples_leaf as u32 {
                continue;
            }

            // Check minimum hessian constraint (any output must meet threshold)
            let mut hessian_ok = true;
            for k in 0..self.num_outputs {
                let left_hess = left_sums[k * 2 + 1];
                let right_hess = totals[k].1 - left_hess;

                if left_hess < self.min_hessian_leaf || right_hess < self.min_hessian_leaf {
                    hessian_ok = false;
                    break;
                }
            }

            if !hessian_ok {
                continue;
            }

            // Compute split gain = Σ_k [left_gain_k + right_gain_k - parent_gain_k]
            let mut total_gain = 0.0f32;

            for k in 0..self.num_outputs {
                let left_grad = left_sums[k * 2];
                let left_hess = left_sums[k * 2 + 1];
                let right_grad = totals[k].0 - left_grad;
                let right_hess = totals[k].1 - left_hess;

                // Gain for this output
                let left_gain = if left_hess + self.lambda > 0.0 {
                    left_grad * left_grad / (left_hess + self.lambda)
                } else {
                    0.0
                };

                let right_gain = if right_hess + self.lambda > 0.0 {
                    right_grad * right_grad / (right_hess + self.lambda)
                } else {
                    0.0
                };

                // Gain = 0.5 * (left_gain + right_gain - parent_gain)
                total_gain += 0.5 * (left_gain + right_gain - parent_gains[k]);
            }

            // Check if this is the best split so far
            if total_gain > best.gain && total_gain >= self.min_gain {
                best.gain = total_gain;
                best.bin_threshold = bin_u8;
                best.left_count = left_count;
                best.right_count = right_count;

                // Store statistics for each output
                for k in 0..self.num_outputs {
                    let left_grad = left_sums[k * 2];
                    let left_hess = left_sums[k * 2 + 1];
                    let right_grad = totals[k].0 - left_grad;
                    let right_hess = totals[k].1 - left_hess;

                    best.set_left_stats(k, left_grad, left_hess);
                    best.set_right_stats(k, right_grad, right_hess);
                }
            }
        }

        best
    }

    /// Find the best split across all features
    ///
    /// # Arguments
    /// * `node_histograms` - Histograms for all features at this node
    ///
    /// # Returns
    /// Split information for the best feature and threshold
    pub fn find_best_split_all_features(
        &self,
        node_histograms: &VectorNodeHistograms,
    ) -> VectorSplitInfo {
        let mut best = VectorSplitInfo::new(self.num_outputs);

        for (feature_idx, histogram) in node_histograms.iter() {
            let split = self.find_best_split(histogram, feature_idx);
            if split.is_valid() && split.gain > best.gain {
                best = split;
            }
        }

        best
    }

    /// Find the best split across specified features only
    ///
    /// # Arguments
    /// * `node_histograms` - Histograms for all features at this node
    /// * `allowed_features` - Indices of features to consider
    ///
    /// # Returns
    /// Split information for the best feature and threshold
    pub fn find_best_split_features(
        &self,
        node_histograms: &VectorNodeHistograms,
        allowed_features: &[usize],
    ) -> VectorSplitInfo {
        let mut best = VectorSplitInfo::new(self.num_outputs);

        for &feature_idx in allowed_features {
            if feature_idx < node_histograms.num_features() {
                let split = self.find_best_split(node_histograms.get(feature_idx), feature_idx);
                if split.is_valid() && split.gain > best.gain {
                    best = split;
                }
            }
        }

        best
    }

    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_split_info_new() {
        let info = VectorSplitInfo::new(3);
        assert_eq!(info.num_outputs(), 3);
        assert!(!info.is_valid());
        assert_eq!(info.left_stats.len(), 6);
        assert_eq!(info.right_stats.len(), 6);
    }

    #[test]
    fn test_vector_split_finder_new() {
        let finder = VectorSplitFinder::new(2)
            .with_lambda(0.5)
            .with_min_samples_leaf(10)
            .with_min_hessian_leaf(1.0)
            .with_min_gain(0.1);

        assert_eq!(finder.num_outputs(), 2);
        assert_eq!(finder.lambda, 0.5);
        assert_eq!(finder.min_samples_leaf, 10);
        assert_eq!(finder.min_hessian_leaf, 1.0);
        assert_eq!(finder.min_gain, 0.1);
    }

    #[test]
    fn test_simple_split() {
        let num_outputs = 1;
        let mut hist = VectorHistogram::new(num_outputs);

        // Put 5 samples in bin 0 with positive gradient
        for _ in 0..5 {
            hist.accumulate(0, &[1.0], &[1.0]);
        }

        // Put 5 samples in bin 128 with negative gradient
        for _ in 0..5 {
            hist.accumulate(128, &[-1.0], &[1.0]);
        }

        let finder = VectorSplitFinder::new(num_outputs).with_lambda(1.0);

        let split = finder.find_best_split(&hist, 0);

        assert!(split.is_valid());
        assert_eq!(split.bin_threshold, 0);
        assert_eq!(split.left_count, 5);
        assert_eq!(split.right_count, 5);
    }
}
