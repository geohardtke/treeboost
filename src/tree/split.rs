//! Split finding with Shannon Entropy regularization

use crate::histogram::{Histogram, NodeHistograms};

/// Information about a potential split
#[derive(Debug, Clone, Copy)]
pub struct SplitInfo {
    /// Feature index
    pub feature_idx: usize,
    /// Bin threshold (samples with bin <= threshold go left)
    pub bin_threshold: u8,
    /// Split gain (higher is better)
    pub gain: f32,
    /// Left child gradient sum
    pub left_gradient: f32,
    /// Left child hessian sum
    pub left_hessian: f32,
    /// Left child sample count
    pub left_count: u32,
    /// Right child gradient sum
    pub right_gradient: f32,
    /// Right child hessian sum
    pub right_hessian: f32,
    /// Right child sample count
    pub right_count: u32,
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self {
            feature_idx: 0,
            bin_threshold: 0,
            gain: f32::NEG_INFINITY,
            left_gradient: 0.0,
            left_hessian: 0.0,
            left_count: 0,
            right_gradient: 0.0,
            right_hessian: 0.0,
            right_count: 0,
        }
    }
}

impl SplitInfo {
    /// Check if this split is valid
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.gain > f32::NEG_INFINITY
            && self.left_count > 0
            && self.right_count > 0
    }
}

/// Split finder with Shannon Entropy regularization
pub struct SplitFinder {
    /// L2 regularization parameter (lambda)
    lambda: f32,
    /// Minimum samples in a leaf
    min_samples_leaf: usize,
    /// Minimum hessian sum in a leaf
    min_hessian_leaf: f32,
    /// Shannon Entropy regularization weight (beta)
    entropy_weight: f32,
    /// Minimum gain to make a split
    min_gain: f32,
}

impl Default for SplitFinder {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            entropy_weight: 0.0, // No entropy regularization by default
            min_gain: 0.0,
        }
    }
}

impl SplitFinder {
    /// Create a new split finder
    pub fn new() -> Self {
        Self::default()
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

    /// Set Shannon Entropy regularization weight (beta)
    ///
    /// Higher values penalize imbalanced splits more,
    /// making the tree more robust to concept drift.
    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Find the best split across all features
    pub fn find_best_split(
        &self,
        histograms: &NodeHistograms,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        self.find_best_split_with_features(histograms, total_gradient, total_hessian, total_count, None)
    }

    /// Find the best split across a subset of features (for column subsampling)
    ///
    /// # Arguments
    /// * `histograms` - Node histograms for all features
    /// * `total_gradient` - Sum of gradients in the node
    /// * `total_hessian` - Sum of hessians in the node
    /// * `total_count` - Number of samples in the node
    /// * `feature_mask` - Optional mask of feature indices to consider (None = all features)
    pub fn find_best_split_with_features(
        &self,
        histograms: &NodeHistograms,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
        feature_mask: Option<&[usize]>,
    ) -> Option<SplitInfo> {
        let mut best_split = SplitInfo::default();

        match feature_mask {
            Some(features) => {
                // Only consider specified features
                for &feature_idx in features {
                    if feature_idx >= histograms.num_features() {
                        continue;
                    }
                    if let Some(split) = self.find_best_split_for_feature(
                        histograms.get(feature_idx),
                        feature_idx,
                        total_gradient,
                        total_hessian,
                        total_count,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
            None => {
                // Consider all features
                for (feature_idx, histogram) in histograms.iter() {
                    if let Some(split) = self.find_best_split_for_feature(
                        histogram,
                        feature_idx,
                        total_gradient,
                        total_hessian,
                        total_count,
                    ) {
                        if split.gain > best_split.gain {
                            best_split = split;
                        }
                    }
                }
            }
        }

        if best_split.is_valid() && best_split.gain >= self.min_gain {
            Some(best_split)
        } else {
            None
        }
    }

    /// Find the best split for a single feature
    fn find_best_split_for_feature(
        &self,
        histogram: &Histogram,
        feature_idx: usize,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        let mut best_split = SplitInfo::default();
        best_split.feature_idx = feature_idx;

        // Cumulative sums for left child
        let mut left_gradient = 0.0f32;
        let mut left_hessian = 0.0f32;
        let mut left_count = 0u32;

        // Scan through bins
        for (bin, entry) in histogram.iter() {
            if entry.count == 0 {
                continue;
            }

            left_gradient += entry.sum_gradients;
            left_hessian += entry.sum_hessians;
            left_count += entry.count;

            // Right child = total - left
            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;
            let right_count = total_count - left_count;

            // Check leaf constraints
            if (left_count as usize) < self.min_samples_leaf
                || (right_count as usize) < self.min_samples_leaf
            {
                continue;
            }
            if left_hessian < self.min_hessian_leaf || right_hessian < self.min_hessian_leaf {
                continue;
            }

            // Compute gain with entropy regularization
            let gain = self.compute_gain(
                left_gradient,
                left_hessian,
                right_gradient,
                right_hessian,
                total_gradient,
                total_hessian,
                left_count,
                right_count,
                total_count,
            );

            if gain > best_split.gain {
                best_split.bin_threshold = bin;
                best_split.gain = gain;
                best_split.left_gradient = left_gradient;
                best_split.left_hessian = left_hessian;
                best_split.left_count = left_count;
                best_split.right_gradient = right_gradient;
                best_split.right_hessian = right_hessian;
                best_split.right_count = right_count;
            }
        }

        if best_split.is_valid() {
            Some(best_split)
        } else {
            None
        }
    }

    /// Compute split gain with optional Shannon Entropy regularization
    ///
    /// Standard gain (Friedman MSE):
    /// gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)]
    ///
    /// With entropy regularization:
    /// gain = standard_gain + β * H(split)
    ///
    /// Where H(split) = -p*log(p) - (1-p)*log(1-p), p = n_L/n
    fn compute_gain(
        &self,
        left_g: f32,
        left_h: f32,
        right_g: f32,
        right_h: f32,
        total_g: f32,
        total_h: f32,
        left_count: u32,
        right_count: u32,
        total_count: u32,
    ) -> f32 {
        // Friedman MSE gain (standard GBDT gain)
        let left_score = (left_g * left_g) / (left_h + self.lambda);
        let right_score = (right_g * right_g) / (right_h + self.lambda);
        let parent_score = (total_g * total_g) / (total_h + self.lambda);

        let standard_gain = 0.5 * (left_score + right_score - parent_score);

        // Add Shannon Entropy regularization if enabled
        if self.entropy_weight > 0.0 {
            let entropy = self.split_entropy(left_count, right_count, total_count);
            standard_gain + self.entropy_weight * entropy
        } else {
            standard_gain
        }
    }

    /// Compute Shannon Entropy of a split
    ///
    /// H(split) = -p*log₂(p) - (1-p)*log₂(1-p)
    ///
    /// Maximum entropy (1.0) when p = 0.5 (balanced split)
    /// Minimum entropy (0.0) when p = 0 or p = 1 (completely imbalanced)
    fn split_entropy(&self, left_count: u32, right_count: u32, total_count: u32) -> f32 {
        if total_count == 0 || left_count == 0 || right_count == 0 {
            return 0.0;
        }

        let p = left_count as f32 / total_count as f32;
        let q = 1.0 - p;

        // Use safe log computation
        let h = if p > 0.0 && q > 0.0 {
            -(p * p.log2() + q * q.log2())
        } else {
            0.0
        };

        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::Histogram;

    #[test]
    fn test_split_finder_basic() {
        let mut histogram = Histogram::new();

        // Create a clear split: bins 0-127 have negative gradients, 128-255 have positive
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0);
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0);
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        let finder = SplitFinder::new().with_lambda(0.0);
        let split = finder
            .find_best_split(&histograms, 0.0, 256.0, 256)
            .unwrap();

        // Best split should be around bin 127
        assert_eq!(split.feature_idx, 0);
        assert_eq!(split.bin_threshold, 127);
        assert!(split.gain > 0.0);
    }

    #[test]
    fn test_entropy_regularization() {
        let mut histogram = Histogram::new();

        // Imbalanced split: most samples on one side
        for _ in 0..10 {
            histogram.accumulate(0, -0.5, 1.0);
        }
        for _ in 0..90 {
            histogram.accumulate(255, 0.5, 1.0);
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // Without entropy regularization
        let finder_no_entropy = SplitFinder::new().with_lambda(0.0);
        let split_no_entropy = finder_no_entropy
            .find_best_split(&histograms, 40.0, 100.0, 100);

        // With entropy regularization (penalizes imbalanced splits)
        let finder_entropy = SplitFinder::new()
            .with_lambda(0.0)
            .with_entropy_weight(10.0);
        let split_entropy = finder_entropy
            .find_best_split(&histograms, 40.0, 100.0, 100);

        // Both should find splits, but gains differ
        assert!(split_no_entropy.is_some());
        assert!(split_entropy.is_some());
    }

    #[test]
    fn test_split_entropy() {
        let finder = SplitFinder::new();

        // Balanced split: maximum entropy
        let balanced = finder.split_entropy(50, 50, 100);
        assert!((balanced - 1.0).abs() < 0.001);

        // Imbalanced split: lower entropy
        let imbalanced = finder.split_entropy(10, 90, 100);
        assert!(imbalanced < 0.5);

        // Edge cases
        assert_eq!(finder.split_entropy(0, 100, 100), 0.0);
        assert_eq!(finder.split_entropy(100, 0, 100), 0.0);
    }

    #[test]
    fn test_min_samples_constraint() {
        let mut histogram = Histogram::new();

        histogram.accumulate(0, -1.0, 1.0); // 1 sample
        for _ in 0..99 {
            histogram.accumulate(255, 1.0, 1.0); // 99 samples
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        let finder = SplitFinder::new()
            .with_min_samples_leaf(5); // Require at least 5 samples

        let split = finder.find_best_split(&histograms, 98.0, 100.0, 100);

        // Should not find split because left would only have 1 sample
        assert!(split.is_none());
    }
}
