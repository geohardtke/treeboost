//! Split finding with Shannon Entropy regularization
//!
//! This module uses SIMD-optimized kernels for split finding when possible.
//! The SIMD path is taken when:
//! - No entropy regularization (entropy_weight == 0.0)
//! - No monotonic constraints
//!
//! Otherwise falls back to scalar implementation with full feature support.

use crate::histogram::{Histogram, NodeHistograms, NUM_BINS};
use crate::kernel::find_best_split as kernel_find_best_split;
use rkyv::{Archive, Deserialize, Serialize};

/// Monotonic constraint for a feature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
pub enum MonotonicConstraint {
    /// No constraint
    None,
    /// Prediction must increase as feature value increases
    Increasing,
    /// Prediction must decrease as feature value increases
    Decreasing,
}

/// Feature interaction constraints
///
/// Defines which features can interact (appear together in the same tree path).
/// Features in the same group can interact; features in different groups cannot.
/// Features not in any group can interact with all features.
#[derive(Debug, Clone, Default)]
pub struct InteractionConstraints {
    /// Groups of feature indices that can interact
    /// Each inner Vec is a group of features that can appear together
    groups: Vec<Vec<usize>>,
    /// Lookup: feature_idx -> group_idx (None if unconstrained)
    feature_to_group: Vec<Option<usize>>,
}

impl InteractionConstraints {
    /// Create empty constraints (all features can interact)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create constraints from groups of feature indices
    ///
    /// # Arguments
    /// * `groups` - Each group is a list of feature indices that can interact
    /// * `num_features` - Total number of features in the dataset
    ///
    /// # Example
    /// ```ignore
    /// // Features 0,1,2 can interact; features 3,4 can interact; 5 is unconstrained
    /// let constraints = InteractionConstraints::from_groups(
    ///     vec![vec![0, 1, 2], vec![3, 4]],
    ///     6
    /// );
    /// ```
    pub fn from_groups(groups: Vec<Vec<usize>>, num_features: usize) -> Self {
        let mut feature_to_group = vec![None; num_features];

        for (group_idx, group) in groups.iter().enumerate() {
            for &feature_idx in group {
                if feature_idx < num_features {
                    feature_to_group[feature_idx] = Some(group_idx);
                }
            }
        }

        Self {
            groups,
            feature_to_group,
        }
    }

    /// Check if constraints are empty (all features can interact)
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Get allowed features given the ancestor path
    ///
    /// # Arguments
    /// * `ancestor_features` - Features used in ancestors of current node
    /// * `num_features` - Total number of features
    ///
    /// # Returns
    /// List of feature indices that can be used for splitting
    pub fn allowed_features(&self, ancestor_features: &[usize], num_features: usize) -> Vec<usize> {
        if self.groups.is_empty() {
            // No constraints: all features allowed
            return (0..num_features).collect();
        }

        // Find which groups have been used by ancestors
        let mut used_groups: Vec<bool> = vec![false; self.groups.len()];
        for &feat in ancestor_features {
            if let Some(group_idx) = self.feature_to_group.get(feat).copied().flatten() {
                used_groups[group_idx] = true;
            }
        }

        // A feature is allowed if:
        // 1. It's unconstrained (not in any group), OR
        // 2. It belongs to a group that was already used by ancestors, OR
        // 3. No group has been used yet
        let any_group_used = used_groups.iter().any(|&u| u);

        (0..num_features)
            .filter(|&feat| {
                match self.feature_to_group.get(feat).copied().flatten() {
                    None => true, // Unconstrained feature
                    Some(group_idx) => {
                        if !any_group_used {
                            true // No group used yet, any feature allowed
                        } else {
                            used_groups[group_idx] // Only if this group was used
                        }
                    }
                }
            })
            .collect()
    }

    /// Get the groups
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
}

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
    /// Monotonic constraints per feature (empty = no constraints)
    monotonic_constraints: Vec<MonotonicConstraint>,
}

impl Default for SplitFinder {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            entropy_weight: 0.0, // No entropy regularization by default
            min_gain: 0.0,
            monotonic_constraints: Vec::new(),
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

    /// Set monotonic constraints for features
    ///
    /// The vector should have one entry per feature. Features beyond the
    /// vector length are treated as unconstrained.
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Get the monotonic constraint for a feature
    fn get_constraint(&self, feature_idx: usize) -> MonotonicConstraint {
        self.monotonic_constraints
            .get(feature_idx)
            .copied()
            .unwrap_or(MonotonicConstraint::None)
    }

    /// Check if we can use the SIMD fast path for a feature
    ///
    /// SIMD is used when:
    /// - No entropy regularization (entropy_weight == 0.0)
    /// - No monotonic constraint for this feature
    #[inline]
    fn can_use_simd(&self, feature_idx: usize) -> bool {
        self.entropy_weight == 0.0
            && self.get_constraint(feature_idx) == MonotonicConstraint::None
    }

    /// Check if a split satisfies monotonic constraints
    ///
    /// For a valid monotonic split:
    /// - Increasing: left_weight <= right_weight
    /// - Decreasing: left_weight >= right_weight
    fn satisfies_monotonic_constraint(
        &self,
        feature_idx: usize,
        left_gradient: f32,
        left_hessian: f32,
        right_gradient: f32,
        right_hessian: f32,
    ) -> bool {
        let constraint = self.get_constraint(feature_idx);

        match constraint {
            MonotonicConstraint::None => true,
            MonotonicConstraint::Increasing | MonotonicConstraint::Decreasing => {
                // Compute leaf weights: weight = -gradient / (hessian + lambda)
                let left_weight = -left_gradient / (left_hessian + self.lambda);
                let right_weight = -right_gradient / (right_hessian + self.lambda);

                match constraint {
                    MonotonicConstraint::Increasing => left_weight <= right_weight,
                    MonotonicConstraint::Decreasing => left_weight >= right_weight,
                    MonotonicConstraint::None => unreachable!(),
                }
            }
        }
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
        // Use SIMD fast path when no entropy regularization or monotonic constraints
        if self.can_use_simd(feature_idx) {
            return self.find_best_split_for_feature_simd(
                histogram,
                feature_idx,
                total_gradient,
                total_hessian,
                total_count,
            );
        }

        // Fall back to scalar implementation with full feature support
        self.find_best_split_for_feature_scalar(
            histogram,
            feature_idx,
            total_gradient,
            total_hessian,
            total_count,
        )
    }

    /// SIMD-optimized split finding for a single feature
    ///
    /// Uses the kernel's vectorized implementation for maximum performance.
    /// Only used when entropy_weight == 0 and no monotonic constraints.
    fn find_best_split_for_feature_simd(
        &self,
        histogram: &Histogram,
        feature_idx: usize,
        total_gradient: f32,
        total_hessian: f32,
        total_count: u32,
    ) -> Option<SplitInfo> {
        // Extract raw arrays from histogram for SIMD kernel
        let bins = histogram.bins();
        let mut hist_grads = [0.0f32; NUM_BINS];
        let mut hist_hess = [0.0f32; NUM_BINS];
        let mut hist_counts = [0u32; NUM_BINS];

        for i in 0..NUM_BINS {
            hist_grads[i] = bins[i].sum_gradients;
            hist_hess[i] = bins[i].sum_hessians;
            hist_counts[i] = bins[i].count;
        }

        // Call SIMD kernel
        let candidate = kernel_find_best_split(
            &hist_grads,
            &hist_hess,
            &hist_counts,
            total_gradient,
            total_hessian,
            total_count,
            self.lambda,
            self.min_samples_leaf as u32,
            self.min_hessian_leaf,
        )?;

        // Check min_gain threshold
        if candidate.gain < self.min_gain {
            return None;
        }

        // Convert kernel result to SplitInfo
        Some(SplitInfo {
            feature_idx,
            bin_threshold: candidate.bin_threshold,
            gain: candidate.gain,
            left_gradient: candidate.left_gradient,
            left_hessian: candidate.left_hessian,
            left_count: candidate.left_count,
            right_gradient: candidate.right_gradient,
            right_hessian: candidate.right_hessian,
            right_count: candidate.right_count,
        })
    }

    /// Scalar split finding with full feature support
    ///
    /// Supports entropy regularization and monotonic constraints.
    fn find_best_split_for_feature_scalar(
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

            // Check monotonic constraint
            if !self.satisfies_monotonic_constraint(
                feature_idx,
                left_gradient,
                left_hessian,
                right_gradient,
                right_hessian,
            ) {
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

    #[test]
    fn test_monotonic_increasing_constraint() {
        let mut histogram = Histogram::new();

        // Setup: bins 0-127 have negative gradient (positive weight),
        // bins 128-255 have positive gradient (negative weight)
        // weight = -gradient / hessian, so:
        // left: weight = -(-128) / 128 = +1.0
        // right: weight = -(+128) / 128 = -1.0
        // left_weight > right_weight => VIOLATES Increasing constraint
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0); // positive gradient = negative weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram.clone();

        // Without constraint: should find split
        let finder_no_constraint = SplitFinder::new().with_lambda(0.0);
        let split = finder_no_constraint.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_some());

        // With increasing constraint: should NOT find split
        // left_weight=+1.0 > right_weight=-1.0 => violates Increasing requirement
        let finder_increasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder_increasing.find_best_split(&histograms, 0.0, 256.0, 256);
        // The split violates monotonicity, should be rejected
        assert!(split.is_none());
    }

    #[test]
    fn test_monotonic_decreasing_constraint() {
        let mut histogram = Histogram::new();

        // Setup for decreasing: left should have higher weight than right
        // negative gradient = positive weight, positive gradient = negative weight
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, 1.0, 1.0); // positive gradient = negative weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // With decreasing constraint: should find split
        // left has positive weight, right has negative weight => left >= right (satisfied)
        let finder_decreasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Decreasing]);
        let split = finder_decreasing.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_some());

        // With increasing constraint: should NOT find split
        let finder_increasing = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder_increasing.find_best_split(&histograms, 0.0, 256.0, 256);
        assert!(split.is_none());
    }

    #[test]
    fn test_monotonic_constraint_allows_valid_split() {
        let mut histogram = Histogram::new();

        // Setup for valid increasing: left has lower weight than right
        for bin in 0..128 {
            histogram.accumulate(bin, -1.0, 1.0); // negative gradient = positive weight
        }
        for bin in 128..=255 {
            histogram.accumulate(bin, -2.0, 1.0); // more negative gradient = more positive weight
        }

        let mut histograms = crate::histogram::NodeHistograms::new(1);
        *histograms.get_mut(0) = histogram;

        // left_weight = -(-128) / 128 = 1.0
        // right_weight = -(-256) / 128 = 2.0
        // left_weight < right_weight => increasing is satisfied
        let finder = SplitFinder::new()
            .with_lambda(0.0)
            .with_monotonic_constraints(vec![MonotonicConstraint::Increasing]);
        let split = finder.find_best_split(&histograms, -384.0, 256.0, 256);
        assert!(split.is_some());
    }

    #[test]
    fn test_interaction_constraints_empty() {
        let constraints = InteractionConstraints::new();
        assert!(constraints.is_empty());

        // All features should be allowed
        let allowed = constraints.allowed_features(&[], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);

        // Even with ancestors, all features allowed
        let allowed = constraints.allowed_features(&[0, 2], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_interaction_constraints_basic() {
        // Group 0: features 0, 1
        // Group 1: features 2, 3
        // Feature 4: unconstrained
        let constraints = InteractionConstraints::from_groups(
            vec![vec![0, 1], vec![2, 3]],
            5,
        );

        // No ancestors: all features allowed
        let allowed = constraints.allowed_features(&[], 5);
        assert_eq!(allowed, vec![0, 1, 2, 3, 4]);

        // After using feature 0 (group 0): only group 0 and unconstrained allowed
        let allowed = constraints.allowed_features(&[0], 5);
        assert_eq!(allowed, vec![0, 1, 4]); // 0, 1 (group 0) and 4 (unconstrained)

        // After using feature 2 (group 1): only group 1 and unconstrained allowed
        let allowed = constraints.allowed_features(&[2], 5);
        assert_eq!(allowed, vec![2, 3, 4]); // 2, 3 (group 1) and 4 (unconstrained)
    }

    #[test]
    fn test_interaction_constraints_unconstrained_always_allowed() {
        // Only constrain some features
        let constraints = InteractionConstraints::from_groups(
            vec![vec![0, 1]],
            5, // Features 2, 3, 4 are unconstrained
        );

        // After using constrained feature, unconstrained still allowed
        let allowed = constraints.allowed_features(&[0], 5);
        assert!(allowed.contains(&2)); // unconstrained
        assert!(allowed.contains(&3)); // unconstrained
        assert!(allowed.contains(&4)); // unconstrained
        assert!(allowed.contains(&0)); // same group
        assert!(allowed.contains(&1)); // same group
        assert_eq!(allowed.len(), 5); // All allowed since 2,3,4 are unconstrained
    }
}
