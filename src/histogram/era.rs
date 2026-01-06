//! Era-stratified histograms for Directional Era Splitting (DES)
//!
//! Provides per-era histogram computation for learning invariant patterns
//! that generalize across time periods or environments.
//!
//! # Architecture
//!
//! - `EraHistograms`: Stores `[num_eras][num_features][num_bins]` histogram data
//! - `EraHistogramBuilder`: Builds era-stratified histograms efficiently
//!
//! # Usage
//!
//! Era splitting filters out spurious correlations that work in some eras
//! but not others, learning only patterns that are consistent across all eras.

use crate::dataset::BinnedDataset;
use crate::histogram::{Histogram, NodeHistograms};
use rayon::prelude::*;

/// Block size for cache-blocked histogram building
const BLOCK_SIZE: usize = 2048;

/// Era-stratified histograms for DES split finding
///
/// Stores histograms per (era, feature) pair, enabling directional
/// agreement checks across eras during split finding.
#[derive(Debug, Clone)]
pub struct EraHistograms {
    /// Histograms indexed as [era * num_features + feature]
    histograms: Vec<Histogram>,
    /// Number of eras
    num_eras: usize,
    /// Number of features
    num_features: usize,
}

impl EraHistograms {
    /// Create new empty era histograms
    pub fn new(num_eras: usize, num_features: usize) -> Self {
        let histograms = vec![Histogram::new(); num_eras * num_features];
        Self {
            histograms,
            num_eras,
            num_features,
        }
    }

    /// Create EraHistograms from a 2D vector of histograms `[era][feature]`
    ///
    /// Used to convert backend output to EraHistograms structure.
    pub fn from_vec(histograms_2d: Vec<Vec<Histogram>>) -> Self {
        let num_eras = histograms_2d.len();
        let num_features = if num_eras > 0 { histograms_2d[0].len() } else { 0 };

        // Flatten [era][feature] to [era * num_features + feature]
        let histograms: Vec<Histogram> = histograms_2d.into_iter().flatten().collect();

        Self {
            histograms,
            num_eras,
            num_features,
        }
    }

    /// Get histogram for a specific era and feature
    #[inline]
    pub fn get(&self, era: usize, feature_idx: usize) -> &Histogram {
        &self.histograms[era * self.num_features + feature_idx]
    }

    /// Get mutable histogram for a specific era and feature
    #[inline]
    pub fn get_mut(&mut self, era: usize, feature_idx: usize) -> &mut Histogram {
        &mut self.histograms[era * self.num_features + feature_idx]
    }

    /// Number of eras
    #[inline]
    pub fn num_eras(&self) -> usize {
        self.num_eras
    }

    /// Number of features
    #[inline]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Clear all histograms
    pub fn clear(&mut self) {
        for hist in &mut self.histograms {
            hist.clear();
        }
    }

    /// Compute global (aggregated across eras) histograms for a feature
    ///
    /// Used when standard (non-DES) split finding is needed as fallback
    pub fn aggregate_feature(&self, feature_idx: usize) -> Histogram {
        let mut result = Histogram::new();
        for era in 0..self.num_eras {
            result.merge(self.get(era, feature_idx));
        }
        result
    }

    /// Compute global (aggregated) histograms for all features
    pub fn aggregate_all(&self) -> NodeHistograms {
        let mut result = NodeHistograms::new(self.num_features);
        for feature_idx in 0..self.num_features {
            *result.get_mut(feature_idx) = self.aggregate_feature(feature_idx);
        }
        result
    }

    /// Subtract another EraHistograms from this one (for sibling computation)
    pub fn subtract(&mut self, other: &EraHistograms) {
        debug_assert_eq!(self.num_eras, other.num_eras);
        debug_assert_eq!(self.num_features, other.num_features);

        for (self_hist, other_hist) in self.histograms.iter_mut().zip(other.histograms.iter()) {
            self_hist.subtract(other_hist);
        }
    }

    /// Compute sibling by subtraction: parent - child
    pub fn from_subtraction(parent: &EraHistograms, child: &EraHistograms) -> Self {
        let mut result = parent.clone();
        result.subtract(child);
        result
    }

    /// Iterate over all (era, feature_idx, histogram) tuples
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &Histogram)> {
        (0..self.num_eras).flat_map(move |era| {
            (0..self.num_features).map(move |f| (era, f, self.get(era, f)))
        })
    }
}

/// Statistics for a single era at a split point
#[derive(Debug, Clone, Copy)]
pub struct EraSplitStats {
    /// Era index
    pub era: usize,
    /// Sum of gradients on left side
    pub grad_left: f32,
    /// Sum of hessians on left side
    pub hess_left: f32,
    /// Sum of gradients on right side
    pub grad_right: f32,
    /// Sum of hessians on right side
    pub hess_right: f32,
    /// Split direction: +1 if left has higher prediction, -1 otherwise
    pub direction: f32,
    /// Split gain for this era
    pub gain: f32,
}

impl EraSplitStats {
    /// Compute split statistics for a single era at a split point
    ///
    /// # Arguments
    /// * `era` - Era index
    /// * `grad_left` - Cumulative gradient sum up to split point
    /// * `hess_left` - Cumulative hessian sum up to split point
    /// * `grad_total` - Total gradient sum for this era
    /// * `hess_total` - Total hessian sum for this era
    /// * `lambda` - L2 regularization parameter
    pub fn compute(
        era: usize,
        grad_left: f32,
        hess_left: f32,
        grad_total: f32,
        hess_total: f32,
        lambda: f32,
    ) -> Self {
        let grad_right = grad_total - grad_left;
        let hess_right = hess_total - hess_left;

        // Compute leaf values: -gradient / (hessian + lambda)
        let eps = 1e-10f32;
        let left_val = -grad_left / (hess_left + lambda + eps);
        let right_val = -grad_right / (hess_right + lambda + eps);

        // Direction: +1 if left has higher prediction value
        let direction = if left_val > right_val { 1.0 } else { -1.0 };

        // Compute gain
        let gain = (grad_left * grad_left) / (hess_left + lambda)
            + (grad_right * grad_right) / (hess_right + lambda)
            - (grad_total * grad_total) / (hess_total + lambda);

        Self {
            era,
            grad_left,
            hess_left,
            grad_right,
            hess_right,
            direction,
            gain,
        }
    }
}

/// Check if all eras agree on split direction
///
/// Returns true if all eras have the same direction (all positive or all negative)
pub fn has_directional_agreement(era_stats: &[EraSplitStats]) -> bool {
    if era_stats.is_empty() {
        return false;
    }

    let first_direction = era_stats[0].direction;
    era_stats.iter().all(|s| s.direction == first_direction)
}

/// Compute average gain across eras
pub fn average_era_gain(era_stats: &[EraSplitStats]) -> f32 {
    if era_stats.is_empty() {
        return 0.0;
    }
    era_stats.iter().map(|s| s.gain).sum::<f32>() / era_stats.len() as f32
}

/// Builder for era-stratified histograms
///
/// Uses Rayon's work-stealing thread pool for parallelism.
/// Thread count is controlled globally via `rayon::ThreadPoolBuilder` or
/// the `RAYON_NUM_THREADS` environment variable.
#[derive(Debug, Clone, Copy)]
pub struct EraHistogramBuilder;

impl Default for EraHistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EraHistogramBuilder {
    /// Create a new era histogram builder
    pub fn new() -> Self {
        Self
    }

    /// Build era-stratified histograms for all features at a node
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset (must have era indices set)
    /// * `row_indices` - Indices of rows belonging to this node
    /// * `gradients` - Gradient for each row in the full dataset
    /// * `hessians` - Hessian for each row in the full dataset
    pub fn build(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> EraHistograms {
        debug_assert!(dataset.has_eras(), "Dataset must have era indices for DES");

        let num_eras = dataset.num_eras();
        let num_features = dataset.num_features();
        let num_rows = row_indices.len();

        // For small datasets, use simple single-threaded approach
        if num_rows < BLOCK_SIZE {
            return self.build_single_block(dataset, row_indices, gradients, hessians);
        }

        // Parallel blocked build
        self.build_blocked(dataset, row_indices, gradients, hessians, num_eras, num_features)
    }

    /// Single-threaded build for small datasets
    fn build_single_block(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> EraHistograms {
        let num_eras = dataset.num_eras();
        let num_features = dataset.num_features();
        let era_indices = dataset.era_indices().expect("Dataset must have era indices");

        let mut result = EraHistograms::new(num_eras, num_features);

        for &row in row_indices {
            let era = era_indices[row] as usize;
            let grad = gradients[row];
            let hess = hessians[row];

            for f in 0..num_features {
                let bin = dataset.get_bin(row, f);
                result.get_mut(era, f).accumulate(bin, grad, hess);
            }
        }

        result
    }

    /// Parallel blocked build for large datasets
    fn build_blocked(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
        num_eras: usize,
        num_features: usize,
    ) -> EraHistograms {
        let era_indices = dataset.era_indices().expect("Dataset must have era indices");

        // Divide rows into blocks and process in parallel
        let partial_results: Vec<EraHistograms> = row_indices
            .par_chunks(BLOCK_SIZE)
            .map(|block| {
                let mut partial = EraHistograms::new(num_eras, num_features);

                for &row in block {
                    let era = era_indices[row] as usize;
                    let grad = gradients[row];
                    let hess = hessians[row];

                    for f in 0..num_features {
                        let bin = dataset.get_bin(row, f);
                        partial.get_mut(era, f).accumulate(bin, grad, hess);
                    }
                }

                partial
            })
            .collect();

        // Merge partial results
        let mut result = EraHistograms::new(num_eras, num_features);
        for partial in partial_results {
            for era in 0..num_eras {
                for f in 0..num_features {
                    result.get_mut(era, f).merge(partial.get(era, f));
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_era_split_stats() {
        // Era 0: left has positive weight, right has negative
        let stats0 = EraSplitStats::compute(0, -10.0, 10.0, 0.0, 20.0, 1.0);
        assert_eq!(stats0.direction, 1.0); // left > right

        // Era 1: same direction
        let stats1 = EraSplitStats::compute(1, -8.0, 8.0, 0.0, 16.0, 1.0);
        assert_eq!(stats1.direction, 1.0);

        // These should agree
        assert!(has_directional_agreement(&[stats0, stats1]));
    }

    #[test]
    fn test_directional_disagreement() {
        // Era 0: left > right
        let stats0 = EraSplitStats::compute(0, -10.0, 10.0, 0.0, 20.0, 1.0);

        // Era 1: right > left (opposite direction)
        let stats1 = EraSplitStats::compute(1, 10.0, 10.0, 0.0, 20.0, 1.0);

        // These should NOT agree
        assert!(!has_directional_agreement(&[stats0, stats1]));
    }

    #[test]
    fn test_era_histograms_aggregate() {
        let mut era_hists = EraHistograms::new(2, 2);

        // Era 0, feature 0
        era_hists.get_mut(0, 0).accumulate(5, 1.0, 2.0);
        era_hists.get_mut(0, 0).accumulate(5, 0.5, 1.0);

        // Era 1, feature 0
        era_hists.get_mut(1, 0).accumulate(5, 2.0, 3.0);

        // Aggregate
        let agg = era_hists.aggregate_feature(0);
        assert_eq!(agg.get(5).sum_gradients, 3.5);
        assert_eq!(agg.get(5).sum_hessians, 6.0);
        assert_eq!(agg.get(5).count, 3);
    }
}
