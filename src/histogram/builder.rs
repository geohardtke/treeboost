//! Parallel histogram construction

use crate::dataset::BinnedDataset;
use crate::histogram::{Histogram, NodeHistograms};
use rayon::prelude::*;

/// Histogram builder with feature-parallel construction
pub struct HistogramBuilder {
    /// Number of threads for parallel construction
    num_threads: usize,
}

impl Default for HistogramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramBuilder {
    /// Create a new histogram builder
    pub fn new() -> Self {
        Self {
            num_threads: rayon::current_num_threads(),
        }
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Build histograms for all features at a node
    ///
    /// # Arguments
    /// * `dataset` - The binned dataset
    /// * `row_indices` - Indices of rows belonging to this node
    /// * `gradients` - Gradient for each row in the full dataset
    /// * `hessians` - Hessian for each row in the full dataset
    pub fn build(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // Build histograms in parallel across features
        let histograms: Vec<Histogram> = (0..num_features)
            .into_par_iter()
            .map(|feature_idx| {
                self.build_single_feature(dataset, feature_idx, row_indices, gradients, hessians)
            })
            .collect();

        NodeHistograms::from_vec(histograms)
    }

    /// Build histogram for a single feature
    fn build_single_feature(
        &self,
        dataset: &BinnedDataset,
        feature_idx: usize,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Histogram {
        let mut histogram = Histogram::new();
        let feature_column = dataset.feature_column(feature_idx);

        for &row_idx in row_indices {
            let bin = feature_column[row_idx];
            let grad = gradients[row_idx];
            let hess = hessians[row_idx];
            histogram.accumulate(bin, grad, hess);
        }

        histogram
    }

    /// Build histograms using data parallelism (for large nodes)
    ///
    /// Chunks the rows across threads, builds partial histograms, then reduces.
    pub fn build_data_parallel(
        &self,
        dataset: &BinnedDataset,
        row_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
    ) -> NodeHistograms {
        let num_features = dataset.num_features();

        // If small enough, use simple feature-parallel
        if row_indices.len() < 10000 {
            return self.build(dataset, row_indices, gradients, hessians);
        }

        // Chunk rows and build partial histograms
        let chunk_size = (row_indices.len() / self.num_threads).max(1000);

        let partial_histograms: Vec<NodeHistograms> = row_indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut node_hists = NodeHistograms::new(num_features);

                for feature_idx in 0..num_features {
                    let feature_column = dataset.feature_column(feature_idx);
                    let hist = node_hists.get_mut(feature_idx);

                    for &row_idx in chunk {
                        let bin = feature_column[row_idx];
                        hist.accumulate(bin, gradients[row_idx], hessians[row_idx]);
                    }
                }

                node_hists
            })
            .collect();

        // Reduce partial histograms
        let mut result = NodeHistograms::new(num_features);
        for partial in partial_histograms {
            result.merge(&partial);
        }

        result
    }

    /// Build sibling histogram using Histogram Subtraction Trick
    ///
    /// Instead of building the larger sibling directly,
    /// compute it as: sibling = parent - smaller_child
    ///
    /// This halves the computation for splits.
    pub fn build_sibling(
        parent: &NodeHistograms,
        smaller_child: &NodeHistograms,
    ) -> NodeHistograms {
        NodeHistograms::from_subtraction(parent, smaller_child)
    }
}

// Extend NodeHistograms with from_vec constructor
impl NodeHistograms {
    /// Create from a vector of histograms
    pub fn from_vec(histograms: Vec<Histogram>) -> Self {
        Self { histograms }
    }

    /// Get internal histograms vector
    pub fn into_vec(self) -> Vec<Histogram> {
        self.histograms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};

    fn create_test_dataset() -> BinnedDataset {
        let num_rows = 100;
        let num_features = 3;

        // Generate deterministic test data
        let mut features = Vec::with_capacity(num_rows * num_features);
        for f in 0..num_features {
            for r in 0..num_rows {
                features.push(((r + f * 7) % 256) as u8);
            }
        }

        let targets: Vec<f32> = (0..num_rows).map(|i| i as f32).collect();
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
    fn test_build_histograms() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();

        let gradients: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = vec![1.0; num_rows];
        let row_indices: Vec<usize> = (0..num_rows).collect();

        let builder = HistogramBuilder::new();
        let hists = builder.build(&dataset, &row_indices, &gradients, &hessians);

        assert_eq!(hists.num_features(), 3);

        // Check total count
        let total_count: u32 = hists.get(0).bins().iter().map(|b| b.count).sum();
        assert_eq!(total_count, num_rows as u32);
    }

    #[test]
    fn test_subtraction_trick() {
        let dataset = create_test_dataset();
        let num_rows = dataset.num_rows();

        let gradients: Vec<f32> = (0..num_rows).map(|i| i as f32 * 0.1).collect();
        let hessians: Vec<f32> = vec![1.0; num_rows];

        let all_rows: Vec<usize> = (0..num_rows).collect();
        let left_rows: Vec<usize> = (0..num_rows / 2).collect();
        let right_rows: Vec<usize> = (num_rows / 2..num_rows).collect();

        let builder = HistogramBuilder::new();

        // Build parent and left child
        let parent = builder.build(&dataset, &all_rows, &gradients, &hessians);
        let left = builder.build(&dataset, &left_rows, &gradients, &hessians);

        // Build right using subtraction
        let right_subtracted = HistogramBuilder::build_sibling(&parent, &left);

        // Build right directly for comparison
        let right_direct = builder.build(&dataset, &right_rows, &gradients, &hessians);

        // Compare - should be identical (within floating point tolerance)
        for f in 0..dataset.num_features() {
            for bin in 0..=255u8 {
                let sub = right_subtracted.get(f).get(bin);
                let direct = right_direct.get(f).get(bin);

                assert!(
                    (sub.sum_gradients - direct.sum_gradients).abs() < 1e-5,
                    "Gradient mismatch at feature {} bin {}",
                    f,
                    bin
                );
                assert_eq!(sub.count, direct.count);
            }
        }
    }
}
