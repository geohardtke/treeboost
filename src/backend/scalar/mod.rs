//! Scalar backend for histogram building.
//!
//! Uses column-major data layout with:
//! - AVX2 SIMD loads on x86-64
//! - NEON SIMD loads on ARM
//! - Scalar scatter for bin updates (inherently sequential)
//!
//! This is the fallback backend for all platforms and the best choice
//! for small datasets (< 10K rows) due to lower overhead.

pub mod kernel;

use crate::backend::traits::{BinStorage, HistogramBackend, SplitCandidate, SplitConfig};
use crate::histogram::Histogram;

/// Scalar histogram building backend.
///
/// Uses cache-blocked processing with interleaved gradient/hessian layout
/// for optimal L1 cache utilization. Features:
///
/// - 2048-row blocks fit gradient cache in L1 (16KB)
/// - 8x unrolled inner loops for instruction-level parallelism
/// - Histogram Subtraction Trick for sibling nodes
/// - Sparse feature optimization (>90% sparsity)
/// - Parallel block processing via Rayon
#[derive(Debug, Clone, Default)]
pub struct ScalarBackend {
    /// Number of threads for parallel processing.
    /// 0 means use Rayon's default (typically number of CPU cores).
    num_threads: usize,
}

impl ScalarBackend {
    /// Create a new scalar backend with default thread count.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scalar backend with a specific thread count.
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
}

impl HistogramBackend for ScalarBackend {
    fn name(&self) -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if kernel::has_avx2() {
                "Scalar (AVX2)"
            } else {
                "Scalar"
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            "Scalar (NEON)"
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Scalar"
        }
    }

    fn is_tensor_tile(&self) -> bool {
        false
    }

    fn build_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram> {
        let builder = crate::histogram::HistogramBuilder::new();
        let num_features = bins.num_features();

        // Convert to the format expected by the builder
        // The builder expects separate gradient and hessian slices
        let gradients: Vec<f32> = grad_hess.iter().map(|(g, _)| *g).collect();
        let hessians: Vec<f32> = grad_hess.iter().map(|(_, h)| *h).collect();

        // Build histograms for all features
        // The builder handles parallelism internally
        let mut histograms = Vec::with_capacity(num_features);

        for feature in 0..num_features {
            if let Some(column) = bins.feature_column(feature) {
                let hist = builder.build_single_feature(
                    column,
                    row_indices,
                    &gradients,
                    &hessians,
                    bins.sparse_column(feature),
                );
                histograms.push(hist);
            } else {
                // Fallback for non-column-major storage
                histograms.push(Histogram::new());
            }
        }

        histograms
    }

    fn build_histograms_sibling(
        &self,
        parent: &[Histogram],
        smaller_child: &[Histogram],
    ) -> Vec<Histogram> {
        parent
            .iter()
            .zip(smaller_child.iter())
            .map(|(p, s)| Histogram::from_subtraction(p, s))
            .collect()
    }

    fn find_best_split(
        &self,
        histograms: &[Histogram],
        config: &SplitConfig,
    ) -> Option<SplitCandidate> {
        let mut best: Option<SplitCandidate> = None;

        for (feature, hist) in histograms.iter().enumerate() {
            // Get total statistics for this feature
            let (total_grad, total_hess, total_count) = hist.totals();

            if total_count < 2 * config.min_samples_leaf {
                continue;
            }

            // Use kernel's split finding
            if let Some(candidate) = kernel::find_best_split(
                &hist.sum_gradients(),
                &hist.sum_hessians(),
                &hist.counts(),
                total_grad,
                total_hess,
                total_count,
                config.lambda,
                config.min_samples_leaf,
                config.min_hessian_leaf,
            ) {
                let split = SplitCandidate {
                    feature,
                    threshold: candidate.bin_threshold,
                    gain: candidate.gain,
                    left_gradient: candidate.left_gradient,
                    left_hessian: candidate.left_hessian,
                    left_count: candidate.left_count,
                    right_gradient: candidate.right_gradient,
                    right_hessian: candidate.right_hessian,
                    right_count: candidate.right_count,
                };

                if split.gain > config.min_gain {
                    match &best {
                        None => best = Some(split),
                        Some(b) if split.gain > b.gain => best = Some(split),
                        _ => {}
                    }
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBinStorage {
        bins: Vec<Vec<u8>>,
        num_rows: usize,
    }

    impl BinStorage for MockBinStorage {
        fn get_bin(&self, row: usize, feature: usize) -> u8 {
            self.bins[feature][row]
        }

        fn num_rows(&self) -> usize {
            self.num_rows
        }

        fn num_features(&self) -> usize {
            self.bins.len()
        }

        fn feature_column(&self, feature: usize) -> Option<&[u8]> {
            Some(&self.bins[feature])
        }

        fn sparse_column(&self, _feature: usize) -> Option<&crate::backend::traits::SparseColumn> {
            None
        }
    }

    #[test]
    fn test_scalar_backend_name() {
        let backend = ScalarBackend::new();
        let name = backend.name();
        assert!(name.starts_with("Scalar"));
    }

    #[test]
    fn test_scalar_backend_is_not_tensor_tile() {
        let backend = ScalarBackend::new();
        assert!(!backend.is_tensor_tile());
    }
}
