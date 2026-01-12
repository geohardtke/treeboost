//! Smart hybrid GPU backend with workload-aware routing.
//!
//! Instead of using a fixed threshold, this backend makes intelligent routing decisions
//! based on the full workload context:
//! - **Batch size**: Rows in current histogram call
//! - **Feature count**: Number of features (affects parallelism)
//! - **GPU characteristics**: Launch overhead (CUDA: 10-100μs, WGPU: 1-2ms)
//!
//! ## Smart Routing Logic
//!
//! The router considers total GPU work: `batch_size × num_features × num_bins`
//! - **Large workloads**: Route to GPU (benefit from massive parallelism)
//! - **Tiny workloads**: Route to CPU (avoid GPU dispatch overhead)
//!
//! This is much more effective than per-call thresholds because GBDT training
//! builds hundreds of histograms in parallel (many nodes × many features).

use crate::backend::traits::{BinStorage, HistogramBackend, SplitCandidate, SplitConfig};
use crate::backend::ScalarBackend;
use crate::histogram::Histogram;

// Smart routing thresholds
// These values determined by empirical benchmarking of GPU dispatch overhead vs CPU compute time

/// CUDA minimum batch size threshold.
/// CUDA has extremely low dispatch overhead (10-100μs), so only skip truly tiny batches.
const CUDA_MIN_BATCH_SIZE: usize = 10;

/// WGPU minimum batch size threshold.
/// WGPU has 1-2ms dispatch overhead, so requires larger batches to be worthwhile.
const WGPU_MIN_BATCH_SIZE: usize = 50;

/// Number of bins per feature used in histogram building.
/// This is a fixed value in TreeBoost's binning strategy.
const BINS_PER_FEATURE: usize = 256;

/// WGPU workload threshold in histogram elements.
/// Total work = batch_size × num_features × BINS_PER_FEATURE
/// Below this threshold, CPU is faster due to avoiding GPU dispatch overhead.
const WGPU_WORK_THRESHOLD: usize = 200_000;

/// Hybrid GPU backend that routes small batches to CPU for optimal performance.
///
/// # Architecture
///
/// This backend wraps a GPU backend (CUDA or WGPU) and makes per-call routing decisions:
/// - Small batches (< 1000 rows): Route to CPU (avoid GPU launch overhead)
/// - Large batches (>= 1000 rows): Route to GPU (benefit from parallelism)
///
/// The threshold was determined by benchmarking GPU overhead vs CPU compute time.
pub struct HybridGpuBackend {
    /// Underlying GPU backend for large workloads
    gpu_backend: Box<dyn HistogramBackend>,
    /// CPU backend for small workloads
    cpu_backend: ScalarBackend,
}

impl HybridGpuBackend {
    /// Create a new hybrid backend wrapping the given GPU backend.
    ///
    /// Uses smart workload-aware routing that considers:
    /// - Batch size (rows in current histogram call)
    /// - Number of features (affects parallelism)
    /// - GPU type (CUDA vs WGPU overhead characteristics)
    ///
    /// # Arguments
    /// * `gpu_backend` - The GPU backend to use for large workloads
    pub fn new(gpu_backend: Box<dyn HistogramBackend>) -> Self {
        Self {
            gpu_backend,
            cpu_backend: ScalarBackend::new(),
        }
    }

    /// Smart routing decision based on full workload context.
    ///
    /// Considers:
    /// - Batch size (rows in current call)
    /// - Number of features (parallelism)
    /// - GPU type (CUDA has lower overhead than WGPU)
    ///
    /// Total GPU work = batch_size × num_features × BINS_PER_FEATURE
    #[inline]
    fn should_use_gpu(&self, batch_size: usize, num_features: usize) -> bool {
        // CUDA has extremely low dispatch overhead (10-100μs), so ALWAYS use GPU
        // Only WGPU needs hybrid routing due to its 1-2ms dispatch overhead
        if matches!(
            self.gpu_backend.backend_type(),
            crate::backend::BackendType::Cuda
        ) {
            // For CUDA: only skip truly tiny batches
            return batch_size >= CUDA_MIN_BATCH_SIZE;
        }

        // For WGPU: use workload-based routing
        // For tiny batches, always use CPU (avoid dispatch overhead)
        if batch_size < WGPU_MIN_BATCH_SIZE {
            return false;
        }

        // Calculate total histogram elements to process
        // Each feature has BINS_PER_FEATURE bins
        let total_work = batch_size * num_features * BINS_PER_FEATURE;

        // WGPU needs larger workloads to amortize dispatch overhead
        total_work >= WGPU_WORK_THRESHOLD
    }
}

impl HistogramBackend for HybridGpuBackend {
    fn name(&self) -> &'static str {
        // Report the underlying GPU backend name directly
        // Smart routing (CPU vs GPU) is an implementation detail, not part of the name
        self.gpu_backend.name()
    }

    fn backend_type(&self) -> crate::backend::BackendType {
        // Delegate to underlying GPU backend for type-safe identification
        self.gpu_backend.backend_type()
    }

    fn is_tensor_tile(&self) -> bool {
        // Hybrid backend behaves as tensor-tile (GPU characteristic)
        true
    }

    fn build_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
    ) -> Vec<Histogram> {
        let batch_size = row_indices.len();
        let num_features = bins.num_features();
        let use_gpu = self.should_use_gpu(batch_size, num_features);

        // Log routing decision (use TRACE level for high-frequency calls)
        tracing::trace!(
            batch_size,
            num_features,
            use_gpu,
            backend = if use_gpu {
                self.gpu_backend.name()
            } else {
                "CPU"
            },
            "Smart router histogram build"
        );

        if use_gpu {
            self.gpu_backend
                .build_histograms(bins, grad_hess, row_indices)
        } else {
            self.cpu_backend
                .build_histograms(bins, grad_hess, row_indices)
        }
    }

    fn build_histograms_sibling(
        &self,
        parent: &[Histogram],
        smaller_child: &[Histogram],
    ) -> Vec<Histogram> {
        // Histogram subtraction is always fast on CPU (no dispatch overhead)
        self.cpu_backend
            .build_histograms_sibling(parent, smaller_child)
    }

    fn find_best_split(
        &self,
        histograms: &[Histogram],
        config: &SplitConfig,
    ) -> Option<SplitCandidate> {
        // Split finding is always done on CPU (fast scan of 256 bins per feature)
        self.cpu_backend.find_best_split(histograms, config)
    }

    fn build_histograms_batched(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        batches: &[&[usize]],
    ) -> Vec<Vec<Histogram>> {
        // For batched operations, check each batch individually
        batches
            .iter()
            .map(|batch| self.build_histograms(bins, grad_hess, batch))
            .collect()
    }

    fn build_era_histograms(
        &self,
        bins: &dyn BinStorage,
        grad_hess: &[(f32, f32)],
        row_indices: &[usize],
        era_indices: &[u16],
        num_eras: usize,
    ) -> Vec<Vec<Histogram>> {
        // Era-based histogram building always uses underlying backend
        // (complex operation, worth GPU overhead)
        self.gpu_backend
            .build_era_histograms(bins, grad_hess, row_indices, era_indices, num_eras)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendType;

    #[test]
    fn test_hybrid_routing_wgpu() {
        // ScalarBackend reports as "Scalar", not "CUDA", so it uses WGPU-style routing
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::new(scalar);

        // WGPU routing: Check minimum batch size threshold
        assert!(!hybrid.should_use_gpu(10, 100)); // < WGPU_MIN_BATCH_SIZE (50) → CPU
        assert!(!hybrid.should_use_gpu(49, 100)); // < WGPU_MIN_BATCH_SIZE (50) → CPU

        // WGPU routing: Check workload threshold
        // With 100 features, work = batch_size * 100 * BINS_PER_FEATURE (256)
        // WGPU_WORK_THRESHOLD = 200,000
        // Need batch_size * 100 * 256 >= 200,000 → batch_size >= 8
        assert!(hybrid.should_use_gpu(50, 100)); // 50 * 100 * 256 = 1,280,000 > 200,000 → GPU
        assert!(!hybrid.should_use_gpu(50, 1)); // 50 * 1 * 256 = 12,800 < 200,000 → CPU
    }

    #[test]
    fn test_hybrid_routing_cuda() {
        // Create a mock backend that reports as "CUDA" to test CUDA routing logic
        // In reality, we'd use a real CudaBackend, but this tests the logic path
        use crate::backend::{BackendType, HistogramBackend};
        use crate::histogram::Histogram;

        struct MockCudaBackend;
        impl HistogramBackend for MockCudaBackend {
            fn name(&self) -> &'static str {
                "CUDA"
            }
            fn backend_type(&self) -> BackendType {
                BackendType::Cuda
            }
            fn is_tensor_tile(&self) -> bool {
                true
            }
            fn build_histograms(
                &self,
                _bins: &dyn crate::backend::BinStorage,
                _grad_hess: &[(f32, f32)],
                _row_indices: &[usize],
            ) -> Vec<Histogram> {
                vec![]
            }
            fn build_histograms_sibling(
                &self,
                _parent: &[Histogram],
                _smaller_child: &[Histogram],
            ) -> Vec<Histogram> {
                vec![]
            }
            fn find_best_split(
                &self,
                _histograms: &[Histogram],
                _config: &crate::backend::SplitConfig,
            ) -> Option<crate::backend::SplitCandidate> {
                None
            }
        }

        let mock_cuda = Box::new(MockCudaBackend);
        let hybrid = HybridGpuBackend::new(mock_cuda);

        // CUDA routing: Only skip truly tiny batches (< CUDA_MIN_BATCH_SIZE = 10)
        assert!(!hybrid.should_use_gpu(5, 100)); // < CUDA_MIN_BATCH_SIZE → CPU
        assert!(!hybrid.should_use_gpu(9, 100)); // < CUDA_MIN_BATCH_SIZE → CPU
        assert!(hybrid.should_use_gpu(10, 100)); // >= CUDA_MIN_BATCH_SIZE → GPU
        assert!(hybrid.should_use_gpu(100, 100)); // Large batch → GPU

        // CUDA doesn't check workload threshold, only batch size
        assert!(hybrid.should_use_gpu(10, 1)); // Even with 1 feature → GPU
    }

    #[test]
    fn test_hybrid_name_passthrough() {
        // Hybrid backend should report the underlying backend's name, not "Hybrid"
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::new(scalar);

        let name = hybrid.name();
        assert!(
            name.starts_with("Scalar"),
            "Expected Scalar name, got: {}",
            name
        );
    }

    #[test]
    fn test_hybrid_backend_type_passthrough() {
        // Hybrid backend should report the underlying backend's type
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::new(scalar);

        assert_eq!(hybrid.backend_type(), BackendType::Scalar);
    }
}
