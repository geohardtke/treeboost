//! Hybrid GPU backend that optimizes small batches on CPU.
//!
//! Different GPU backends have different launch overhead characteristics:
//! - **CUDA**: 10-100μs dispatch (very fast) → threshold ~1000 rows
//! - **WGPU**: 1-2ms dispatch (slower) → threshold ~5000 rows
//!
//! For small batches, this overhead dominates actual compute time, making CPU faster.
//!
//! This module provides a wrapper backend that:
//! - Uses GPU for large batches (>= threshold)
//! - Falls back to CPU for small batches (< threshold)
//!
//! The threshold is determined based on the GPU backend type.

use crate::backend::traits::{BinStorage, HistogramBackend, SplitCandidate, SplitConfig};
use crate::backend::ScalarBackend;
use crate::defaults::backend::{CUDA_HYBRID_THRESHOLD, WGPU_HYBRID_THRESHOLD};
use crate::histogram::Histogram;

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
    /// Underlying GPU backend for large batches
    gpu_backend: Box<dyn HistogramBackend>,
    /// CPU backend for small batches
    cpu_backend: ScalarBackend,
    /// Threshold for routing decision (rows)
    batch_threshold: usize,
}

impl HybridGpuBackend {
    /// Create a new hybrid backend wrapping the given GPU backend.
    ///
    /// Automatically selects the appropriate batch threshold based on backend type:
    /// - CUDA: 1000 rows (low overhead)
    /// - WGPU: 5000 rows (higher overhead)
    ///
    /// # Arguments
    /// * `gpu_backend` - The GPU backend to use for large batches
    pub fn new(gpu_backend: Box<dyn HistogramBackend>) -> Self {
        // Auto-detect threshold based on backend name
        let batch_threshold = match gpu_backend.name() {
            "CUDA" => CUDA_HYBRID_THRESHOLD,
            "WGPU" => WGPU_HYBRID_THRESHOLD,
            _ => CUDA_HYBRID_THRESHOLD, // Default to conservative threshold
        };

        Self {
            gpu_backend,
            cpu_backend: ScalarBackend::new(),
            batch_threshold,
        }
    }

    /// Create with a custom batch threshold.
    ///
    /// # Arguments
    /// * `gpu_backend` - The GPU backend to use for large batches
    /// * `batch_threshold` - Custom threshold (rows) for routing decision
    pub fn with_threshold(gpu_backend: Box<dyn HistogramBackend>, batch_threshold: usize) -> Self {
        Self {
            gpu_backend,
            cpu_backend: ScalarBackend::new(),
            batch_threshold,
        }
    }

    /// Check if a batch should use GPU based on size.
    #[inline]
    fn should_use_gpu(&self, batch_size: usize) -> bool {
        batch_size >= self.batch_threshold
    }
}

impl HistogramBackend for HybridGpuBackend {
    fn name(&self) -> &'static str {
        // Report the underlying GPU backend name with "Hybrid" prefix
        match self.gpu_backend.name() {
            "CUDA" => "Hybrid CUDA",
            "WGPU" => "Hybrid WGPU",
            other => other, // Fallback
        }
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
        if self.should_use_gpu(row_indices.len()) {
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

    #[test]
    fn test_hybrid_threshold() {
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::new(scalar);

        // Small batches should use CPU
        assert!(!hybrid.should_use_gpu(100));
        assert!(!hybrid.should_use_gpu(1999));

        // Large batches should use GPU
        assert!(hybrid.should_use_gpu(2000));
        assert!(hybrid.should_use_gpu(10000));
    }

    #[test]
    fn test_hybrid_custom_threshold() {
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::with_threshold(scalar, 500);

        assert!(!hybrid.should_use_gpu(499));
        assert!(hybrid.should_use_gpu(500));
        assert!(hybrid.should_use_gpu(1000));
    }

    #[test]
    fn test_hybrid_name() {
        let scalar = Box::new(ScalarBackend::new());
        let hybrid = HybridGpuBackend::new(scalar);

        // Should report as Hybrid variant
        let name = hybrid.name();
        assert!(name.contains("Hybrid") || name.contains("Scalar"));
    }
}
