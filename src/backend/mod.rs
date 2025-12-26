//! Backend abstraction for histogram building.
//!
//! This module provides a vendor-agnostic abstraction layer for histogram
//! building operations, enabling different hardware backends:
//!
//! - **Scalar**: Current CPU implementation (AVX2/NEON loads)
//! - **WGPU**: GPU via Vulkan/Metal/DX12 (future)
//! - **AVX-512**: Tensor-tile with vpconflictd (future)
//! - **SVE2**: ARM tensor-tile with HISTCNT (future)
//! - **Native**: CUDA, ROCm, Metal direct (future extreme optimization)

mod config;
pub mod scalar;
mod traits;

pub use config::{BackendConfig, BackendType};
pub use scalar::ScalarBackend;
pub use traits::{BinStorage, HistogramBackend, SparseColumn, SplitCandidate, SplitConfig};

/// Backend selector for choosing the best available backend.
///
/// Uses runtime detection to select the optimal backend based on:
/// - Available hardware (GPU, AVX-512, SVE2)
/// - Dataset size (smaller datasets prefer scalar for lower overhead)
/// - User configuration
#[derive(Debug)]
pub struct BackendSelector {
    config: BackendConfig,
}

impl BackendSelector {
    /// Create a new backend selector with default configuration.
    pub fn new() -> Self {
        Self {
            config: BackendConfig::default(),
        }
    }

    /// Create a backend selector with custom configuration.
    pub fn with_config(config: BackendConfig) -> Self {
        Self { config }
    }

    /// Select the best backend for the given dataset size.
    ///
    /// # Arguments
    /// * `num_rows` - Number of rows in the dataset
    ///
    /// # Returns
    /// A boxed trait object implementing HistogramBackend
    pub fn select(&self, num_rows: usize) -> Box<dyn HistogramBackend> {
        // For small datasets, always use scalar (lower overhead)
        if num_rows < self.config.tensor_tile_min_rows {
            return Box::new(ScalarBackend::new());
        }

        match self.config.preferred {
            BackendType::Auto => self.detect_best(),
            BackendType::Scalar => Box::new(ScalarBackend::new()),
            BackendType::Wgpu => self.try_wgpu_or_fallback(),
            BackendType::Avx512 => self.try_avx512_or_fallback(),
            BackendType::Sve2 => self.try_sve2_or_fallback(),
            BackendType::Cuda => self.try_cuda_or_fallback(),
            BackendType::Rocm => self.try_rocm_or_fallback(),
            BackendType::Metal => self.try_metal_or_fallback(),
        }
    }

    /// Detect the best available backend.
    fn detect_best(&self) -> Box<dyn HistogramBackend> {
        // Priority order: GPU > AVX-512 > SVE2 > Scalar
        // Currently only scalar is implemented

        // TODO: Check for WGPU availability
        // TODO: Check for AVX-512 availability
        // TODO: Check for SVE2 availability

        Box::new(ScalarBackend::new())
    }

    fn try_wgpu_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement WGPU backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("WGPU backend not yet implemented")
        }
    }

    fn try_avx512_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement AVX-512 tensor-tile backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("AVX-512 tensor-tile backend not yet implemented")
        }
    }

    fn try_sve2_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement SVE2 tensor-tile backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("SVE2 tensor-tile backend not yet implemented")
        }
    }

    fn try_cuda_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement native CUDA backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("CUDA backend not yet implemented")
        }
    }

    fn try_rocm_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement native ROCm backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("ROCm backend not yet implemented")
        }
    }

    fn try_metal_or_fallback(&self) -> Box<dyn HistogramBackend> {
        // TODO: Implement native Metal backend
        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("Metal backend not yet implemented")
        }
    }

    /// Get the name of the currently selected backend.
    pub fn backend_name(&self, num_rows: usize) -> &'static str {
        self.select(num_rows).name()
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selector_default() {
        let selector = BackendSelector::new();
        let backend = selector.select(1000);
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_selector_small_dataset() {
        let selector = BackendSelector::new();
        // Small dataset should always use scalar
        let backend = selector.select(100);
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_selector_large_dataset() {
        let selector = BackendSelector::new();
        // Large dataset - currently falls back to scalar since no GPU impl
        let backend = selector.select(100_000);
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_config_scalar() {
        let config = BackendConfig::scalar();
        let selector = BackendSelector::with_config(config);
        let backend = selector.select(1_000_000);
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_config_prefer_gpu_fallback() {
        let config = BackendConfig::prefer_gpu();
        let selector = BackendSelector::with_config(config);
        // Should fall back to scalar since WGPU not implemented
        let backend = selector.select(100_000);
        assert!(backend.name().starts_with("Scalar"));
    }
}
