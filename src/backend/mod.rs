//! Backend abstraction for histogram building.
//!
//! This module provides a vendor-agnostic abstraction layer for histogram
//! building operations, enabling different hardware backends:
//!
//! - **Scalar**: CPU implementation (AVX2/NEON loads)
//! - **WGPU**: GPU via Vulkan/Metal/DX12 (all GPU vendors, portable)
//! - **CUDA**: Native NVIDIA GPU (extreme performance, 10-100μs dispatch)
//! - **AVX-512**: Tensor-tile with vpconflictd (future)
//! - **SVE2**: ARM tensor-tile with HISTCNT (future)
//! - **ROCm**: AMD GPU direct (future)
//! - **Metal**: Apple GPU direct (future)

mod config;
pub mod scalar;
mod traits;

#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use config::{BackendConfig, BackendType, GpuMode};
pub use scalar::ScalarBackend;
pub use traits::{BinStorage, HistogramBackend, SparseColumn, SplitCandidate, SplitConfig};

#[cfg(feature = "gpu")]
pub use wgpu::WgpuBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

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
        // Priority order: CUDA > WGPU > AVX-512 > SVE2 > Scalar

        // Try CUDA first (NVIDIA-only but fastest: 10-100μs dispatch)
        #[cfg(feature = "cuda")]
        {
            if let Some(backend) = cuda::CudaBackend::new() {
                return Box::new(backend);
            }
        }

        // Try WGPU (covers all GPUs via Vulkan/Metal/DX12)
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = wgpu::WgpuBackend::new() {
                backend.set_use_subgroups(self.config.use_gpu_subgroups);
                return Box::new(backend);
            }
        }

        // TODO: Check for AVX-512 availability
        // TODO: Check for SVE2 availability

        Box::new(ScalarBackend::new())
    }

    fn try_wgpu_or_fallback(&self) -> Box<dyn HistogramBackend> {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = wgpu::WgpuBackend::new() {
                backend.set_use_subgroups(self.config.use_gpu_subgroups);
                return Box::new(backend);
            }
        }

        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("WGPU backend requested but not available (no GPU or 'gpu' feature disabled)")
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
        #[cfg(feature = "cuda")]
        {
            if let Some(backend) = cuda::CudaBackend::new() {
                return Box::new(backend);
            }
        }

        if self.config.fallback_to_scalar {
            Box::new(ScalarBackend::new())
        } else {
            panic!("CUDA backend requested but not available (no NVIDIA GPU or 'cuda' feature disabled)")
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
        // Large dataset - uses GPU if available, otherwise scalar
        let backend = selector.select(100_000);
        // Accept CUDA, WGPU, or Scalar depending on GPU availability
        assert!(
            backend.name() == "CUDA" || backend.name() == "WGPU" || backend.name().starts_with("Scalar"),
            "Expected CUDA, WGPU, or Scalar, got: {}",
            backend.name()
        );
    }

    #[test]
    fn test_backend_config_scalar() {
        let config = BackendConfig::scalar();
        let selector = BackendSelector::with_config(config);
        let backend = selector.select(1_000_000);
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_config_prefer_gpu() {
        let config = BackendConfig::prefer_gpu();
        let selector = BackendSelector::with_config(config);
        // Uses GPU if available, otherwise falls back to scalar
        let backend = selector.select(100_000);
        assert!(
            backend.name() == "CUDA" || backend.name() == "WGPU" || backend.name().starts_with("Scalar"),
            "Expected CUDA, WGPU, or Scalar, got: {}",
            backend.name()
        );
    }
}
