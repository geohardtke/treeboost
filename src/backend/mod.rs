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

use crate::{Result, TreeBoostError};

mod config;
mod hybrid;
pub mod scalar;
mod traits;

#[cfg(feature = "gpu")]
pub mod wgpu;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use config::{BackendConfig, BackendPreset, BackendType, GpuMode};
pub use hybrid::HybridGpuBackend;
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
    pub fn select(&self, num_rows: usize) -> Result<Box<dyn HistogramBackend>> {
        match self.config.preferred {
            BackendType::Auto => {
                // For Auto mode, apply threshold check for small datasets
                if num_rows < self.config.tensor_tile_min_rows {
                    return Ok(Box::new(ScalarBackend::new()));
                }
                self.detect_best()
            }
            // For explicit backend choices, respect them regardless of dataset size
            BackendType::Scalar => Ok(Box::new(ScalarBackend::new())),
            BackendType::Wgpu => self.try_wgpu_or_fallback(),
            BackendType::Avx512 => self.try_avx512_or_fallback(),
            BackendType::Sve2 => self.try_sve2_or_fallback(),
            BackendType::Cuda => self.try_cuda_or_fallback(),
            BackendType::Rocm => self.try_rocm_or_fallback(),
            BackendType::Metal => self.try_metal_or_fallback(),
        }
    }

    /// Detect the best available backend.
    fn detect_best(&self) -> Result<Box<dyn HistogramBackend>> {
        // Priority order: CUDA > WGPU > AVX-512 > SVE2 > Scalar

        // Try CUDA first (NVIDIA-only but fastest: 10-100μs dispatch)
        #[cfg(feature = "cuda")]
        {
            if let Some(backend) = cuda::CudaBackend::new() {
                // Wrap CUDA in hybrid backend with user-configured threshold
                return Ok(Box::new(HybridGpuBackend::with_threshold(
                    Box::new(backend),
                    self.config.cuda_hybrid_threshold,
                )));
            }
        }

        // Try WGPU (covers all GPUs via Vulkan/Metal/DX12)
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = wgpu::WgpuBackend::new() {
                backend.set_use_subgroups(self.config.use_gpu_subgroups);
                // Wrap WGPU in hybrid backend with user-configured threshold
                return Ok(Box::new(HybridGpuBackend::with_threshold(
                    Box::new(backend),
                    self.config.wgpu_hybrid_threshold,
                )));
            }
        }

        // TODO: Check for AVX-512 availability
        // TODO: Check for SVE2 availability

        Ok(Box::new(ScalarBackend::new()))
    }

    fn try_wgpu_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = wgpu::WgpuBackend::new() {
                backend.set_use_subgroups(self.config.use_gpu_subgroups);
                // Wrap WGPU in hybrid backend with user-configured threshold
                return Ok(Box::new(HybridGpuBackend::with_threshold(
                    Box::new(backend),
                    self.config.wgpu_hybrid_threshold,
                )));
            }
        }

        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "WGPU backend unavailable: no GPU detected or 'gpu' feature not enabled. \
                 Enable 'gpu' feature or set BackendConfig::fallback_to_scalar = true"
                    .to_string(),
            ))
        }
    }

    fn try_avx512_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        // TODO: Implement AVX-512 tensor-tile backend
        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "AVX-512 tensor-tile backend not yet implemented. Use BackendType::Auto or enable fallback_to_scalar".to_string()
            ))
        }
    }

    fn try_sve2_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        // TODO: Implement SVE2 tensor-tile backend
        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "SVE2 tensor-tile backend not yet implemented. Use BackendType::Auto or enable fallback_to_scalar".to_string()
            ))
        }
    }

    fn try_cuda_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        #[cfg(feature = "cuda")]
        {
            if let Some(backend) = cuda::CudaBackend::new() {
                // Wrap CUDA in hybrid backend with user-configured threshold
                return Ok(Box::new(HybridGpuBackend::with_threshold(
                    Box::new(backend),
                    self.config.cuda_hybrid_threshold,
                )));
            }
        }

        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "CUDA backend unavailable: no NVIDIA GPU detected or 'cuda' feature not enabled. \
                 Enable 'cuda' feature or set BackendConfig::fallback_to_scalar = true"
                    .to_string(),
            ))
        }
    }

    fn try_rocm_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        // TODO: Implement native ROCm backend
        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "ROCm backend not yet implemented. Use BackendType::Auto or enable fallback_to_scalar".to_string()
            ))
        }
    }

    fn try_metal_or_fallback(&self) -> Result<Box<dyn HistogramBackend>> {
        // TODO: Implement native Metal backend
        if self.config.fallback_to_scalar {
            Ok(Box::new(ScalarBackend::new()))
        } else {
            Err(TreeBoostError::Backend(
                "Metal backend not yet implemented. Use BackendType::Auto or enable fallback_to_scalar".to_string()
            ))
        }
    }

    /// Get the name of the currently selected backend.
    pub fn backend_name(&self, num_rows: usize) -> Result<&'static str> {
        Ok(self.select(num_rows)?.name())
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
        let backend = selector.select(1000).expect("failed to select backend");
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_selector_small_dataset() {
        let selector = BackendSelector::new();
        // Small dataset should always use scalar
        let backend = selector.select(100).expect("failed to select backend");
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_selector_large_dataset() {
        let selector = BackendSelector::new();
        // Large dataset - uses GPU if available, otherwise scalar
        let backend = selector.select(100_000).expect("failed to select backend");
        // Accept Hybrid CUDA, Hybrid WGPU, or Scalar depending on GPU availability
        assert!(
            backend.name() == "Hybrid CUDA"
                || backend.name() == "Hybrid WGPU"
                || backend.name().starts_with("Scalar"),
            "Expected Hybrid CUDA, Hybrid WGPU, or Scalar, got: {}",
            backend.name()
        );
    }

    #[test]
    fn test_backend_config_scalar() {
        let config = BackendConfig::scalar();
        let selector = BackendSelector::with_config(config);
        let backend = selector
            .select(1_000_000)
            .expect("failed to select backend");
        assert!(backend.name().starts_with("Scalar"));
    }

    #[test]
    fn test_backend_config_prefer_gpu() {
        let config = BackendConfig::prefer_gpu();
        let selector = BackendSelector::with_config(config);
        // Uses GPU if available, otherwise falls back to scalar
        let backend = selector.select(100_000).expect("failed to select backend");
        assert!(
            backend.name() == "Hybrid CUDA"
                || backend.name() == "Hybrid WGPU"
                || backend.name().starts_with("Scalar"),
            "Expected Hybrid CUDA, Hybrid WGPU, or Scalar, got: {}",
            backend.name()
        );
    }
}
