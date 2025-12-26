//! Backend configuration types.

/// Available backend types for histogram building.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendType {
    /// Automatically detect the best available backend.
    /// Priority: WGPU > AVX-512 > SVE2 > Scalar
    #[default]
    Auto,

    /// Scalar backend with SIMD loads (AVX2 on x86, NEON on ARM).
    /// Uses column-major layout with scalar scatter.
    Scalar,

    /// WGPU backend for all GPUs (via Vulkan/Metal/DX12).
    /// Uses tensor-tile (row-major) layout with atomic workgroups.
    Wgpu,

    /// AVX-512 tensor-tile backend for x86-64.
    /// Uses vpconflictd for parallel histogram updates.
    Avx512,

    /// SVE2 tensor-tile backend for ARM.
    /// Uses HISTCNT instruction for direct histogram computation.
    Sve2,

    // Future native backends for extreme optimization
    /// CUDA backend for NVIDIA GPUs (bypasses WGPU).
    Cuda,

    /// ROCm backend for AMD GPUs (bypasses WGPU).
    Rocm,

    /// Metal backend for Apple GPUs (bypasses WGPU).
    Metal,
}

impl BackendType {
    /// Check if this backend type is currently implemented.
    pub fn is_implemented(&self) -> bool {
        matches!(self, BackendType::Auto | BackendType::Scalar)
    }

    /// Check if this backend uses tensor-tile (2D row-major) layout.
    pub fn is_tensor_tile(&self) -> bool {
        matches!(
            self,
            BackendType::Wgpu
                | BackendType::Avx512
                | BackendType::Sve2
                | BackendType::Cuda
                | BackendType::Rocm
                | BackendType::Metal
        )
    }
}

/// Configuration for backend selection.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    /// Preferred backend type.
    pub preferred: BackendType,

    /// Whether to fall back to scalar if preferred backend is unavailable.
    pub fallback_to_scalar: bool,

    /// Minimum dataset size to use tensor-tile backends.
    /// Below this threshold, scalar is always used (lower overhead).
    pub tensor_tile_min_rows: usize,

    /// GPU histogram batch size for tree growth.
    /// When growing trees, multiple small histogram builds are batched together
    /// into a single GPU dispatch to amortize dispatch overhead.
    /// Default: 32 (optimal for trees with max_depth 5-6)
    pub gpu_batch_size: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            preferred: BackendType::Auto,
            fallback_to_scalar: true,
            tensor_tile_min_rows: 10_000,
            gpu_batch_size: 32,
        }
    }
}

impl BackendConfig {
    /// Create config that always uses scalar backend.
    pub fn scalar() -> Self {
        Self {
            preferred: BackendType::Scalar,
            fallback_to_scalar: true,
            tensor_tile_min_rows: usize::MAX,
            gpu_batch_size: 32,
        }
    }

    /// Create config that prefers GPU if available.
    pub fn prefer_gpu() -> Self {
        Self {
            preferred: BackendType::Wgpu,
            fallback_to_scalar: true,
            tensor_tile_min_rows: 10_000,
            gpu_batch_size: 32,
        }
    }

    /// Set the GPU batch size for histogram building.
    pub fn with_gpu_batch_size(mut self, batch_size: usize) -> Self {
        self.gpu_batch_size = batch_size;
        self
    }
}
