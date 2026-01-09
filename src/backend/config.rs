//! Backend configuration types.

use crate::defaults::backend as backend_defaults;

/// GPU execution mode for GPU backends.
///
/// This setting controls how much work is offloaded to the GPU.
/// Ignored when using CPU-only backends (Scalar, AVX-512, SVE2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GpuMode {
    /// Automatically select optimal mode based on backend.
    ///
    /// - CUDA: Uses Full mode (low dispatch latency makes it worthwhile)
    /// - WGPU: Uses Hybrid mode (high dispatch latency makes Full slower)
    #[default]
    Auto,

    /// GPU histogram + CPU partition/split (best-first tree growth).
    ///
    /// Uses GPU for the expensive histogram building, but keeps partition
    /// and split-finding on CPU. Best-first tree growth produces higher
    /// quality trees. Recommended for WGPU due to dispatch overhead.
    Hybrid,

    /// Full GPU: histogram + partition + split (level-wise tree growth).
    ///
    /// Keeps all data on GPU, minimizing PCIe transfers. Uses level-wise
    /// tree growth which processes all nodes at each depth level together.
    /// Recommended for CUDA where dispatch overhead is minimal.
    Full,
}

impl GpuMode {
    /// Resolve Auto mode to the optimal concrete mode for a given backend.
    ///
    /// - CUDA: Full (low dispatch latency)
    /// - WGPU/ROCm/Metal: Hybrid (high dispatch latency)
    pub fn resolve(self, backend: BackendType) -> GpuMode {
        match self {
            GpuMode::Auto => match backend {
                BackendType::Cuda => GpuMode::Full,
                BackendType::Wgpu | BackendType::Rocm | BackendType::Metal => GpuMode::Hybrid,
                // Non-GPU backends don't use gpu_mode, but default to Hybrid
                _ => GpuMode::Hybrid,
            },
            other => other,
        }
    }

    /// Check if this mode uses full GPU pipeline (level-wise tree growth).
    pub fn is_full(self, backend: BackendType) -> bool {
        matches!(self.resolve(backend), GpuMode::Full)
    }
}

/// Available backend types for histogram building.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackendType {
    /// Automatically detect the best available backend.
    /// Priority: CUDA > WGPU > AVX-512 > SVE2 > Scalar
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

    /// GPU execution mode (Hybrid or Full).
    /// Only applies to GPU backends (WGPU, CUDA, ROCm, Metal).
    /// Default: Hybrid (GPU histogram + CPU partition/split)
    pub gpu_mode: GpuMode,

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

    /// Enable GPU subgroup operations for histogram building.
    ///
    /// Subgroups can reduce atomic contention but show minimal benefit on
    /// modern NVIDIA GPUs (~1.0x speedup). May help on older AMD or Intel.
    /// Default: false
    pub use_gpu_subgroups: bool,

    /// Batch size threshold for CUDA hybrid backend (rows).
    ///
    /// For batches smaller than this, use CPU to avoid GPU launch overhead.
    /// CUDA has low dispatch overhead (10-100μs), so threshold is low.
    ///
    /// Default: 1000 rows (measured empirically)
    /// Set to 0 to always use GPU, or usize::MAX to always use CPU.
    ///
    /// Run `cargo test --release --features cuda test_gpu_threshold -- --ignored --nocapture`
    /// to measure optimal threshold for your hardware.
    pub cuda_hybrid_threshold: usize,

    /// Batch size threshold for WGPU hybrid backend (rows).
    ///
    /// For batches smaller than this, use CPU to avoid GPU launch overhead.
    /// WGPU has higher dispatch overhead (1-2ms), so threshold is higher.
    ///
    /// Default: 5000 rows (measured empirically)
    /// Set to 0 to always use GPU, or usize::MAX to always use CPU.
    ///
    /// Run `cargo test --release --features gpu test_wgpu_threshold -- --ignored --nocapture`
    /// to measure optimal threshold for your hardware.
    pub wgpu_hybrid_threshold: usize,
}

/// Presets for backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPreset {
    /// Auto-detect best backend.
    Auto,
    /// Force CPU with SIMD.
    CpuOnly,
    /// Force GPU (fail if unavailable).
    GpuRequired,
    /// GPU with scalar fallback.
    GpuPreferred,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            preferred: BackendType::Auto,
            gpu_mode: GpuMode::default(),
            fallback_to_scalar: true,
            tensor_tile_min_rows: backend_defaults::TENSOR_TILE_MIN_ROWS,
            gpu_batch_size: backend_defaults::DEFAULT_GPU_BATCH_SIZE,
            use_gpu_subgroups: false,
            cuda_hybrid_threshold: backend_defaults::CUDA_HYBRID_THRESHOLD,
            wgpu_hybrid_threshold: backend_defaults::WGPU_HYBRID_THRESHOLD,
        }
    }
}

impl BackendConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a preset configuration.
    pub fn with_preset(mut self, preset: BackendPreset) -> Self {
        match preset {
            BackendPreset::Auto => {
                self.preferred = BackendType::Auto;
                self.fallback_to_scalar = true;
            }
            BackendPreset::CpuOnly => {
                self.preferred = BackendType::Scalar;
                self.fallback_to_scalar = true;
            }
            BackendPreset::GpuRequired => {
                self.preferred = BackendType::Wgpu;
                self.fallback_to_scalar = false;
                self.gpu_mode = GpuMode::Auto;
            }
            BackendPreset::GpuPreferred => {
                self.preferred = BackendType::Wgpu;
                self.fallback_to_scalar = true;
                self.gpu_mode = GpuMode::Auto;
            }
        }
        self
    }

    /// Create config that always uses scalar backend.
    pub fn scalar() -> Self {
        Self {
            preferred: BackendType::Scalar,
            gpu_mode: GpuMode::Hybrid,
            fallback_to_scalar: true,
            tensor_tile_min_rows: usize::MAX,
            gpu_batch_size: backend_defaults::DEFAULT_GPU_BATCH_SIZE,
            use_gpu_subgroups: false,
            cuda_hybrid_threshold: backend_defaults::CUDA_HYBRID_THRESHOLD,
            wgpu_hybrid_threshold: backend_defaults::WGPU_HYBRID_THRESHOLD,
        }
    }

    /// Create config that prefers GPU if available.
    ///
    /// Uses `GpuMode::Auto` which selects:
    /// - CUDA: Full mode (low dispatch latency)
    /// - WGPU: Hybrid mode (high dispatch latency)
    pub fn prefer_gpu() -> Self {
        Self {
            preferred: BackendType::Wgpu,
            gpu_mode: GpuMode::Auto,
            fallback_to_scalar: true,
            tensor_tile_min_rows: backend_defaults::TENSOR_TILE_MIN_ROWS,
            gpu_batch_size: backend_defaults::DEFAULT_GPU_BATCH_SIZE,
            use_gpu_subgroups: false,
            cuda_hybrid_threshold: backend_defaults::CUDA_HYBRID_THRESHOLD,
            wgpu_hybrid_threshold: backend_defaults::WGPU_HYBRID_THRESHOLD,
        }
    }

    /// Set the preferred backend type.
    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.preferred = backend;
        self
    }

    /// Set the GPU execution mode.
    ///
    /// - `GpuMode::Hybrid` (default): GPU histogram + CPU partition/split
    /// - `GpuMode::Full`: Full GPU with level-wise tree growth
    ///
    /// Only affects GPU backends (WGPU, CUDA, ROCm, Metal).
    pub fn with_gpu_mode(mut self, mode: GpuMode) -> Self {
        self.gpu_mode = mode;
        self
    }

    /// Set the GPU batch size for histogram building.
    pub fn with_gpu_batch_size(mut self, batch_size: usize) -> Self {
        self.gpu_batch_size = batch_size;
        self
    }

    /// Enable or disable GPU subgroup operations.
    pub fn with_gpu_subgroups(mut self, enabled: bool) -> Self {
        self.use_gpu_subgroups = enabled;
        self
    }

    /// Set the CUDA hybrid threshold (batch size in rows).
    ///
    /// Batches smaller than this use CPU, larger batches use GPU.
    /// Set to 0 to always use GPU, or `usize::MAX` to always use CPU.
    ///
    /// # Example
    /// ```ignore
    /// // Always use GPU for CUDA, even for small batches
    /// let config = BackendConfig::new().with_cuda_threshold(0);
    ///
    /// // Use CPU for all CUDA batches (testing)
    /// let config = BackendConfig::new().with_cuda_threshold(usize::MAX);
    ///
    /// // Custom threshold based on benchmarks
    /// let config = BackendConfig::new().with_cuda_threshold(2000);
    /// ```
    pub fn with_cuda_threshold(mut self, threshold: usize) -> Self {
        self.cuda_hybrid_threshold = threshold;
        self
    }

    /// Set the WGPU hybrid threshold (batch size in rows).
    ///
    /// Batches smaller than this use CPU, larger batches use GPU.
    /// Set to 0 to always use GPU, or `usize::MAX` to always use CPU.
    pub fn with_wgpu_threshold(mut self, threshold: usize) -> Self {
        self.wgpu_hybrid_threshold = threshold;
        self
    }

    /// Set both CUDA and WGPU thresholds to the same value.
    ///
    /// Useful when you want a consistent threshold regardless of GPU type.
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.cuda_hybrid_threshold = threshold;
        self.wgpu_hybrid_threshold = threshold;
        self
    }
}
