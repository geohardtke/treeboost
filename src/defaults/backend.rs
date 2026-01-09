pub const TENSOR_TILE_MIN_ROWS: usize = 10_000;
pub const DEFAULT_GPU_BATCH_SIZE: usize = 32;

/// Default hybrid threshold for CUDA backend (rows).
///
/// CUDA has low dispatch overhead (10-100μs), so can use GPU for smaller batches.
/// This default is measured empirically - run benchmark to measure for your hardware:
///
/// ```bash
/// cargo test --release --features cuda test_gpu_threshold -- --ignored --nocapture
/// ```
///
/// Measured on RTX 3060: GPU becomes faster at ~1500 rows, using 2000 for safety margin
pub const CUDA_HYBRID_THRESHOLD: usize = 2000;

/// Default hybrid threshold for WGPU backend (rows).
///
/// WGPU has higher dispatch overhead (1-2ms), so needs larger batches to be worthwhile.
/// This default is measured empirically - run benchmark to measure for your hardware:
///
/// ```bash
/// cargo test --release --features gpu test_wgpu_threshold -- --ignored --nocapture
/// ```
///
/// Measured on RTX 3060: GPU becomes faster at ~2000 rows, using 3000 for safety margin
pub const WGPU_HYBRID_THRESHOLD: usize = 3000;
