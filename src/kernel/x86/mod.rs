//! x86_64 SIMD kernel implementations
//!
//! This module contains AVX2/AVX-512 optimized implementations for GBDT operations.
//!
//! # Submodules
//!
//! - `histogram`: Histogram accumulation kernels (gather/scatter)
//! - `split`: Split finding kernels (gain calculation)

pub mod histogram;
pub mod split;

pub use histogram::{histogram_accumulate_avx2, histogram_accumulate_contiguous_avx2};
pub use split::{find_best_split_scalar, find_best_split_simd, SplitCandidate};
