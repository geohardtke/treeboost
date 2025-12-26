//! x86_64 SIMD kernel implementations
//!
//! This module contains AVX2/AVX-512 optimized implementations for GBDT operations.
//!
//! # Submodules
//!
//! - `histogram`: Histogram accumulation kernels (gather/scatter)
//! - `merge`: Histogram merge/subtract kernels (SIMD reduction)
//! - `split`: Split finding kernels (gain calculation)
//! - `unpack`: 4-bit bin unpacking kernels

pub mod histogram;
pub mod merge;
pub mod split;
pub mod unpack;

pub use histogram::{histogram_accumulate_avx2, histogram_accumulate_contiguous_avx2};
pub use merge::{
    merge_histogram_counts_avx2, merge_histogram_grads_avx2, merge_histogram_hess_avx2,
    subtract_histogram_counts_avx2, subtract_histogram_grads_avx2, subtract_histogram_hess_avx2,
};
pub use split::{find_best_split_scalar, find_best_split_simd, SplitCandidate};
pub use unpack::{unpack_4bit, unpack_4bit_scalar};
