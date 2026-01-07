//! SIMD-optimized 4-bit bin unpacking
//!
//! Efficiently extracts packed 4-bit bins into u8 buffers using AVX2.
//!
//! # Algorithm
//!
//! Packed format: each byte contains 2 bins (high 4 bits + low 4 bits)
//! - Byte 0: [bin0_high, bin1_low]
//! - Byte 1: [bin2_high, bin3_low]
//! - etc.
//!
//! SIMD unpacking:
//! 1. Load 32 packed bytes (64 bins) with `_mm256_loadu_si256`
//! 2. Extract high nibbles: `(packed >> 4) & 0x0F`
//! 3. Extract low nibbles: `packed & 0x0F`
//! 4. Interleave high and low to restore original order
//!
//! # Performance
//!
//! - Scalar: ~1 cycle per bin (load, shift/mask, store)
//! - SIMD: ~0.5 cycles per bin (32 bytes → 64 bins in ~32 cycles)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Unpack 32 packed bytes (64 bins) into a 64-byte output buffer using AVX2
///
/// # Safety
/// - Requires AVX2 support
/// - `packed` must point to at least 32 valid bytes
/// - `output` must point to at least 64 valid bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_4bit_avx2(packed: *const u8, output: *mut u8) {
    // Load 32 packed bytes (64 bins)
    let data = _mm256_loadu_si256(packed as *const __m256i);

    // Mask for low nibbles
    let mask_low = _mm256_set1_epi8(0x0F);

    // Extract high nibbles: shift right by 4, then mask
    let high = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask_low);

    // Extract low nibbles: just mask
    let low = _mm256_and_si256(data, mask_low);

    // Now we need to interleave high and low:
    // Original byte i contains [bin 2i (high), bin 2i+1 (low)]
    // So we need: output[2i] = high[i], output[2i+1] = low[i]

    // Use unpack instructions to interleave
    // unpacklo interleaves bytes from low halves of both vectors
    // unpackhi interleaves bytes from high halves

    // First, interleave within 128-bit lanes
    let interleaved_lo = _mm256_unpacklo_epi8(high, low);
    let interleaved_hi = _mm256_unpackhi_epi8(high, low);

    // The 256-bit unpack operates on 128-bit halves independently
    // So we need to fix the lane crossing

    // Extract 128-bit halves
    let lo_128 = _mm256_castsi256_si128(interleaved_lo);
    let hi_128_lo = _mm256_extracti128_si256(interleaved_lo, 1);
    let lo_128_hi = _mm256_castsi256_si128(interleaved_hi);
    let hi_128_hi = _mm256_extracti128_si256(interleaved_hi, 1);

    // Reconstruct proper order
    let result_0 = _mm256_set_m128i(lo_128_hi, lo_128);
    let result_1 = _mm256_set_m128i(hi_128_hi, hi_128_lo);

    // Store 64 bytes
    _mm256_storeu_si256(output as *mut __m256i, result_0);
    _mm256_storeu_si256(output.add(32) as *mut __m256i, result_1);
}

/// Unpack packed 4-bit data to a buffer using AVX2
///
/// Handles arbitrary lengths by processing 32-byte chunks with SIMD
/// and falling back to scalar for the remainder.
///
/// # Safety
/// - Requires AVX2 support
/// - `packed` must point to at least `packed_len` valid bytes
/// - `output` must point to at least `packed_len * 2` valid bytes
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_4bit_buffer_avx2(packed: *const u8, packed_len: usize, output: *mut u8) {
    let chunks = packed_len / 32;
    let remainder = packed_len % 32;

    // Process 32-byte chunks with SIMD
    for i in 0..chunks {
        let packed_ptr = packed.add(i * 32);
        let output_ptr = output.add(i * 64);
        unpack_4bit_avx2(packed_ptr, output_ptr);
    }

    // Handle remainder with scalar
    let rem_start = chunks * 32;
    for i in 0..remainder {
        let byte = *packed.add(rem_start + i);
        let high = byte >> 4;
        let low = byte & 0x0F;
        *output.add((rem_start + i) * 2) = high;
        *output.add((rem_start + i) * 2 + 1) = low;
    }
}

/// Scalar fallback for 4-bit unpacking
#[inline]
pub fn unpack_4bit_scalar(packed: &[u8], output: &mut [u8]) {
    debug_assert!(output.len() >= packed.len() * 2);

    for (i, &byte) in packed.iter().enumerate() {
        output[i * 2] = byte >> 4;
        output[i * 2 + 1] = byte & 0x0F;
    }
}

/// Unpack a range of packed data to a buffer with runtime SIMD dispatch
///
/// # Arguments
/// - `packed`: Packed 4-bit data
/// - `output`: Output buffer (must be at least `packed.len() * 2` bytes)
#[inline]
pub fn unpack_4bit(packed: &[u8], output: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe {
                unpack_4bit_buffer_avx2(packed.as_ptr(), packed.len(), output.as_mut_ptr());
            }
            return;
        }
    }

    unpack_4bit_scalar(packed, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_scalar() {
        // Pack: [0x01, 0x23, 0x45, 0x67] → bins [0, 1, 2, 3, 4, 5, 6, 7]
        let packed = vec![0x01, 0x23, 0x45, 0x67];
        let mut output = vec![0u8; 8];

        unpack_4bit_scalar(&packed, &mut output);

        assert_eq!(output, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_unpack_all_values() {
        // Test all 4-bit values
        let packed: Vec<u8> = (0..=255u8).collect();
        let mut output = vec![0u8; 512];

        unpack_4bit_scalar(&packed, &mut output);

        for i in 0..256 {
            let expected_high = (i >> 4) as u8;
            let expected_low = (i & 0x0F) as u8;
            assert_eq!(
                output[i * 2],
                expected_high,
                "High nibble mismatch at {}",
                i
            );
            assert_eq!(
                output[i * 2 + 1],
                expected_low,
                "Low nibble mismatch at {}",
                i
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_unpack_avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Test various sizes
        for size in [32, 64, 100, 128, 256, 1000] {
            let packed: Vec<u8> = (0..size).map(|i| i as u8).collect();
            let mut output_scalar = vec![0u8; size * 2];
            let mut output_simd = vec![0u8; size * 2];

            unpack_4bit_scalar(&packed, &mut output_scalar);
            unpack_4bit(&packed, &mut output_simd);

            assert_eq!(output_scalar, output_simd, "Mismatch for size {}", size);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_unpack_avx2_exact_chunk() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }

        // Exactly 32 bytes (64 bins) - one SIMD chunk
        let packed: Vec<u8> = (0..32).collect();
        let mut output = vec![0u8; 64];

        unsafe {
            unpack_4bit_avx2(packed.as_ptr(), output.as_mut_ptr());
        }

        // Verify all values
        for i in 0..32 {
            let expected_high = (i >> 4) as u8;
            let expected_low = (i & 0x0F) as u8;
            assert_eq!(
                output[i * 2],
                expected_high,
                "High nibble mismatch at packed byte {}",
                i
            );
            assert_eq!(
                output[i * 2 + 1],
                expected_low,
                "Low nibble mismatch at packed byte {}",
                i
            );
        }
    }

    #[test]
    fn test_unpack_runtime_dispatch() {
        let packed: Vec<u8> = (0..100).collect();
        let mut output = vec![0u8; 200];

        unpack_4bit(&packed, &mut output);

        // Verify correctness
        for i in 0..100 {
            let expected_high = (i >> 4) as u8;
            let expected_low = (i & 0x0F) as u8;
            assert_eq!(output[i * 2], expected_high);
            assert_eq!(output[i * 2 + 1], expected_low);
        }
    }
}
