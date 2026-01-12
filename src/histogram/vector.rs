//! Vector histogram data structures for multi-output learning
//!
//! Provides histogram types that support multiple output dimensions (labels/targets).
//! Used for multi-label classification, multi-target regression, and the unified
//! approach where a single tree contains vector leaf values.
//!
//! # Memory Layout
//!
//! The gradient/hessian buffer uses a flat, strided layout for cache efficiency:
//! ```text
//! [bin0_grad0, bin0_hess0, bin0_grad1, bin0_hess1, ..., bin0_gradK, bin0_hessK,
//!  bin1_grad0, bin1_hess0, ..., binN_gradK, binN_hessK]
//! ```
//!
//! Counts are stored separately since they're shared across outputs:
//! ```text
//! [count0, count1, ..., count255]
//! ```
//!
//! This layout provides:
//! - Good cache locality when iterating through bins for split finding
//! - Efficient strided access for per-output operations
//! - Minimal memory overhead compared to separate histograms

use rkyv::{Archive, Deserialize, Serialize};

/// Number of bins per histogram (u8 range)
pub const NUM_BINS: usize = 256;

/// Vector histogram for a single feature with multiple outputs
///
/// Stores gradient and hessian sums for each (bin, output) combination.
/// Counts are shared across outputs since the same samples fall into the same bin
/// regardless of which output we're considering.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VectorHistogram {
    /// Number of output dimensions
    num_outputs: usize,

    /// Flat buffer for gradients and hessians
    /// Layout: [bin0_grad0, bin0_hess0, bin0_grad1, bin0_hess1, ..., binN_gradK, binN_hessK]
    /// Size: NUM_BINS * num_outputs * 2
    grad_hess_buffer: Vec<f32>,

    /// Count per bin (shared across outputs)
    /// Size: NUM_BINS
    counts: Vec<u32>,
}

impl VectorHistogram {
    /// Create a new empty vector histogram
    pub fn new(num_outputs: usize) -> Self {
        debug_assert!(num_outputs > 0, "num_outputs must be positive");

        Self {
            num_outputs,
            grad_hess_buffer: vec![0.0; NUM_BINS * num_outputs * 2],
            counts: vec![0; NUM_BINS],
        }
    }

    /// Get number of outputs
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Get stride for accessing different bins
    /// Each bin has num_outputs * 2 values (grad + hess for each output)
    #[inline]
    fn bin_stride(&self) -> usize {
        self.num_outputs * 2
    }

    /// Get the buffer index for a specific (bin, output, is_hessian) combination
    #[inline]
    fn buffer_index(&self, bin: u8, output_idx: usize, is_hessian: bool) -> usize {
        let bin_offset = (bin as usize) * self.bin_stride();
        let output_offset = output_idx * 2;
        bin_offset + output_offset + if is_hessian { 1 } else { 0 }
    }

    /// Accumulate gradients and hessians for one sample into a bin
    ///
    /// # Arguments
    /// * `bin` - Bin index (0-255)
    /// * `gradients` - Gradient values for each output (length must be num_outputs)
    /// * `hessians` - Hessian values for each output (length must be num_outputs)
    #[inline]
    pub fn accumulate(&mut self, bin: u8, gradients: &[f32], hessians: &[f32]) {
        debug_assert_eq!(gradients.len(), self.num_outputs);
        debug_assert_eq!(hessians.len(), self.num_outputs);

        let bin_offset = (bin as usize) * self.bin_stride();

        // Accumulate all outputs
        unsafe {
            for k in 0..self.num_outputs {
                let offset = bin_offset + k * 2;
                *self.grad_hess_buffer.get_unchecked_mut(offset) += gradients.get_unchecked(k);
                *self.grad_hess_buffer.get_unchecked_mut(offset + 1) += hessians.get_unchecked(k);
            }
        }

        // Increment count (shared across outputs)
        self.counts[bin as usize] += 1;
    }

    /// Accumulate gradients and hessians for one sample using interleaved input
    ///
    /// # Arguments
    /// * `bin` - Bin index (0-255)
    /// * `grad_hess` - Interleaved [grad0, hess0, grad1, hess1, ...] values
    #[inline]
    pub fn accumulate_interleaved(&mut self, bin: u8, grad_hess: &[f32]) {
        debug_assert_eq!(grad_hess.len(), self.num_outputs * 2);

        let bin_offset = (bin as usize) * self.bin_stride();

        unsafe {
            for k in 0..(self.num_outputs * 2) {
                *self.grad_hess_buffer.get_unchecked_mut(bin_offset + k) +=
                    grad_hess.get_unchecked(k);
            }
        }

        self.counts[bin as usize] += 1;
    }

    /// Batch accumulate multiple samples
    ///
    /// Uses an unrolled loop (8x) for better instruction-level parallelism.
    ///
    /// # Arguments
    /// * `bins` - Bin values for each sample
    /// * `gradients` - Flat gradient buffer [sample0_out0, sample0_out1, ..., sample1_out0, ...]
    /// * `hessians` - Flat hessian buffer with same layout
    #[inline]
    pub fn accumulate_batch(&mut self, bins: &[u8], gradients: &[f32], hessians: &[f32]) {
        let num_samples = bins.len();
        let num_outputs = self.num_outputs;

        debug_assert_eq!(gradients.len(), num_samples * num_outputs);
        debug_assert_eq!(hessians.len(), num_samples * num_outputs);

        let chunks = num_samples / 8;
        let remainder = num_samples % 8;
        let bin_stride = self.bin_stride();

        unsafe {
            // Process 8 samples at a time for better ILP
            for chunk in 0..chunks {
                let base = chunk * 8;

                // Load all 8 bins
                let bin0 = *bins.get_unchecked(base) as usize;
                let bin1 = *bins.get_unchecked(base + 1) as usize;
                let bin2 = *bins.get_unchecked(base + 2) as usize;
                let bin3 = *bins.get_unchecked(base + 3) as usize;
                let bin4 = *bins.get_unchecked(base + 4) as usize;
                let bin5 = *bins.get_unchecked(base + 5) as usize;
                let bin6 = *bins.get_unchecked(base + 6) as usize;
                let bin7 = *bins.get_unchecked(base + 7) as usize;

                // Increment counts
                *self.counts.get_unchecked_mut(bin0) += 1;
                *self.counts.get_unchecked_mut(bin1) += 1;
                *self.counts.get_unchecked_mut(bin2) += 1;
                *self.counts.get_unchecked_mut(bin3) += 1;
                *self.counts.get_unchecked_mut(bin4) += 1;
                *self.counts.get_unchecked_mut(bin5) += 1;
                *self.counts.get_unchecked_mut(bin6) += 1;
                *self.counts.get_unchecked_mut(bin7) += 1;

                // For each output, accumulate gradients and hessians
                for k in 0..num_outputs {
                    let g_offset = k;
                    let h_offset = k;
                    let buf_offset = k * 2;

                    // Load gradients for this output
                    let g0 = *gradients.get_unchecked((base) * num_outputs + g_offset);
                    let g1 = *gradients.get_unchecked((base + 1) * num_outputs + g_offset);
                    let g2 = *gradients.get_unchecked((base + 2) * num_outputs + g_offset);
                    let g3 = *gradients.get_unchecked((base + 3) * num_outputs + g_offset);
                    let g4 = *gradients.get_unchecked((base + 4) * num_outputs + g_offset);
                    let g5 = *gradients.get_unchecked((base + 5) * num_outputs + g_offset);
                    let g6 = *gradients.get_unchecked((base + 6) * num_outputs + g_offset);
                    let g7 = *gradients.get_unchecked((base + 7) * num_outputs + g_offset);

                    // Load hessians for this output
                    let h0 = *hessians.get_unchecked((base) * num_outputs + h_offset);
                    let h1 = *hessians.get_unchecked((base + 1) * num_outputs + h_offset);
                    let h2 = *hessians.get_unchecked((base + 2) * num_outputs + h_offset);
                    let h3 = *hessians.get_unchecked((base + 3) * num_outputs + h_offset);
                    let h4 = *hessians.get_unchecked((base + 4) * num_outputs + h_offset);
                    let h5 = *hessians.get_unchecked((base + 5) * num_outputs + h_offset);
                    let h6 = *hessians.get_unchecked((base + 6) * num_outputs + h_offset);
                    let h7 = *hessians.get_unchecked((base + 7) * num_outputs + h_offset);

                    // Accumulate into histogram bins
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin0 * bin_stride + buf_offset) += g0;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin0 * bin_stride + buf_offset + 1) += h0;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin1 * bin_stride + buf_offset) += g1;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin1 * bin_stride + buf_offset + 1) += h1;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin2 * bin_stride + buf_offset) += g2;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin2 * bin_stride + buf_offset + 1) += h2;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin3 * bin_stride + buf_offset) += g3;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin3 * bin_stride + buf_offset + 1) += h3;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin4 * bin_stride + buf_offset) += g4;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin4 * bin_stride + buf_offset + 1) += h4;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin5 * bin_stride + buf_offset) += g5;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin5 * bin_stride + buf_offset + 1) += h5;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin6 * bin_stride + buf_offset) += g6;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin6 * bin_stride + buf_offset + 1) += h6;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin7 * bin_stride + buf_offset) += g7;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin7 * bin_stride + buf_offset + 1) += h7;
                }
            }

            // Handle remainder
            let rem_base = chunks * 8;
            for i in 0..remainder {
                let sample_idx = rem_base + i;
                let bin = *bins.get_unchecked(sample_idx) as usize;

                *self.counts.get_unchecked_mut(bin) += 1;

                for k in 0..num_outputs {
                    let g = *gradients.get_unchecked(sample_idx * num_outputs + k);
                    let h = *hessians.get_unchecked(sample_idx * num_outputs + k);
                    let buf_offset = k * 2;

                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin * bin_stride + buf_offset) += g;
                    *self
                        .grad_hess_buffer
                        .get_unchecked_mut(bin * bin_stride + buf_offset + 1) += h;
                }
            }
        }
    }

    /// Get gradient and hessian sum for a specific bin and output
    #[inline]
    pub fn get_output_stats(&self, bin: u8, output_idx: usize) -> (f32, f32) {
        debug_assert!(output_idx < self.num_outputs);

        let grad_idx = self.buffer_index(bin, output_idx, false);
        let hess_idx = self.buffer_index(bin, output_idx, true);

        (
            self.grad_hess_buffer[grad_idx],
            self.grad_hess_buffer[hess_idx],
        )
    }

    /// Get all gradient sums for a bin (one per output)
    #[inline]
    pub fn get_all_gradients(&self, bin: u8) -> Vec<f32> {
        let bin_offset = (bin as usize) * self.bin_stride();
        (0..self.num_outputs)
            .map(|k| self.grad_hess_buffer[bin_offset + k * 2])
            .collect()
    }

    /// Get all hessian sums for a bin (one per output)
    #[inline]
    pub fn get_all_hessians(&self, bin: u8) -> Vec<f32> {
        let bin_offset = (bin as usize) * self.bin_stride();
        (0..self.num_outputs)
            .map(|k| self.grad_hess_buffer[bin_offset + k * 2 + 1])
            .collect()
    }

    /// Get count for a bin (shared across outputs)
    #[inline]
    pub fn get_count(&self, bin: u8) -> u32 {
        self.counts[bin as usize]
    }

    /// Get total gradient and hessian sum for a specific output across all bins
    pub fn total_output(&self, output_idx: usize) -> (f32, f32) {
        debug_assert!(output_idx < self.num_outputs);

        let bin_stride = self.bin_stride();
        let buf_offset = output_idx * 2;

        let mut total_grad = 0.0f32;
        let mut total_hess = 0.0f32;

        for bin in 0..NUM_BINS {
            total_grad += self.grad_hess_buffer[bin * bin_stride + buf_offset];
            total_hess += self.grad_hess_buffer[bin * bin_stride + buf_offset + 1];
        }

        (total_grad, total_hess)
    }

    /// Get total count across all bins
    pub fn total_count(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// Get totals for all outputs: Vec<(gradient_sum, hessian_sum)>
    pub fn totals_all_outputs(&self) -> Vec<(f32, f32)> {
        (0..self.num_outputs)
            .map(|k| self.total_output(k))
            .collect()
    }

    /// Clear all bins
    pub fn clear(&mut self) {
        self.grad_hess_buffer.fill(0.0);
        self.counts.fill(0);
    }

    /// Merge another histogram into this one
    pub fn merge(&mut self, other: &VectorHistogram) {
        debug_assert_eq!(self.num_outputs, other.num_outputs);

        for (a, b) in self
            .grad_hess_buffer
            .iter_mut()
            .zip(other.grad_hess_buffer.iter())
        {
            *a += *b;
        }

        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a += *b;
        }
    }

    /// Subtract another histogram from this one (for Histogram Subtraction Trick)
    pub fn subtract(&mut self, other: &VectorHistogram) {
        debug_assert_eq!(self.num_outputs, other.num_outputs);

        for (a, b) in self
            .grad_hess_buffer
            .iter_mut()
            .zip(other.grad_hess_buffer.iter())
        {
            *a -= *b;
        }

        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a -= *b;
        }
    }

    /// Create histogram by subtracting child from parent (for sibling computation)
    pub fn from_subtraction(parent: &VectorHistogram, child: &VectorHistogram) -> Self {
        debug_assert_eq!(parent.num_outputs, child.num_outputs);

        let mut result = parent.clone();
        result.subtract(child);
        result
    }

    /// Get length of grad_hess buffer (for testing/debugging)
    pub fn grad_hess_buffer_len(&self) -> usize {
        self.grad_hess_buffer.len()
    }

    /// Get length of counts buffer (for testing/debugging)
    pub fn counts_len(&self) -> usize {
        self.counts.len()
    }

    /// Get raw grad_hess buffer slice (for advanced operations)
    pub fn grad_hess_buffer(&self) -> &[f32] {
        &self.grad_hess_buffer
    }

    /// Get mutable raw grad_hess buffer slice (for advanced operations)
    pub fn grad_hess_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.grad_hess_buffer
    }

    /// Get raw counts slice
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// Get mutable raw counts slice
    pub fn counts_mut(&mut self) -> &mut [u32] {
        &mut self.counts
    }
}

/// Collection of vector histograms for all features at a node
#[derive(Debug, Clone)]
pub struct VectorNodeHistograms {
    /// One histogram per feature
    histograms: Vec<VectorHistogram>,
    /// Number of outputs (cached for convenience)
    num_outputs: usize,
}

impl VectorNodeHistograms {
    /// Create vector histograms for all features
    pub fn new(num_features: usize, num_outputs: usize) -> Self {
        Self {
            histograms: (0..num_features)
                .map(|_| VectorHistogram::new(num_outputs))
                .collect(),
            num_outputs,
        }
    }

    /// Get histogram for a feature
    #[inline]
    pub fn get(&self, feature_idx: usize) -> &VectorHistogram {
        &self.histograms[feature_idx]
    }

    /// Get mutable histogram for a feature
    #[inline]
    pub fn get_mut(&mut self, feature_idx: usize) -> &mut VectorHistogram {
        &mut self.histograms[feature_idx]
    }

    /// Number of features
    pub fn num_features(&self) -> usize {
        self.histograms.len()
    }

    /// Number of outputs
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Clear all histograms
    pub fn clear(&mut self) {
        for hist in &mut self.histograms {
            hist.clear();
        }
    }

    /// Merge another set of histograms
    pub fn merge(&mut self, other: &VectorNodeHistograms) {
        for (self_hist, other_hist) in self.histograms.iter_mut().zip(other.histograms.iter()) {
            self_hist.merge(other_hist);
        }
    }

    /// Subtract another set of histograms
    pub fn subtract(&mut self, other: &VectorNodeHistograms) {
        for (self_hist, other_hist) in self.histograms.iter_mut().zip(other.histograms.iter()) {
            self_hist.subtract(other_hist);
        }
    }

    /// Compute sibling histograms from parent and child
    pub fn from_subtraction(parent: &VectorNodeHistograms, child: &VectorNodeHistograms) -> Self {
        Self {
            histograms: parent
                .histograms
                .iter()
                .zip(child.histograms.iter())
                .map(|(p, c)| VectorHistogram::from_subtraction(p, c))
                .collect(),
            num_outputs: parent.num_outputs,
        }
    }

    /// Iterate over histograms
    pub fn iter(&self) -> impl Iterator<Item = (usize, &VectorHistogram)> {
        self.histograms.iter().enumerate()
    }

    /// Create from a vector of histograms
    pub fn from_vec(histograms: Vec<VectorHistogram>, num_outputs: usize) -> Self {
        Self {
            histograms,
            num_outputs,
        }
    }

    /// Get internal histograms vector
    pub fn into_vec(self) -> Vec<VectorHistogram> {
        self.histograms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_histogram_new() {
        let hist = VectorHistogram::new(3);
        assert_eq!(hist.num_outputs(), 3);
        assert_eq!(hist.total_count(), 0);
    }

    #[test]
    fn test_vector_histogram_accumulate() {
        let mut hist = VectorHistogram::new(2);

        hist.accumulate(0, &[1.0, 2.0], &[0.5, 0.5]);

        let (g0, h0) = hist.get_output_stats(0, 0);
        let (g1, h1) = hist.get_output_stats(0, 1);

        assert_eq!(g0, 1.0);
        assert_eq!(h0, 0.5);
        assert_eq!(g1, 2.0);
        assert_eq!(h1, 0.5);
        assert_eq!(hist.get_count(0), 1);
    }

    #[test]
    fn test_vector_histogram_totals() {
        let mut hist = VectorHistogram::new(2);

        hist.accumulate(0, &[1.0, 2.0], &[0.5, 0.5]);
        hist.accumulate(1, &[3.0, 4.0], &[1.0, 1.0]);

        let (total_g0, total_h0) = hist.total_output(0);
        let (total_g1, total_h1) = hist.total_output(1);

        assert_eq!(total_g0, 4.0);
        assert_eq!(total_h0, 1.5);
        assert_eq!(total_g1, 6.0);
        assert_eq!(total_h1, 1.5);
        assert_eq!(hist.total_count(), 2);
    }

    #[test]
    fn test_buffer_layout() {
        let num_outputs = 3;
        let hist = VectorHistogram::new(num_outputs);

        // Buffer should be: 256 bins * 3 outputs * 2 (grad+hess)
        assert_eq!(hist.grad_hess_buffer_len(), 256 * 3 * 2);
        assert_eq!(hist.counts_len(), 256);

        // Verify stride
        assert_eq!(hist.bin_stride(), 6); // 3 outputs * 2
    }
}
