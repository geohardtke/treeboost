//! Histogram data structure

use crate::dataset::BinEntry;
use rkyv::{Archive, Deserialize, Serialize};

/// Number of bins per histogram (u8 range)
pub const NUM_BINS: usize = 256;

/// Histogram for a single feature
///
/// Fixed-size array of 256 bin entries for gradient/hessian accumulation.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin entries indexed by bin value (0-255)
    bins: [BinEntry; NUM_BINS],
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Create a new empty histogram
    pub fn new() -> Self {
        Self {
            bins: [BinEntry::default(); NUM_BINS],
        }
    }

    /// Clear all bins
    pub fn clear(&mut self) {
        for bin in &mut self.bins {
            *bin = BinEntry::default();
        }
    }

    /// Get a bin entry
    #[inline]
    pub fn get(&self, bin: u8) -> &BinEntry {
        &self.bins[bin as usize]
    }

    /// Get a mutable bin entry
    #[inline]
    pub fn get_mut(&mut self, bin: u8) -> &mut BinEntry {
        &mut self.bins[bin as usize]
    }

    /// Accumulate gradient/hessian into a bin
    #[inline]
    pub fn accumulate(&mut self, bin: u8, gradient: f32, hessian: f32) {
        self.bins[bin as usize].accumulate(gradient, hessian);
    }

    /// Merge another histogram into this one
    pub fn merge(&mut self, other: &Histogram) {
        for (self_bin, other_bin) in self.bins.iter_mut().zip(other.bins.iter()) {
            self_bin.merge(other_bin);
        }
    }

    /// Subtract another histogram from this one (Histogram Subtraction Trick)
    ///
    /// Used to compute sibling histogram: sibling = parent - child
    pub fn subtract(&mut self, other: &Histogram) {
        for (self_bin, other_bin) in self.bins.iter_mut().zip(other.bins.iter()) {
            self_bin.subtract(other_bin);
        }
    }

    /// Compute histogram by subtracting child from parent
    ///
    /// Returns: parent - child (the sibling histogram)
    pub fn from_subtraction(parent: &Histogram, child: &Histogram) -> Self {
        let mut result = parent.clone();
        result.subtract(child);
        result
    }

    /// Get total gradient sum across all bins
    pub fn total_gradient(&self) -> f32 {
        self.bins.iter().map(|b| b.sum_gradients).sum()
    }

    /// Get total hessian sum across all bins
    pub fn total_hessian(&self) -> f32 {
        self.bins.iter().map(|b| b.sum_hessians).sum()
    }

    /// Get total count across all bins
    pub fn total_count(&self) -> u32 {
        self.bins.iter().map(|b| b.count).sum()
    }

    /// Iterate over bins
    pub fn iter(&self) -> impl Iterator<Item = (u8, &BinEntry)> {
        self.bins.iter().enumerate().map(|(i, b)| (i as u8, b))
    }

    /// Get raw bins slice
    pub fn bins(&self) -> &[BinEntry; NUM_BINS] {
        &self.bins
    }
}

/// Collection of histograms for all features at a node
#[derive(Debug, Clone)]
pub struct NodeHistograms {
    /// One histogram per feature
    pub(crate) histograms: Vec<Histogram>,
}

impl NodeHistograms {
    /// Create histograms for all features
    pub fn new(num_features: usize) -> Self {
        Self {
            histograms: vec![Histogram::new(); num_features],
        }
    }

    /// Get histogram for a feature
    #[inline]
    pub fn get(&self, feature_idx: usize) -> &Histogram {
        &self.histograms[feature_idx]
    }

    /// Get mutable histogram for a feature
    #[inline]
    pub fn get_mut(&mut self, feature_idx: usize) -> &mut Histogram {
        &mut self.histograms[feature_idx]
    }

    /// Number of features
    pub fn num_features(&self) -> usize {
        self.histograms.len()
    }

    /// Clear all histograms
    pub fn clear(&mut self) {
        for hist in &mut self.histograms {
            hist.clear();
        }
    }

    /// Merge another set of histograms
    pub fn merge(&mut self, other: &NodeHistograms) {
        for (self_hist, other_hist) in self.histograms.iter_mut().zip(other.histograms.iter()) {
            self_hist.merge(other_hist);
        }
    }

    /// Subtract another set of histograms
    pub fn subtract(&mut self, other: &NodeHistograms) {
        for (self_hist, other_hist) in self.histograms.iter_mut().zip(other.histograms.iter()) {
            self_hist.subtract(other_hist);
        }
    }

    /// Compute sibling histograms from parent and child
    pub fn from_subtraction(parent: &NodeHistograms, child: &NodeHistograms) -> Self {
        Self {
            histograms: parent
                .histograms
                .iter()
                .zip(child.histograms.iter())
                .map(|(p, c)| Histogram::from_subtraction(p, c))
                .collect(),
        }
    }

    /// Iterate over histograms
    pub fn iter(&self) -> impl Iterator<Item = (usize, &Histogram)> {
        self.histograms.iter().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_accumulate() {
        let mut hist = Histogram::new();

        hist.accumulate(0, 1.0, 2.0);
        hist.accumulate(0, 0.5, 1.0);
        hist.accumulate(255, 3.0, 4.0);

        assert_eq!(hist.get(0).sum_gradients, 1.5);
        assert_eq!(hist.get(0).sum_hessians, 3.0);
        assert_eq!(hist.get(0).count, 2);
        assert_eq!(hist.get(255).count, 1);
    }

    #[test]
    fn test_histogram_subtraction_trick() {
        let mut parent = Histogram::new();
        let mut child = Histogram::new();

        // Parent has all data
        parent.accumulate(0, 10.0, 20.0);
        parent.accumulate(1, 5.0, 10.0);

        // Child has subset
        child.accumulate(0, 3.0, 6.0);
        child.accumulate(1, 2.0, 4.0);

        // Sibling = parent - child
        let sibling = Histogram::from_subtraction(&parent, &child);

        assert_eq!(sibling.get(0).sum_gradients, 7.0);
        assert_eq!(sibling.get(0).sum_hessians, 14.0);
        assert_eq!(sibling.get(1).sum_gradients, 3.0);
    }

    #[test]
    fn test_node_histograms() {
        let mut hists = NodeHistograms::new(3);

        hists.get_mut(0).accumulate(5, 1.0, 2.0);
        hists.get_mut(1).accumulate(10, 3.0, 4.0);
        hists.get_mut(2).accumulate(15, 5.0, 6.0);

        assert_eq!(hists.num_features(), 3);
        assert_eq!(hists.get(0).get(5).sum_gradients, 1.0);
        assert_eq!(hists.get(1).get(10).sum_gradients, 3.0);
    }
}
