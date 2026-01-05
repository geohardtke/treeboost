//! Distribution distance metrics for drift detection
//!
//! Provides metrics for comparing probability distributions:
//! - PSI (Population Stability Index)
//! - Kolmogorov-Smirnov test statistic
//! - Jensen-Shannon divergence

/// Trait for computing distribution distance metrics
pub trait DistributionMetric: Send + Sync {
    /// Compute metric between reference and target distributions
    ///
    /// # Arguments
    /// * `reference` - Reference distribution (e.g., training data)
    /// * `target` - Target distribution (e.g., inference data)
    ///
    /// # Returns
    /// A non-negative value where higher means more drift
    fn compute(&self, reference: &[f32], target: &[f32]) -> f32;

    /// Name of the metric
    fn name(&self) -> &'static str;

    /// Default warning threshold
    fn warning_threshold(&self) -> f32;

    /// Default critical threshold
    fn critical_threshold(&self) -> f32;
}

/// Population Stability Index (PSI)
///
/// Measures how much a distribution has shifted from a reference.
///
/// PSI = Σ (target% - reference%) * ln(target% / reference%)
///
/// # Interpretation
/// - PSI < 0.1: No significant shift
/// - 0.1 <= PSI < 0.25: Moderate shift (investigate)
/// - PSI >= 0.25: Significant shift (action required)
///
/// Commonly used in credit risk and fraud detection for monitoring
/// feature distribution stability.
#[derive(Debug, Clone)]
pub struct PSI {
    /// Number of bins for continuous features
    num_bins: usize,
    /// Small value to avoid log(0)
    epsilon: f32,
    /// Maximum contribution from a single bin (for numerical stability)
    max_term: f32,
}

impl PSI {
    /// Create a new PSI metric
    ///
    /// # Arguments
    /// * `num_bins` - Number of bins for binning continuous features (default: 10)
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: num_bins.max(2),
            epsilon: 1e-10,
            max_term: 1.0, // Cap individual bin contributions for stability
        }
    }

    /// Create PSI with custom stability parameters
    pub fn with_stability(num_bins: usize, epsilon: f32, max_term: f32) -> Self {
        Self {
            num_bins: num_bins.max(2),
            epsilon: epsilon.max(1e-15),
            max_term: max_term.max(0.1),
        }
    }
}

impl Default for PSI {
    fn default() -> Self {
        Self::new(10)
    }
}

impl DistributionMetric for PSI {
    fn compute(&self, reference: &[f32], target: &[f32]) -> f32 {
        if reference.is_empty() || target.is_empty() {
            return 0.0;
        }

        // Get min and max from reference for bin edges
        let (min_val, max_val) = reference.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });

        if (max_val - min_val).abs() < self.epsilon {
            // All values are the same
            return 0.0;
        }

        let bin_width = (max_val - min_val) / self.num_bins as f32;

        // Count reference distribution
        let mut ref_counts = vec![0usize; self.num_bins];
        for &v in reference {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(self.num_bins - 1);
            ref_counts[bin] += 1;
        }

        // Count target distribution
        let mut target_counts = vec![0usize; self.num_bins];
        for &v in target {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(self.num_bins - 1);
            target_counts[bin] += 1;
        }

        // Convert to proportions
        let ref_total = reference.len() as f32;
        let target_total = target.len() as f32;

        // Compute PSI with numerical stability
        let mut psi = 0.0f32;
        for i in 0..self.num_bins {
            let ref_pct = (ref_counts[i] as f32 / ref_total).max(self.epsilon);
            let target_pct = (target_counts[i] as f32 / target_total).max(self.epsilon);

            // Standard PSI term with capping for numerical stability
            let term = (target_pct - ref_pct) * (target_pct / ref_pct).ln();
            psi += term.abs().min(self.max_term);
        }

        psi
    }

    fn name(&self) -> &'static str {
        "psi"
    }

    fn warning_threshold(&self) -> f32 {
        0.1
    }

    fn critical_threshold(&self) -> f32 {
        0.25
    }
}

/// Kolmogorov-Smirnov test statistic
///
/// Maximum absolute difference between two cumulative distribution functions.
///
/// KS = max |CDF_ref(x) - CDF_target(x)|
///
/// # Interpretation
/// - KS < 0.1: Very similar distributions
/// - 0.1 <= KS < 0.2: Some difference
/// - KS >= 0.2: Significant difference
///
/// Distribution-free test that works well for detecting shifts in any
/// part of the distribution (not just mean or variance).
#[derive(Debug, Clone, Default)]
pub struct KolmogorovSmirnov;

impl KolmogorovSmirnov {
    /// Create a new KS metric
    pub fn new() -> Self {
        Self
    }
}

impl DistributionMetric for KolmogorovSmirnov {
    fn compute(&self, reference: &[f32], target: &[f32]) -> f32 {
        if reference.is_empty() || target.is_empty() {
            return 0.0;
        }

        // Sort both distributions
        let mut ref_sorted: Vec<f32> = reference.to_vec();
        let mut target_sorted: Vec<f32> = target.to_vec();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        target_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_ref = reference.len() as f32;
        let n_target = target.len() as f32;

        // Two-pointer algorithm for computing max CDF difference
        // KS statistic = max_x |F_ref(x) - F_target(x)|
        // where F(x) = proportion of values <= x
        let mut i = 0usize;
        let mut j = 0usize;
        let mut max_diff = 0.0f32;

        while i < ref_sorted.len() || j < target_sorted.len() {
            // Get current values (use infinity if exhausted)
            let ref_val = if i < ref_sorted.len() { ref_sorted[i] } else { f32::INFINITY };
            let target_val = if j < target_sorted.len() { target_sorted[j] } else { f32::INFINITY };

            // Advance pointer(s) for the smaller value, or both if equal
            if ref_val < target_val {
                i += 1;
            } else if target_val < ref_val {
                j += 1;
            } else {
                // Equal values: advance both
                i += 1;
                j += 1;
            }

            // Compute CDFs after advancing
            let ref_cdf = i as f32 / n_ref;
            let target_cdf = j as f32 / n_target;
            max_diff = max_diff.max((ref_cdf - target_cdf).abs());
        }

        max_diff
    }

    fn name(&self) -> &'static str {
        "ks"
    }

    fn warning_threshold(&self) -> f32 {
        0.1
    }

    fn critical_threshold(&self) -> f32 {
        0.2
    }
}

/// Jensen-Shannon divergence
///
/// Symmetric and bounded version of KL divergence.
///
/// JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5 * (P + Q)
///
/// # Properties
/// - Range: [0, ln(2)] ≈ [0, 0.693]
/// - Symmetric: JSD(P, Q) = JSD(Q, P)
/// - Bounded: Unlike KL divergence, always finite
///
/// # Interpretation
/// - JSD < 0.05: Very similar
/// - 0.05 <= JSD < 0.15: Moderate difference
/// - JSD >= 0.15: Significant difference
#[derive(Debug, Clone)]
pub struct JensenShannon {
    /// Number of bins for continuous features
    num_bins: usize,
    /// Small value to avoid log(0)
    epsilon: f32,
}

impl JensenShannon {
    /// Create a new Jensen-Shannon metric
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: num_bins.max(2),
            epsilon: 1e-10,
        }
    }
}

impl Default for JensenShannon {
    fn default() -> Self {
        Self::new(10)
    }
}

impl DistributionMetric for JensenShannon {
    fn compute(&self, reference: &[f32], target: &[f32]) -> f32 {
        if reference.is_empty() || target.is_empty() {
            return 0.0;
        }

        // Get min and max from both distributions
        let all_min = reference
            .iter()
            .chain(target.iter())
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let all_max = reference
            .iter()
            .chain(target.iter())
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        if (all_max - all_min).abs() < self.epsilon {
            return 0.0;
        }

        let bin_width = (all_max - all_min) / self.num_bins as f32;

        // Count distributions
        let mut ref_counts = vec![0usize; self.num_bins];
        let mut target_counts = vec![0usize; self.num_bins];

        for &v in reference {
            let bin = ((v - all_min) / bin_width).floor() as usize;
            let bin = bin.min(self.num_bins - 1);
            ref_counts[bin] += 1;
        }

        for &v in target {
            let bin = ((v - all_min) / bin_width).floor() as usize;
            let bin = bin.min(self.num_bins - 1);
            target_counts[bin] += 1;
        }

        // Convert to probability distributions and compute JSD inline (no allocations)
        let ref_total = reference.len() as f32;
        let target_total = target.len() as f32;

        // Compute JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5 * (P + Q)
        // Done in a single pass without intermediate vectors
        let mut kl_pm = 0.0f32;
        let mut kl_qm = 0.0f32;

        for i in 0..self.num_bins {
            let pi = (ref_counts[i] as f32 / ref_total).max(self.epsilon);
            let qi = (target_counts[i] as f32 / target_total).max(self.epsilon);
            let mi = 0.5 * (pi + qi);

            if pi > self.epsilon {
                kl_pm += pi * (pi / mi).ln();
            }
            if qi > self.epsilon {
                kl_qm += qi * (qi / mi).ln();
            }
        }

        0.5 * kl_pm + 0.5 * kl_qm
    }

    fn name(&self) -> &'static str {
        "jsd"
    }

    fn warning_threshold(&self) -> f32 {
        0.05
    }

    fn critical_threshold(&self) -> f32 {
        0.15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_identical() {
        let psi = PSI::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score = psi.compute(&data, &data);
        assert!(score < 0.01, "PSI for identical data should be ~0, got {}", score);
    }

    #[test]
    fn test_psi_shifted() {
        let psi = PSI::new(5);
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5];
        let target = vec![5.0, 6.0, 7.0, 8.0, 9.0, 5.5, 6.5, 7.5, 8.5];
        let score = psi.compute(&reference, &target);
        assert!(score > 0.1, "PSI for shifted data should be > 0.1, got {}", score);
    }

    #[test]
    fn test_psi_empty() {
        let psi = PSI::default();
        let empty: Vec<f32> = vec![];
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(psi.compute(&empty, &data), 0.0);
        assert_eq!(psi.compute(&data, &empty), 0.0);
    }

    #[test]
    fn test_ks_identical() {
        let ks = KolmogorovSmirnov::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score = ks.compute(&data, &data);
        assert!(score < 0.01, "KS for identical data should be ~0, got {}", score);
    }

    #[test]
    fn test_ks_shifted() {
        let ks = KolmogorovSmirnov::new();
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let target = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let score = ks.compute(&reference, &target);
        assert!((score - 1.0).abs() < 0.01, "KS for completely shifted data should be ~1, got {}", score);
    }

    #[test]
    fn test_ks_partial_overlap() {
        let ks = KolmogorovSmirnov::new();
        let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let target = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        let score = ks.compute(&reference, &target);
        assert!(score > 0.3 && score < 0.7, "KS for partial overlap should be moderate, got {}", score);
    }

    #[test]
    fn test_ks_empty_distributions() {
        let ks = KolmogorovSmirnov::new();
        let empty: Vec<f32> = vec![];
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(ks.compute(&empty, &data), 0.0);
        assert_eq!(ks.compute(&data, &empty), 0.0);
        assert_eq!(ks.compute(&empty, &empty), 0.0);
    }

    #[test]
    fn test_ks_single_element() {
        let ks = KolmogorovSmirnov::new();
        let single = vec![5.0];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let score = ks.compute(&single, &data);
        // Single element vs 5 elements: CDF jumps from 0 to 1 at value 5
        // At value 1: single_cdf=0, data_cdf=0.2 -> diff=0.2
        // At value 5: single_cdf=1, data_cdf=1 -> diff=0
        assert!(score > 0.0 && score <= 1.0, "KS with single element should be valid, got {}", score);
    }

    #[test]
    fn test_ks_duplicate_values() {
        let ks = KolmogorovSmirnov::new();
        let reference = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 2.0, 2.0, 3.0];
        let score = ks.compute(&reference, &target);
        // Similar distributions with different multiplicities
        assert!(score < 0.3, "KS for similar distributions should be low, got {}", score);
    }

    #[test]
    fn test_ks_completely_separated() {
        let ks = KolmogorovSmirnov::new();
        let reference = vec![1.0, 2.0, 3.0];
        let target = vec![10.0, 11.0, 12.0];
        let score = ks.compute(&reference, &target);
        assert!((score - 1.0).abs() < 0.01, "KS for completely separated should be 1.0, got {}", score);
    }

    #[test]
    fn test_ks_unequal_sizes() {
        let ks = KolmogorovSmirnov::new();
        let small = vec![1.0, 5.0, 9.0];
        let large = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let score = ks.compute(&small, &large);
        // Both cover same range, but different densities
        assert!(score < 0.5, "KS for same-range different sizes should be moderate, got {}", score);
    }

    #[test]
    fn test_ks_symmetry() {
        let ks = KolmogorovSmirnov::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let ab = ks.compute(&a, &b);
        let ba = ks.compute(&b, &a);
        assert!((ab - ba).abs() < 0.01, "KS should be symmetric: {} vs {}", ab, ba);
    }

    #[test]
    fn test_jsd_identical() {
        let jsd = JensenShannon::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5];
        let score = jsd.compute(&data, &data);
        assert!(score < 0.01, "JSD for identical data should be ~0, got {}", score);
    }

    #[test]
    fn test_jsd_different() {
        let jsd = JensenShannon::new(5);
        let reference = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let target = vec![7.0, 7.5, 8.0, 8.5, 9.0];
        let score = jsd.compute(&reference, &target);
        assert!(score > 0.1, "JSD for different data should be > 0.1, got {}", score);
    }

    #[test]
    fn test_jsd_bounded() {
        let jsd = JensenShannon::default();
        let reference = vec![1.0; 100];
        let target = vec![100.0; 100];
        let score = jsd.compute(&reference, &target);
        // JSD is bounded by ln(2) ≈ 0.693
        assert!(score <= 0.7, "JSD should be bounded by ln(2), got {}", score);
    }

    #[test]
    fn test_jsd_symmetric() {
        let jsd = JensenShannon::default();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let ab = jsd.compute(&a, &b);
        let ba = jsd.compute(&b, &a);
        assert!((ab - ba).abs() < 0.01, "JSD should be symmetric: {} vs {}", ab, ba);
    }

    #[test]
    fn test_metric_thresholds() {
        let psi = PSI::default();
        assert!((psi.warning_threshold() - 0.1).abs() < 0.01);
        assert!((psi.critical_threshold() - 0.25).abs() < 0.01);

        let ks = KolmogorovSmirnov::new();
        assert!((ks.warning_threshold() - 0.1).abs() < 0.01);
        assert!((ks.critical_threshold() - 0.2).abs() < 0.01);

        let jsd = JensenShannon::default();
        assert!((jsd.warning_threshold() - 0.05).abs() < 0.01);
        assert!((jsd.critical_threshold() - 0.15).abs() < 0.01);
    }
}
