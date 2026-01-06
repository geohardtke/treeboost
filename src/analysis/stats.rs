//! Statistical utilities for dataset analysis
//!
//! Fast, numerically stable implementations of common statistics.

/// Compute R² (coefficient of determination)
///
/// R² = 1 - SS_res / SS_tot
///
/// Where:
/// - SS_res = Σ(y_true - y_pred)²
/// - SS_tot = Σ(y_true - y_mean)²
///
/// Returns value in [0, 1] for reasonable predictions.
/// Can be negative if predictions are worse than mean.
#[inline]
pub fn compute_r2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return 0.0;
    }

    let n = y_true.len() as f32;
    let y_mean = y_true.iter().sum::<f32>() / n;

    let ss_tot: f32 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
    let ss_res: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();

    if ss_tot < 1e-10 {
        // Target is constant - R² undefined, return 1.0 (perfect fit)
        return 1.0;
    }

    // Clamp to [0, 1] for interpretability (negative means worse than mean)
    (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
}

/// Compute Pearson correlation coefficient
///
/// r = Σ[(x - x̄)(y - ȳ)] / √[Σ(x - x̄)² × Σ(y - ȳ)²]
///
/// Returns value in [-1, 1].
#[inline]
pub fn compute_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f32;
    let x_mean = x.iter().sum::<f32>() / n;
    let y_mean = y.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (cov / denom).clamp(-1.0, 1.0)
}

/// Compute variance
#[inline]
pub fn compute_variance(x: &[f32]) -> f32 {
    if x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / (n - 1.0)
}

/// Compute mean
#[inline]
pub fn compute_mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

/// Compute standard deviation
#[inline]
pub fn compute_std(x: &[f32]) -> f32 {
    compute_variance(x).sqrt()
}

/// Compute range (max - min)
#[inline]
pub fn compute_range(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max - min
}

/// Compute Pearson correlation with mixed types (f64 feature, f32 target)
///
/// This is useful for profiling where numeric columns are often f64 but
/// targets are converted to f32.
#[inline]
pub fn compute_correlation_mixed(x: &[f64], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().map(|&yi| yi as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let yi_f64 = yi as f64;
        let dx = xi - x_mean;
        let dy = yi_f64 - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    ((cov / denom) as f32).clamp(-1.0, 1.0)
}

/// Compute Mean Squared Error
#[inline]
pub fn compute_mse(y_true: &[f32], y_pred: &[f32]) -> f32 {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return f32::MAX;
    }

    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f32>()
        / y_true.len() as f32
}

/// Compute residuals
#[inline]
pub fn compute_residuals(y_true: &[f32], y_pred: &[f32]) -> Vec<f32> {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect()
}

/// Estimate noise floor using local variance
///
/// Splits data into bins and computes average within-bin variance.
/// This estimates irreducible error (noise that no model can predict).
///
/// High noise floor → RandomForest (variance reduction helps)
/// Low noise floor → Can fit tighter with GBDT
///
/// `best_feature_idx`: Index of the feature most correlated with target (for meaningful binning)
pub fn estimate_noise_floor(
    features: &[f32],
    targets: &[f32],
    num_features: usize,
    best_feature_idx: usize,
) -> f32 {
    if features.is_empty() || targets.is_empty() || num_features == 0 {
        return 0.0;
    }

    let num_rows = targets.len();
    let num_bins = 20.min(num_rows / 10).max(2); // At least 2 bins, at most 20

    // Use the most correlated feature for binning (passed from analyzer)
    let feature_idx = best_feature_idx.min(num_features - 1);

    let mut feature_target: Vec<(f32, f32)> = features
        .chunks(num_features)
        .zip(targets.iter())
        .map(|(row, &t)| (row.get(feature_idx).copied().unwrap_or(0.0), t))
        .collect();

    // Sort by feature value
    feature_target.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Compute within-bin variance
    let bin_size = (num_rows / num_bins).max(2);
    let mut total_variance = 0.0f32;
    let mut num_valid_bins = 0;

    for bin in feature_target.chunks(bin_size) {
        if bin.len() < 2 {
            continue;
        }

        let bin_targets: Vec<f32> = bin.iter().map(|(_, t)| *t).collect();
        let bin_var = compute_variance(&bin_targets);

        if bin_var.is_finite() {
            total_variance += bin_var;
            num_valid_bins += 1;
        }
    }

    if num_valid_bins == 0 {
        return 0.0;
    }

    let avg_local_var = total_variance / num_valid_bins as f32;
    let total_var = compute_variance(targets);

    if total_var < 1e-10 {
        return 0.0;
    }

    // Noise ratio = local variance / total variance
    // High ratio means most variance is noise (unpredictable)
    (avg_local_var / total_var).clamp(0.0, 1.0)
}

/// Compute monotonicity score for a feature-target relationship
///
/// Returns value in [0, 1]:
/// - 1.0 = Perfectly monotonic (always increasing or always decreasing)
/// - 0.5 = Random (no monotonic relationship)
/// - 0.0 = Would mean inverse monotonicity at every step (rare)
///
/// Based on Spearman's rank correlation converted to [0, 1].
pub fn compute_monotonicity(feature: &[f32], target: &[f32]) -> f32 {
    if feature.len() != target.len() || feature.len() < 3 {
        return 0.5; // Unknown
    }

    // Simple approach: count concordant vs discordant pairs (Kendall-like)
    // For efficiency, we sample if dataset is large
    let n = feature.len();
    let sample_size = n.min(1000);
    let step = n / sample_size;

    let mut concordant = 0i64;
    let mut discordant = 0i64;

    for i in (0..n).step_by(step.max(1)) {
        for j in (i + step..n).step_by(step.max(1)) {
            let feature_diff = feature[j] - feature[i];
            let target_diff = target[j] - target[i];

            if feature_diff.abs() < 1e-10 || target_diff.abs() < 1e-10 {
                continue; // Skip ties
            }

            if (feature_diff > 0.0) == (target_diff > 0.0) {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 0.5;
    }

    // Convert to [0, 1] where 1.0 = perfectly monotonic (either direction)
    let tau = (concordant - discordant) as f32 / total as f32;
    tau.abs() // We care about strength, not direction
}

/// Detect if target appears to have high-cardinality discrete values
/// (might indicate classification disguised as regression)
pub fn detect_discrete_target(targets: &[f32]) -> (bool, usize) {
    let mut unique: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for &t in targets.iter().take(10000) {
        // Quantize to detect discrete values
        unique.insert((t * 1000.0) as u32);

        // If more than 100 unique values in first 10k, likely continuous
        if unique.len() > 100 {
            return (false, unique.len());
        }
    }

    (unique.len() <= 20, unique.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r2_perfect_prediction() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((compute_r2(&y_true, &y_pred) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r2_mean_prediction() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![3.0, 3.0, 3.0, 3.0, 3.0]; // Mean
        assert!(compute_r2(&y_true, &y_pred) < 0.01);
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((compute_correlation(&x, &y) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((compute_correlation(&x, &y) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_monotonicity_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(compute_monotonicity(&x, &y) > 0.9);
    }

    #[test]
    fn test_variance() {
        let x = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = compute_variance(&x);
        assert!((var - 4.571429).abs() < 0.01); // Known variance
    }
}
