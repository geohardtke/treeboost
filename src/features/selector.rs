//! Feature selection to prevent combinatorial explosion
//!
//! Filters generated features based on variance, correlation, and target importance.

use crate::defaults::feature_selection as feature_selection_defaults;

/// Configuration for feature selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Minimum variance threshold (filter near-constant features)
    pub min_variance: f32,
    /// Maximum correlation with existing features (filter redundant)
    pub max_correlation: f32,
    /// Maximum total features to keep
    pub max_features: usize,
    /// Whether to use target-based selection
    pub use_target_selection: bool,
    /// Whether to drop collinear features (highly correlated pairs)
    pub drop_collinear: bool,
    /// Correlation threshold for collinearity detection (default: 0.99)
    pub collinearity_threshold: f32,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            min_variance: feature_selection_defaults::DEFAULT_MIN_VARIANCE,
            max_correlation: feature_selection_defaults::DEFAULT_MAX_CORRELATION,
            max_features: feature_selection_defaults::DEFAULT_MAX_FEATURES,
            use_target_selection: feature_selection_defaults::DEFAULT_USE_TARGET_SELECTION,
            drop_collinear: feature_selection_defaults::DEFAULT_DROP_COLLINEAR,
            collinearity_threshold: feature_selection_defaults::DEFAULT_COLLINEARITY_THRESHOLD,
        }
    }
}

impl SelectionConfig {
    /// Create a new selection config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum variance threshold
    pub fn with_min_variance(mut self, min: f32) -> Self {
        self.min_variance = min;
        self
    }

    /// Set maximum correlation threshold
    pub fn with_max_correlation(mut self, max: f32) -> Self {
        self.max_correlation = max;
        self
    }

    /// Set maximum features to keep
    pub fn with_max_features(mut self, max: usize) -> Self {
        self.max_features = max;
        self
    }

    /// Enable or disable target-based selection
    pub fn with_target_selection(mut self, enabled: bool) -> Self {
        self.use_target_selection = enabled;
        self
    }

    /// Enable or disable collinear feature dropping
    ///
    /// When enabled, features that are highly correlated with each other
    /// will be filtered, keeping only the one with higher variance or
    /// higher target correlation.
    pub fn with_drop_collinear(mut self, enabled: bool) -> Self {
        self.drop_collinear = enabled;
        self
    }

    /// Set the collinearity threshold
    ///
    /// Feature pairs with correlation above this threshold are considered
    /// collinear. One of the pair will be dropped based on variance or
    /// target correlation. Default: 0.99
    pub fn with_collinearity_threshold(mut self, threshold: f32) -> Self {
        self.collinearity_threshold = threshold;
        self
    }
}

/// Feature selector to prevent combinatorial explosion
///
/// Filters generated features using multiple criteria:
/// 1. Variance filter - remove near-constant features
/// 2. Correlation filter - remove features highly correlated with originals
/// 3. Target selection - keep features most correlated with target
pub struct FeatureSelector {
    config: SelectionConfig,
}

impl FeatureSelector {
    /// Create a new feature selector
    pub fn new(config: SelectionConfig) -> Self {
        Self { config }
    }

    /// Select best features from candidates
    ///
    /// # Arguments
    /// * `original_data` - Original feature data (row-major)
    /// * `original_num_features` - Number of original features
    /// * `candidate_data` - Candidate generated feature data (row-major)
    /// * `candidate_num_features` - Number of candidate features
    /// * `candidate_names` - Names of candidate features
    /// * `targets` - Target values (for target-based selection)
    ///
    /// # Returns
    /// Tuple of (selected_data, selected_names, selected_indices)
    pub fn select(
        &self,
        original_data: &[f32],
        original_num_features: usize,
        candidate_data: &[f32],
        candidate_num_features: usize,
        candidate_names: &[String],
        targets: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<String>, Vec<usize>) {
        if candidate_num_features == 0 || candidate_data.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let num_rows = candidate_data.len() / candidate_num_features;

        // Stage 1: Variance filter
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        for f in 0..candidate_num_features {
            let variance = compute_variance(candidate_data, num_rows, candidate_num_features, f);

            if variance >= self.config.min_variance {
                candidates.push((f, variance));
            }
        }

        // Stage 2: Correlation filter (with original features)
        if original_num_features > 0 && !original_data.is_empty() {
            candidates.retain(|(f, _)| {
                let max_corr = compute_max_correlation_with_originals(
                    original_data,
                    original_num_features,
                    candidate_data,
                    candidate_num_features,
                    num_rows,
                    *f,
                );
                max_corr < self.config.max_correlation
            });
        }

        // Stage 3: Target-based selection (if targets provided)
        if self.config.use_target_selection {
            if let Some(targets) = targets {
                if targets.len() == num_rows {
                    // Compute target correlations
                    let mut scored: Vec<(usize, f32)> = candidates
                        .iter()
                        .map(|&(f, _)| {
                            let corr = compute_target_correlation(
                                candidate_data,
                                candidate_num_features,
                                num_rows,
                                f,
                                targets,
                            );
                            (f, corr.abs())
                        })
                        .collect();

                    // Sort by target correlation (descending)
                    scored
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    candidates = scored;
                }
            }
        }

        // Stage 4: Limit to max_features
        if candidates.len() > self.config.max_features {
            candidates.truncate(self.config.max_features);
        }

        // Extract selected features
        let selected_indices: Vec<usize> = candidates.iter().map(|(f, _)| *f).collect();

        let selected_names: Vec<String> = selected_indices
            .iter()
            .filter_map(|&f| candidate_names.get(f).cloned())
            .collect();

        let n_selected = selected_indices.len();
        let mut selected_data = vec![0.0f32; num_rows * n_selected];

        for (new_idx, &orig_idx) in selected_indices.iter().enumerate() {
            for r in 0..num_rows {
                selected_data[r * n_selected + new_idx] =
                    candidate_data[r * candidate_num_features + orig_idx];
            }
        }

        (selected_data, selected_names, selected_indices)
    }

    /// Drop collinear features from the dataset
    ///
    /// Identifies highly correlated feature pairs and drops one from each pair.
    /// When target values are provided, the feature with higher target correlation
    /// is kept. Otherwise, the feature with higher variance is kept.
    ///
    /// # Arguments
    /// * `data` - Feature data (row-major)
    /// * `num_features` - Number of features
    /// * `feature_names` - Names of features
    /// * `targets` - Optional target values for deciding which feature to keep
    ///
    /// # Returns
    /// Tuple of (filtered_data, filtered_names, kept_indices)
    pub fn drop_collinear_features(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
        targets: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<String>, Vec<usize>) {
        if num_features <= 1 || data.is_empty() {
            return (
                data.to_vec(),
                feature_names.to_vec(),
                (0..num_features).collect(),
            );
        }

        let num_rows = data.len() / num_features;
        let threshold = self.config.collinearity_threshold;

        // Compute feature statistics (mean, std, variance)
        let stats: Vec<(f32, f32, f32)> = (0..num_features)
            .map(|f| compute_feature_stats(data, num_rows, num_features, f))
            .collect();

        // Compute target correlations if targets provided
        let target_corrs: Option<Vec<f32>> = targets.map(|t| {
            (0..num_features)
                .map(|f| compute_target_correlation(data, num_features, num_rows, f, t).abs())
                .collect()
        });

        // Track which features to drop
        let mut dropped = vec![false; num_features];

        // Greedy collinearity filter: iterate over all pairs
        for i in 0..num_features {
            if dropped[i] {
                continue;
            }
            for j in (i + 1)..num_features {
                if dropped[j] {
                    continue;
                }

                // Compute correlation between features i and j
                let corr = compute_pairwise_correlation(
                    data,
                    num_features,
                    num_rows,
                    i,
                    j,
                    &stats[i],
                    &stats[j],
                );

                if corr.abs() > threshold {
                    // Decide which feature to drop based on target correlation or variance
                    let drop_j = match &target_corrs {
                        Some(tc) => tc[i] >= tc[j],       // Keep feature with higher target corr
                        None => stats[i].2 >= stats[j].2, // Keep feature with higher variance
                    };

                    if drop_j {
                        dropped[j] = true;
                    } else {
                        dropped[i] = true;
                        break; // Feature i dropped, stop checking its pairs
                    }
                }
            }
        }

        // Extract non-dropped features
        let kept_indices: Vec<usize> = (0..num_features).filter(|&f| !dropped[f]).collect();

        let kept_names: Vec<String> = kept_indices
            .iter()
            .filter_map(|&f| feature_names.get(f).cloned())
            .collect();

        let n_kept = kept_indices.len();
        let mut kept_data = vec![0.0f32; num_rows * n_kept];

        for (new_idx, &orig_idx) in kept_indices.iter().enumerate() {
            for r in 0..num_rows {
                kept_data[r * n_kept + new_idx] = data[r * num_features + orig_idx];
            }
        }

        (kept_data, kept_names, kept_indices)
    }
}

/// Compute mean, std, and variance of a feature column
fn compute_feature_stats(
    data: &[f32],
    num_rows: usize,
    num_features: usize,
    feature_idx: usize,
) -> (f32, f32, f32) {
    if num_rows == 0 {
        return (0.0, 1.0, 0.0);
    }

    let mean: f32 = (0..num_rows)
        .map(|r| data[r * num_features + feature_idx])
        .filter(|v| v.is_finite())
        .sum::<f32>()
        / num_rows as f32;

    let variance: f32 = (0..num_rows)
        .map(|r| {
            let v = data[r * num_features + feature_idx];
            if v.is_finite() {
                let diff = v - mean;
                diff * diff
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / num_rows as f32;

    let std = variance.sqrt().max(1e-10);

    (mean, std, variance)
}

/// Compute Pearson correlation between two feature columns
fn compute_pairwise_correlation(
    data: &[f32],
    num_features: usize,
    num_rows: usize,
    idx_a: usize,
    idx_b: usize,
    stats_a: &(f32, f32, f32),
    stats_b: &(f32, f32, f32),
) -> f32 {
    let (mean_a, std_a, _) = stats_a;
    let (mean_b, std_b, _) = stats_b;

    let covar: f32 = (0..num_rows)
        .map(|r| {
            let a = data[r * num_features + idx_a];
            let b = data[r * num_features + idx_b];
            if a.is_finite() && b.is_finite() {
                (a - mean_a) * (b - mean_b)
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / num_rows as f32;

    covar / (std_a * std_b)
}

/// Compute variance of a feature column
fn compute_variance(data: &[f32], num_rows: usize, num_features: usize, feature_idx: usize) -> f32 {
    if num_rows == 0 {
        return 0.0;
    }

    let mean: f32 = (0..num_rows)
        .map(|r| data[r * num_features + feature_idx])
        .sum::<f32>()
        / num_rows as f32;

    let variance: f32 = (0..num_rows)
        .map(|r| {
            let diff = data[r * num_features + feature_idx] - mean;
            diff * diff
        })
        .sum::<f32>()
        / num_rows as f32;

    variance
}

/// Compute maximum correlation between a candidate feature and any original feature
fn compute_max_correlation_with_originals(
    original_data: &[f32],
    original_num_features: usize,
    candidate_data: &[f32],
    candidate_num_features: usize,
    num_rows: usize,
    candidate_idx: usize,
) -> f32 {
    let mut max_corr = 0.0f32;

    // Get candidate column stats
    let cand_mean: f32 = (0..num_rows)
        .map(|r| candidate_data[r * candidate_num_features + candidate_idx])
        .sum::<f32>()
        / num_rows as f32;

    let cand_std: f32 = (0..num_rows)
        .map(|r| {
            let diff = candidate_data[r * candidate_num_features + candidate_idx] - cand_mean;
            diff * diff
        })
        .sum::<f32>()
        / num_rows as f32;
    let cand_std = cand_std.sqrt().max(1e-10);

    for orig_idx in 0..original_num_features {
        // Get original column stats
        let orig_mean: f32 = (0..num_rows)
            .map(|r| original_data[r * original_num_features + orig_idx])
            .sum::<f32>()
            / num_rows as f32;

        let orig_std: f32 = (0..num_rows)
            .map(|r| {
                let diff = original_data[r * original_num_features + orig_idx] - orig_mean;
                diff * diff
            })
            .sum::<f32>()
            / num_rows as f32;
        let orig_std = orig_std.sqrt().max(1e-10);

        // Compute correlation
        let covar: f32 = (0..num_rows)
            .map(|r| {
                let x = candidate_data[r * candidate_num_features + candidate_idx] - cand_mean;
                let y = original_data[r * original_num_features + orig_idx] - orig_mean;
                x * y
            })
            .sum::<f32>()
            / num_rows as f32;

        let corr = (covar / (cand_std * orig_std)).abs();
        max_corr = max_corr.max(corr);
    }

    max_corr
}

/// Compute correlation between a candidate feature and the target
fn compute_target_correlation(
    candidate_data: &[f32],
    candidate_num_features: usize,
    num_rows: usize,
    candidate_idx: usize,
    targets: &[f32],
) -> f32 {
    if num_rows == 0 {
        return 0.0;
    }

    let cand_mean: f32 = (0..num_rows)
        .map(|r| candidate_data[r * candidate_num_features + candidate_idx])
        .sum::<f32>()
        / num_rows as f32;

    let target_mean: f32 = targets.iter().sum::<f32>() / num_rows as f32;

    let cand_std: f32 = (0..num_rows)
        .map(|r| {
            let diff = candidate_data[r * candidate_num_features + candidate_idx] - cand_mean;
            diff * diff
        })
        .sum::<f32>()
        / num_rows as f32;
    let cand_std = cand_std.sqrt().max(1e-10);

    let target_std: f32 = targets
        .iter()
        .map(|&t| {
            let diff = t - target_mean;
            diff * diff
        })
        .sum::<f32>()
        / num_rows as f32;
    let target_std = target_std.sqrt().max(1e-10);

    let covar: f32 = (0..num_rows)
        .map(|r| {
            let x = candidate_data[r * candidate_num_features + candidate_idx] - cand_mean;
            let y = targets[r] - target_mean;
            x * y
        })
        .sum::<f32>()
        / num_rows as f32;

    covar / (cand_std * target_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_config_default() {
        let config = SelectionConfig::default();
        assert!((config.min_variance - 0.01).abs() < 1e-6);
        assert!((config.max_correlation - 0.95).abs() < 1e-6);
        assert_eq!(config.max_features, 50);
        assert!(config.use_target_selection);
    }

    #[test]
    fn test_selection_config_builder() {
        let config = SelectionConfig::new()
            .with_min_variance(0.1)
            .with_max_correlation(0.9)
            .with_max_features(10)
            .with_target_selection(false);

        assert!((config.min_variance - 0.1).abs() < 1e-6);
        assert!((config.max_correlation - 0.9).abs() < 1e-6);
        assert_eq!(config.max_features, 10);
        assert!(!config.use_target_selection);
    }

    #[test]
    fn test_variance_filter() {
        let config = SelectionConfig::new()
            .with_min_variance(0.5)
            .with_max_features(100);
        let selector = FeatureSelector::new(config);

        // Feature 0: high variance (1,2,3,4) -> var = 1.25
        // Feature 1: low variance (1,1,1,1) -> var = 0
        let candidates = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0];
        let names = vec!["high_var".to_string(), "low_var".to_string()];

        let (_, selected_names, _) = selector.select(&[], 0, &candidates, 2, &names, None);

        // Only high variance feature should be selected
        assert_eq!(selected_names.len(), 1);
        assert_eq!(selected_names[0], "high_var");
    }

    #[test]
    fn test_max_features_limit() {
        let config = SelectionConfig::new()
            .with_min_variance(0.0)
            .with_max_features(2);
        let selector = FeatureSelector::new(config);

        // 4 features
        let candidates = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let names = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];

        let (_, selected_names, _) = selector.select(&[], 0, &candidates, 4, &names, None);

        // Should be limited to 2
        assert_eq!(selected_names.len(), 2);
    }

    #[test]
    fn test_target_selection() {
        let config = SelectionConfig::new()
            .with_min_variance(0.0)
            .with_max_features(1)
            .with_target_selection(true);
        let selector = FeatureSelector::new(config);

        // Feature 0: uncorrelated with target
        // Feature 1: perfectly correlated with target
        let candidates = vec![
            1.0, 1.0, // row 0
            1.0, 2.0, // row 1
            1.0, 3.0, // row 2
            1.0, 4.0, // row 3
        ];
        let names = vec!["uncorrelated".to_string(), "correlated".to_string()];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let (_, selected_names, _) =
            selector.select(&[], 0, &candidates, 2, &names, Some(&targets));

        // Correlated feature should be selected
        assert_eq!(selected_names.len(), 1);
        assert_eq!(selected_names[0], "correlated");
    }

    #[test]
    fn test_empty_input() {
        let selector = FeatureSelector::new(SelectionConfig::default());
        let (data, names, indices) = selector.select(&[], 0, &[], 0, &[], None);

        assert!(data.is_empty());
        assert!(names.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_variance_computation() {
        // 4 rows, 1 feature: [1, 2, 3, 4]
        // mean = 2.5, variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let var = compute_variance(&data, 4, 1, 0);
        assert!((var - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_collinearity_config_defaults() {
        let config = SelectionConfig::default();
        assert!(!config.drop_collinear);
        assert!((config.collinearity_threshold - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_collinearity_config_builder() {
        let config = SelectionConfig::new()
            .with_drop_collinear(true)
            .with_collinearity_threshold(0.95);

        assert!(config.drop_collinear);
        assert!((config.collinearity_threshold - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_drop_collinear_perfect_correlation() {
        // Feature 0: [1, 2, 3, 4]
        // Feature 1: [2, 4, 6, 8] (perfectly correlated with feature 0, just 2x)
        // Feature 2: [4, 3, 2, 1] (negatively correlated with feature 0)
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let data = vec![
            1.0, 2.0, 4.0, // row 0
            2.0, 4.0, 3.0, // row 1
            3.0, 6.0, 2.0, // row 2
            4.0, 8.0, 1.0, // row 3
        ];
        let names = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];

        let (_, kept_names, kept_indices) =
            selector.drop_collinear_features(&data, 3, &names, None);

        // Features 0 and 1 are perfectly correlated (corr = 1.0), one should be dropped
        // Features 0 and 2 are perfectly negatively correlated (corr = -1.0), one should be dropped
        // Result: at most 1 feature should remain (since all are collinear)
        assert!(kept_names.len() <= 2);
        assert_eq!(kept_names.len(), kept_indices.len());
    }

    #[test]
    fn test_drop_collinear_uncorrelated_features() {
        // Three uncorrelated features
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let data = vec![
            1.0, 5.0, 9.0, // row 0
            2.0, 3.0, 8.0, // row 1
            3.0, 8.0, 2.0, // row 2
            4.0, 1.0, 5.0, // row 3
        ];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let (_, kept_names, _) = selector.drop_collinear_features(&data, 3, &names, None);

        // No features should be dropped as none are highly correlated
        assert_eq!(kept_names.len(), 3);
    }

    #[test]
    fn test_drop_collinear_with_target() {
        // Feature 0: [1, 2, 3, 4] - perfectly correlated with target
        // Feature 1: [2, 4, 6, 8] - same correlation pattern (perfectly correlated with f0)
        // When target is provided, keep the one with higher target correlation
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let data = vec![
            1.0, 2.0, // row 0
            2.0, 4.0, // row 1
            3.0, 6.0, // row 2
            4.0, 8.0, // row 3
        ];
        let names = vec!["f0".to_string(), "f1".to_string()];
        let targets = vec![1.0, 2.0, 3.0, 4.0]; // Perfectly correlated with both

        let (_, kept_names, _) = selector.drop_collinear_features(&data, 2, &names, Some(&targets));

        // One feature should be dropped (both have same target correlation, so first wins)
        assert_eq!(kept_names.len(), 1);
        assert_eq!(kept_names[0], "f0");
    }

    #[test]
    fn test_drop_collinear_keeps_higher_variance() {
        // Feature 0: [1, 2, 3, 4] - variance = 1.25
        // Feature 1: [10, 20, 30, 40] - variance = 125 (same pattern, higher scale)
        // Without target, should keep the one with higher variance
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let data = vec![
            1.0, 10.0, // row 0
            2.0, 20.0, // row 1
            3.0, 30.0, // row 2
            4.0, 40.0, // row 3
        ];
        let names = vec!["low_var".to_string(), "high_var".to_string()];

        let (_, kept_names, _) = selector.drop_collinear_features(&data, 2, &names, None);

        // Should keep the higher variance feature
        assert_eq!(kept_names.len(), 1);
        assert_eq!(kept_names[0], "high_var");
    }

    #[test]
    fn test_drop_collinear_single_feature() {
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let names = vec!["only_one".to_string()];

        let (kept_data, kept_names, kept_indices) =
            selector.drop_collinear_features(&data, 1, &names, None);

        // Single feature should remain unchanged
        assert_eq!(kept_data, data);
        assert_eq!(kept_names, names);
        assert_eq!(kept_indices, vec![0]);
    }

    #[test]
    fn test_drop_collinear_empty_input() {
        let config = SelectionConfig::new().with_collinearity_threshold(0.9);
        let selector = FeatureSelector::new(config);

        let (kept_data, kept_names, kept_indices) =
            selector.drop_collinear_features(&[], 0, &[], None);

        assert!(kept_data.is_empty());
        assert!(kept_names.is_empty());
        assert!(kept_indices.is_empty());
    }

    #[test]
    fn test_pairwise_correlation() {
        // Feature 0: [1, 2, 3, 4], mean=2.5, std=sqrt(1.25)
        // Feature 1: [2, 4, 6, 8], mean=5.0, std=sqrt(5.0)
        // Covariance = ((1-2.5)(2-5) + (2-2.5)(4-5) + (3-2.5)(6-5) + (4-2.5)(8-5)) / 4
        //            = ((-1.5)(-3) + (-0.5)(-1) + (0.5)(1) + (1.5)(3)) / 4
        //            = (4.5 + 0.5 + 0.5 + 4.5) / 4 = 2.5
        // Correlation = 2.5 / (sqrt(1.25) * sqrt(5.0)) = 2.5 / 2.5 = 1.0
        let data = vec![
            1.0, 2.0, // row 0
            2.0, 4.0, // row 1
            3.0, 6.0, // row 2
            4.0, 8.0, // row 3
        ];

        let stats_0 = compute_feature_stats(&data, 4, 2, 0);
        let stats_1 = compute_feature_stats(&data, 4, 2, 1);
        let corr = compute_pairwise_correlation(&data, 2, 4, 0, 1, &stats_0, &stats_1);

        assert!((corr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_feature_stats() {
        // Feature: [1, 2, 3, 4], mean=2.5, var=1.25, std=sqrt(1.25)
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, std, var) = compute_feature_stats(&data, 4, 1, 0);

        assert!((mean - 2.5).abs() < 1e-6);
        assert!((var - 1.25).abs() < 1e-6);
        assert!((std - 1.25f32.sqrt()).abs() < 1e-6);
    }
}
