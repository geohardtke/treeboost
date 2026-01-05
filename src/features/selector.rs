//! Feature selection to prevent combinatorial explosion
//!
//! Filters generated features based on variance, correlation, and target importance.

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
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            min_variance: 0.01,
            max_correlation: 0.95,
            max_features: 50,
            use_target_selection: true,
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
                        let corr =
                            compute_target_correlation(candidate_data, candidate_num_features, num_rows, f, targets);
                        (f, corr.abs())
                    })
                    .collect();

                // Sort by target correlation (descending)
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
                selected_data[r * n_selected + new_idx] = candidate_data[r * candidate_num_features + orig_idx];
            }
        }

        (selected_data, selected_names, selected_indices)
    }
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
        let config = SelectionConfig::new().with_min_variance(0.5).with_max_features(100);
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
        let config = SelectionConfig::new().with_min_variance(0.0).with_max_features(2);
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

        let (_, selected_names, _) = selector.select(&[], 0, &candidates, 2, &names, Some(&targets));

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
}
