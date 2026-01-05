//! Ratio feature generator
//!
//! Generates ratio features (x_i / x_j) for pairs of input features.

use super::FeatureGenerator;

/// Ratio feature generator
///
/// Generates ratio features: x_i / (x_j + epsilon) for pairs of features.
///
/// Pairs can be:
/// - Explicitly specified
/// - Auto-selected based on correlation
/// - All pairs (combinatorial)
///
/// # Example
///
/// ```ignore
/// // Generate ratios for specific pairs
/// let ratio = RatioGenerator::from_pairs(vec![(0, 1), (1, 2)]);
///
/// // Or auto-select based on correlation
/// let ratio = RatioGenerator::auto_select(&data, num_features, 3);
/// ```
#[derive(Debug, Clone)]
pub struct RatioGenerator {
    /// Pairs of (numerator_idx, denominator_idx)
    pairs: Vec<(usize, usize)>,
    /// Small value to avoid division by zero
    epsilon: f32,
}

impl RatioGenerator {
    /// Create a generator with explicit pairs
    pub fn from_pairs(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            pairs,
            epsilon: 1e-10,
        }
    }

    /// Create a generator for all pairs (combinatorial)
    ///
    /// Warning: This generates O(n²) features which can be expensive.
    pub fn all_pairs(num_features: usize) -> Self {
        let mut pairs = Vec::new();
        for i in 0..num_features {
            for j in 0..num_features {
                if i != j {
                    pairs.push((i, j));
                }
            }
        }
        Self {
            pairs,
            epsilon: 1e-10,
        }
    }

    /// Auto-select pairs based on correlation
    ///
    /// Selects the top-k most correlated pairs for each feature.
    pub fn auto_select(
        data: &[f32],
        num_features: usize,
        max_per_feature: usize,
    ) -> Self {
        if num_features == 0 || data.is_empty() || max_per_feature == 0 {
            return Self::from_pairs(Vec::new());
        }

        let num_rows = data.len() / num_features;
        if num_rows < 2 {
            return Self::from_pairs(Vec::new());
        }

        // Compute correlation matrix
        let correlations = compute_correlation_matrix(data, num_features, num_rows);

        // For each feature, select top-k most correlated (by absolute correlation)
        let mut pairs = Vec::new();
        for i in 0..num_features {
            let mut feature_correlations: Vec<(usize, f32)> = (0..num_features)
                .filter(|&j| j != i)
                .map(|j| (j, correlations[i * num_features + j].abs()))
                .collect();

            // Sort by absolute correlation (descending)
            feature_correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k
            for (j, _) in feature_correlations.into_iter().take(max_per_feature) {
                // Avoid duplicate pairs (i/j and j/i)
                if !pairs.contains(&(i, j)) && !pairs.contains(&(j, i)) {
                    pairs.push((i, j));
                }
            }
        }

        Self {
            pairs,
            epsilon: 1e-10,
        }
    }

    /// Set the epsilon value for division
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Get the number of ratio features that will be generated
    pub fn n_features(&self) -> usize {
        self.pairs.len()
    }

    /// Get the pairs
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }
}

impl FeatureGenerator for RatioGenerator {
    fn generate(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>) {
        if self.pairs.is_empty() || num_features == 0 || data.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_rows = data.len() / num_features;
        let n_new = self.pairs.len();

        let mut new_data = vec![0.0f32; num_rows * n_new];
        let mut new_names = Vec::with_capacity(n_new);

        for (idx, &(i, j)) in self.pairs.iter().enumerate() {
            if i >= num_features || j >= num_features {
                continue;
            }

            // Generate feature name
            let name_i = feature_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("f{}", i));
            let name_j = feature_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("f{}", j));
            new_names.push(format!("{}_div_{}", name_i, name_j));

            // Compute ratios
            for r in 0..num_rows {
                let numerator = data[r * num_features + i];
                let denominator = data[r * num_features + j];
                new_data[r * n_new + idx] = numerator / (denominator + self.epsilon);
            }
        }

        (new_data, new_names)
    }

    fn name(&self) -> &'static str {
        "ratio"
    }
}

/// Compute correlation matrix for feature columns
fn compute_correlation_matrix(data: &[f32], num_features: usize, num_rows: usize) -> Vec<f32> {
    let mut correlations = vec![0.0f32; num_features * num_features];

    // Compute means
    let means: Vec<f32> = (0..num_features)
        .map(|f| {
            let sum: f32 = (0..num_rows).map(|r| data[r * num_features + f]).sum();
            sum / num_rows as f32
        })
        .collect();

    // Compute standard deviations
    let stds: Vec<f32> = (0..num_features)
        .map(|f| {
            let var: f32 = (0..num_rows)
                .map(|r| {
                    let diff = data[r * num_features + f] - means[f];
                    diff * diff
                })
                .sum::<f32>()
                / num_rows as f32;
            var.sqrt().max(1e-10)
        })
        .collect();

    // Compute correlations
    for i in 0..num_features {
        for j in 0..num_features {
            if i == j {
                correlations[i * num_features + j] = 1.0;
            } else if j > i {
                // Only compute once (matrix is symmetric)
                let covar: f32 = (0..num_rows)
                    .map(|r| {
                        let xi = data[r * num_features + i] - means[i];
                        let xj = data[r * num_features + j] - means[j];
                        xi * xj
                    })
                    .sum::<f32>()
                    / num_rows as f32;

                let corr = covar / (stds[i] * stds[j]);
                correlations[i * num_features + j] = corr;
                correlations[j * num_features + i] = corr;
            }
        }
    }

    correlations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratio_from_pairs() {
        let ratio = RatioGenerator::from_pairs(vec![(0, 1), (1, 2)]);
        assert_eq!(ratio.n_features(), 2);
    }

    #[test]
    fn test_ratio_all_pairs() {
        let ratio = RatioGenerator::all_pairs(3);
        // 3 features = 6 pairs (3 * 2)
        assert_eq!(ratio.n_features(), 6);
    }

    #[test]
    fn test_ratio_generate() {
        let ratio = RatioGenerator::from_pairs(vec![(0, 1)]);

        // 2 rows, 2 features
        let data = vec![
            4.0, 2.0, // row 0: a=4, b=2
            6.0, 3.0, // row 1: a=6, b=3
        ];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = ratio.generate(&data, 2, &names);

        assert_eq!(new_names.len(), 1);
        assert_eq!(new_names[0], "a_div_b");

        assert_eq!(new_data.len(), 2);
        assert!((new_data[0] - 2.0).abs() < 1e-6); // 4 / 2 = 2
        assert!((new_data[1] - 2.0).abs() < 1e-6); // 6 / 3 = 2
    }

    #[test]
    fn test_ratio_division_by_zero() {
        let ratio = RatioGenerator::from_pairs(vec![(0, 1)]).with_epsilon(1e-10);

        let data = vec![1.0, 0.0]; // 1 row: a=1, b=0
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, _) = ratio.generate(&data, 2, &names);

        // Should not panic, should return large value
        assert!(new_data[0].is_finite());
        assert!(new_data[0] > 1e8); // Very large due to small epsilon
    }

    #[test]
    fn test_ratio_auto_select() {
        // Create data where features 0 and 1 are highly correlated
        let data = vec![
            1.0, 2.0, 10.0, // row 0
            2.0, 4.0, 20.0, // row 1
            3.0, 6.0, 30.0, // row 2
            4.0, 8.0, 40.0, // row 3
        ];

        let ratio = RatioGenerator::auto_select(&data, 3, 1);

        // Should select most correlated pairs
        assert!(ratio.n_features() > 0);
    }

    #[test]
    fn test_ratio_empty() {
        let ratio = RatioGenerator::from_pairs(vec![]);
        let (new_data, new_names) = ratio.generate(&[1.0, 2.0], 2, &["a".to_string(), "b".to_string()]);
        assert!(new_data.is_empty());
        assert!(new_names.is_empty());
    }

    #[test]
    fn test_correlation_matrix() {
        // Perfect positive correlation
        let data = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
        let corr = compute_correlation_matrix(&data, 2, 3);

        assert!((corr[0] - 1.0).abs() < 1e-6); // self-correlation
        assert!((corr[1] - 1.0).abs() < 1e-6); // perfect correlation
        assert!((corr[2] - 1.0).abs() < 1e-6); // symmetric
        assert!((corr[3] - 1.0).abs() < 1e-6); // self-correlation
    }
}
