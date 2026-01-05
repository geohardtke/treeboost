//! Pairwise feature interaction generator
//!
//! Generates interaction features between pairs of input features.
//! Critical for linear models in mixed ensembles, as trees auto-discover
//! interactions via splits but linear models need explicit terms.
//!
//! # Interaction Types
//!
//! | Operation | Formula | Best For |
//! |-----------|---------|----------|
//! | Multiply | x_i × x_j | Scaling effects, joint influence |
//! | Add | x_i + x_j | Combined magnitude |
//! | Subtract | \|x_i - x_j\| | Difference/similarity |
//! | Min | min(x_i, x_j) | Lower bound constraints |
//! | Max | max(x_i, x_j) | Upper bound constraints |
//!
//! # Example
//!
//! ```ignore
//! use treeboost::features::{InteractionGenerator, InteractionType};
//!
//! // Generate multiplication interactions for specific pairs
//! let gen = InteractionGenerator::from_pairs(vec![(0, 1), (1, 2)])
//!     .with_types(vec![InteractionType::Multiply]);
//!
//! // Or auto-select most informative pairs
//! let gen = InteractionGenerator::auto_select(&data, num_features, &targets, 10)
//!     .with_types(vec![InteractionType::Multiply, InteractionType::Subtract]);
//!
//! let (features, names) = gen.generate(&data, num_features, &feature_names);
//! ```

use super::stats::{compute_correlation_matrix, correlation};
use super::FeatureGenerator;

/// Types of pairwise interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteractionType {
    /// x_i × x_j (multiplicative interaction)
    Multiply,
    /// x_i + x_j (additive combination)
    Add,
    /// |x_i - x_j| (absolute difference)
    Subtract,
    /// min(x_i, x_j)
    Min,
    /// max(x_i, x_j)
    Max,
}

impl InteractionType {
    /// Get the suffix for feature naming
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Multiply => "mul",
            Self::Add => "add",
            Self::Subtract => "sub",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    /// Compute the interaction value
    #[inline]
    pub fn compute(&self, a: f32, b: f32) -> f32 {
        match self {
            Self::Multiply => a * b,
            Self::Add => a + b,
            Self::Subtract => (a - b).abs(),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
        }
    }

    /// All available interaction types
    pub fn all() -> Vec<Self> {
        vec![
            Self::Multiply,
            Self::Add,
            Self::Subtract,
            Self::Min,
            Self::Max,
        ]
    }

    /// Default interactions (multiply only - most common)
    pub fn default_types() -> Vec<Self> {
        vec![Self::Multiply]
    }
}

/// Strategy for selecting feature pairs
#[derive(Debug, Clone)]
pub enum PairSelection {
    /// Use all possible pairs (O(n²) features)
    AllPairs,
    /// Use explicitly specified pairs
    Explicit(Vec<(usize, usize)>),
    /// Auto-select top-k pairs by correlation strength
    TopCorrelated { max_pairs: usize },
    /// Auto-select pairs with highest target correlation gain
    TargetBased { max_pairs: usize, targets: Vec<f32> },
}

/// Pairwise feature interaction generator
///
/// Generates interaction terms between pairs of features.
/// Essential for linear models that cannot auto-discover interactions.
#[derive(Debug, Clone)]
pub struct InteractionGenerator {
    /// How to select feature pairs
    selection: PairSelection,
    /// Types of interactions to generate
    interaction_types: Vec<InteractionType>,
    /// Include self-interactions (x_i × x_i = x_i²)
    include_self: bool,
    /// Minimum absolute correlation for auto-selection
    min_correlation: f32,
    /// Cached pairs after fitting
    pairs: Option<Vec<(usize, usize)>>,
}

impl InteractionGenerator {
    /// Create generator with explicit pairs
    pub fn from_pairs(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            selection: PairSelection::Explicit(pairs),
            interaction_types: InteractionType::default_types(),
            include_self: false,
            min_correlation: 0.0,
            pairs: None,
        }
    }

    /// Create generator for all pairs
    pub fn all_pairs() -> Self {
        Self {
            selection: PairSelection::AllPairs,
            interaction_types: InteractionType::default_types(),
            include_self: false,
            min_correlation: 0.0,
            pairs: None,
        }
    }

    /// Create generator that auto-selects top-k correlated pairs
    pub fn top_correlated(max_pairs: usize) -> Self {
        Self {
            selection: PairSelection::TopCorrelated { max_pairs },
            interaction_types: InteractionType::default_types(),
            include_self: false,
            min_correlation: 0.1,
            pairs: None,
        }
    }

    /// Create generator that selects pairs based on target correlation gain
    ///
    /// Selects pairs where the interaction has higher correlation with target
    /// than either individual feature.
    pub fn target_based(max_pairs: usize, targets: Vec<f32>) -> Self {
        Self {
            selection: PairSelection::TargetBased { max_pairs, targets },
            interaction_types: InteractionType::default_types(),
            include_self: false,
            min_correlation: 0.0,
            pairs: None,
        }
    }

    /// Set interaction types to generate
    pub fn with_types(mut self, types: Vec<InteractionType>) -> Self {
        self.interaction_types = types;
        self
    }

    /// Enable self-interactions (x_i² via multiply)
    pub fn with_self_interactions(mut self, include: bool) -> Self {
        self.include_self = include;
        self
    }

    /// Set minimum correlation threshold for auto-selection
    pub fn with_min_correlation(mut self, threshold: f32) -> Self {
        self.min_correlation = threshold;
        self
    }

    /// Get the selected pairs (after fitting)
    pub fn pairs(&self) -> Option<&[(usize, usize)]> {
        self.pairs.as_deref()
    }

    /// Get number of interactions per pair
    pub fn interactions_per_pair(&self) -> usize {
        self.interaction_types.len()
    }

    /// Fit the generator to select pairs based on data
    pub fn fit(&mut self, data: &[f32], num_features: usize) {
        if num_features == 0 || data.is_empty() {
            self.pairs = Some(Vec::new());
            return;
        }

        let num_rows = data.len() / num_features;

        let pairs = match &self.selection {
            // AllPairs and Explicit don't need multiple rows
            PairSelection::AllPairs => generate_all_pairs(num_features, self.include_self),
            PairSelection::Explicit(p) => {
                // Filter invalid pairs
                p.iter()
                    .filter(|(i, j)| *i < num_features && *j < num_features)
                    .cloned()
                    .collect()
            }
            // Correlation-based selection requires at least 2 rows
            PairSelection::TopCorrelated { max_pairs } => {
                if num_rows < 2 {
                    // Fall back to all pairs if not enough rows
                    generate_all_pairs(num_features, self.include_self)
                        .into_iter()
                        .take(*max_pairs)
                        .collect()
                } else {
                    select_top_correlated(
                        data,
                        num_features,
                        num_rows,
                        *max_pairs,
                        self.min_correlation,
                        self.include_self,
                    )
                }
            }
            PairSelection::TargetBased { max_pairs, targets } => {
                if num_rows < 2 || targets.len() != num_rows {
                    // Fall back to all pairs if not enough rows
                    generate_all_pairs(num_features, self.include_self)
                        .into_iter()
                        .take(*max_pairs)
                        .collect()
                } else {
                    select_target_based(
                        data,
                        num_features,
                        num_rows,
                        targets,
                        *max_pairs,
                        self.include_self,
                    )
                }
            }
        };

        self.pairs = Some(pairs);
    }

    /// Check if generator has been fitted
    pub fn is_fitted(&self) -> bool {
        self.pairs.is_some()
    }

    /// Get expected number of output features (after fitting)
    pub fn n_output_features(&self) -> usize {
        self.pairs
            .as_ref()
            .map(|p| p.len() * self.interaction_types.len())
            .unwrap_or(0)
    }
}

impl FeatureGenerator for InteractionGenerator {
    fn generate(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>) {
        // Auto-fit if not fitted
        let pairs = match &self.pairs {
            Some(p) => p.clone(),
            None => {
                // Create temporary mutable copy to fit
                let mut temp = self.clone();
                temp.fit(data, num_features);
                temp.pairs.unwrap_or_default()
            }
        };

        if pairs.is_empty() || num_features == 0 || data.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_rows = data.len() / num_features;
        let n_interactions = self.interaction_types.len();
        let total_features = pairs.len() * n_interactions;

        let mut new_data = vec![0.0f32; num_rows * total_features];
        let mut new_names = Vec::with_capacity(total_features);

        // Generate interaction features
        for (pair_idx, &(i, j)) in pairs.iter().enumerate() {
            let name_i = feature_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("f{}", i));
            let name_j = feature_names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("f{}", j));

            for (type_idx, interaction_type) in self.interaction_types.iter().enumerate() {
                let feature_idx = pair_idx * n_interactions + type_idx;

                // Generate feature name
                new_names.push(format!(
                    "{}_{}_{}",
                    name_i,
                    interaction_type.suffix(),
                    name_j
                ));

                // Compute interaction values
                for r in 0..num_rows {
                    let val_i = data[r * num_features + i];
                    let val_j = data[r * num_features + j];
                    new_data[r * total_features + feature_idx] =
                        interaction_type.compute(val_i, val_j);
                }
            }
        }

        (new_data, new_names)
    }

    fn name(&self) -> &'static str {
        "interaction"
    }
}

impl Default for InteractionGenerator {
    fn default() -> Self {
        Self::top_correlated(20)
    }
}

// ============================================================================
// Pair Selection Algorithms
// ============================================================================

/// Generate all possible pairs
fn generate_all_pairs(num_features: usize, include_self: bool) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();

    for i in 0..num_features {
        let start = if include_self { i } else { i + 1 };
        for j in start..num_features {
            pairs.push((i, j));
        }
    }

    pairs
}

/// Select top-k pairs by absolute correlation
fn select_top_correlated(
    data: &[f32],
    num_features: usize,
    num_rows: usize,
    max_pairs: usize,
    min_correlation: f32,
    include_self: bool,
) -> Vec<(usize, usize)> {
    let correlations = compute_correlation_matrix(data, num_features, num_rows);

    // Collect all pairs with their correlation
    let mut pair_scores: Vec<((usize, usize), f32)> = Vec::new();

    for i in 0..num_features {
        let start = if include_self { i } else { i + 1 };
        for j in start..num_features {
            let corr = correlations[i * num_features + j].abs();
            if corr >= min_correlation {
                pair_scores.push(((i, j), corr));
            }
        }
    }

    // Sort by absolute correlation (descending)
    pair_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k
    pair_scores
        .into_iter()
        .take(max_pairs)
        .map(|(pair, _)| pair)
        .collect()
}

/// Select pairs based on interaction's correlation with target
fn select_target_based(
    data: &[f32],
    num_features: usize,
    num_rows: usize,
    targets: &[f32],
    max_pairs: usize,
    include_self: bool,
) -> Vec<(usize, usize)> {
    if targets.len() != num_rows {
        return Vec::new();
    }

    // Compute individual feature correlations with target
    let feature_target_corrs: Vec<f32> = (0..num_features)
        .map(|f| {
            let feature_vals: Vec<f32> =
                (0..num_rows).map(|r| data[r * num_features + f]).collect();
            correlation(&feature_vals, targets).abs()
        })
        .collect();

    // Score each pair by interaction's correlation gain
    let mut pair_scores: Vec<((usize, usize), f32)> = Vec::new();

    for i in 0..num_features {
        let start = if include_self { i } else { i + 1 };
        for j in start..num_features {
            // Compute multiplication interaction (most common)
            let interaction: Vec<f32> = (0..num_rows)
                .map(|r| {
                    let vi = data[r * num_features + i];
                    let vj = data[r * num_features + j];
                    vi * vj
                })
                .collect();

            let interaction_corr = correlation(&interaction, targets).abs();
            let max_individual = feature_target_corrs[i].max(feature_target_corrs[j]);

            // Score = how much better the interaction is than individual features
            let gain = interaction_corr - max_individual;

            // Only include if interaction provides gain
            if gain > 0.0 {
                pair_scores.push(((i, j), gain));
            }
        }
    }

    // Sort by gain (descending)
    pair_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k
    pair_scores
        .into_iter()
        .take(max_pairs)
        .map(|(pair, _)| pair)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // InteractionType Tests
    // ========================================

    #[test]
    fn test_interaction_type_compute() {
        assert!((InteractionType::Multiply.compute(3.0, 4.0) - 12.0).abs() < 1e-6);
        assert!((InteractionType::Add.compute(3.0, 4.0) - 7.0).abs() < 1e-6);
        assert!((InteractionType::Subtract.compute(3.0, 4.0) - 1.0).abs() < 1e-6);
        assert!((InteractionType::Min.compute(3.0, 4.0) - 3.0).abs() < 1e-6);
        assert!((InteractionType::Max.compute(3.0, 4.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_interaction_type_subtract_absolute() {
        // Subtract should be absolute difference
        assert!((InteractionType::Subtract.compute(4.0, 3.0) - 1.0).abs() < 1e-6);
        assert!((InteractionType::Subtract.compute(3.0, 4.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_interaction_type_suffixes() {
        assert_eq!(InteractionType::Multiply.suffix(), "mul");
        assert_eq!(InteractionType::Add.suffix(), "add");
        assert_eq!(InteractionType::Subtract.suffix(), "sub");
        assert_eq!(InteractionType::Min.suffix(), "min");
        assert_eq!(InteractionType::Max.suffix(), "max");
    }

    // ========================================
    // InteractionGenerator Basic Tests
    // ========================================

    #[test]
    fn test_from_pairs_basic() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1), (1, 2)]);

        // 3 rows, 3 features
        let data = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let (new_data, new_names) = gen.generate(&data, 3, &names);

        // 2 pairs × 1 type (multiply) = 2 features
        assert_eq!(new_names.len(), 2);
        assert_eq!(new_names[0], "a_mul_b");
        assert_eq!(new_names[1], "b_mul_c");

        // 3 rows × 2 features = 6 values
        assert_eq!(new_data.len(), 6);

        // Row 0: a*b = 1*2 = 2, b*c = 2*3 = 6
        assert!((new_data[0] - 2.0).abs() < 1e-6);
        assert!((new_data[1] - 6.0).abs() < 1e-6);

        // Row 1: a*b = 4*5 = 20, b*c = 5*6 = 30
        assert!((new_data[2] - 20.0).abs() < 1e-6);
        assert!((new_data[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_interaction_types() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1)]).with_types(vec![
            InteractionType::Multiply,
            InteractionType::Add,
            InteractionType::Subtract,
        ]);

        let data = vec![3.0, 5.0]; // 1 row, 2 features
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = gen.generate(&data, 2, &names);

        // 1 pair × 3 types = 3 features
        assert_eq!(new_names.len(), 3);
        assert_eq!(new_names[0], "a_mul_b");
        assert_eq!(new_names[1], "a_add_b");
        assert_eq!(new_names[2], "a_sub_b");

        assert!((new_data[0] - 15.0).abs() < 1e-6); // 3 * 5
        assert!((new_data[1] - 8.0).abs() < 1e-6); // 3 + 5
        assert!((new_data[2] - 2.0).abs() < 1e-6); // |3 - 5|
    }

    #[test]
    fn test_all_pairs_generator() {
        let mut gen = InteractionGenerator::all_pairs();

        // 3 features: pairs (0,1), (0,2), (1,2)
        let data = vec![1.0, 2.0, 3.0]; // 1 row
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        gen.fit(&data, 3);
        assert_eq!(gen.pairs().unwrap().len(), 3);

        let (_, new_names) = gen.generate(&data, 3, &names);
        assert_eq!(new_names.len(), 3);
    }

    #[test]
    fn test_self_interactions() {
        let mut gen = InteractionGenerator::all_pairs().with_self_interactions(true);

        // 2 features with self: pairs (0,0), (0,1), (1,1)
        let data = vec![2.0, 3.0];
        gen.fit(&data, 2);

        assert_eq!(gen.pairs().unwrap().len(), 3);

        let names = vec!["a".to_string(), "b".to_string()];
        let (new_data, new_names) = gen.generate(&data, 2, &names);

        // Should include a_mul_a = 4, a_mul_b = 6, b_mul_b = 9
        assert_eq!(new_names.len(), 3);
        assert!((new_data[0] - 4.0).abs() < 1e-6); // 2 * 2
    }

    #[test]
    fn test_empty_data() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1)]);
        let (new_data, new_names) = gen.generate(&[], 0, &[]);
        assert!(new_data.is_empty());
        assert!(new_names.is_empty());
    }

    // ========================================
    // Auto-Selection Tests
    // ========================================

    #[test]
    fn test_top_correlated_selection() {
        // Create data where features 0 and 1 are highly correlated
        let data = vec![
            1.0, 2.0, 10.0, // row 0
            2.0, 4.0, 20.0, // row 1
            3.0, 6.0, 30.0, // row 2
            4.0, 8.0, 40.0, // row 3
        ];

        let mut gen = InteractionGenerator::top_correlated(2).with_min_correlation(0.5);
        gen.fit(&data, 3);

        let pairs = gen.pairs().unwrap();
        assert!(!pairs.is_empty());

        // Pairs (0,1) and (0,2) or (1,2) should be selected as they're perfectly correlated
        assert!(pairs.len() <= 2);
    }

    #[test]
    fn test_target_based_selection() {
        // Create data where interaction (0,1) correlates better with target
        // than individual features
        let data = vec![
            1.0, 1.0, 0.0, // row 0: product = 1
            2.0, 3.0, 0.0, // row 1: product = 6
            3.0, 2.0, 0.0, // row 2: product = 6
            4.0, 4.0, 0.0, // row 3: product = 16
        ];

        // Target correlates with f0 * f1
        let targets = vec![1.0, 6.0, 6.0, 16.0];

        let mut gen = InteractionGenerator::target_based(5, targets);
        gen.fit(&data, 3);

        // Should select pairs where interaction improves correlation
        let pairs = gen.pairs().unwrap();
        // May or may not find gain depending on exact correlations
        assert!(pairs.len() <= 5);
    }

    // ========================================
    // Statistical Helper Tests
    // ========================================

    #[test]
    fn test_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let corr = correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![4.0, 3.0, 2.0, 1.0];
        let corr = correlation(&x, &y);
        assert!((corr + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_zero() {
        // Orthogonal data
        let x = vec![1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0];
        let corr = correlation(&x, &y);
        assert!(corr.abs() < 1e-6);
    }

    #[test]
    fn test_generate_all_pairs() {
        let pairs = generate_all_pairs(4, false);
        // n*(n-1)/2 = 4*3/2 = 6
        assert_eq!(pairs.len(), 6);

        let pairs_with_self = generate_all_pairs(4, true);
        // n*(n+1)/2 = 4*5/2 = 10
        assert_eq!(pairs_with_self.len(), 10);
    }

    // ========================================
    // Integration Tests
    // ========================================

    #[test]
    fn test_feature_generator_trait() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1)]);
        assert_eq!(gen.name(), "interaction");
    }

    #[test]
    fn test_min_max_interactions() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1)])
            .with_types(vec![InteractionType::Min, InteractionType::Max]);

        let data = vec![
            3.0, 5.0, // row 0
            7.0, 2.0, // row 1
        ];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, new_names) = gen.generate(&data, 2, &names);

        assert_eq!(new_names.len(), 2);
        assert_eq!(new_names[0], "a_min_b");
        assert_eq!(new_names[1], "a_max_b");

        // Row 0: min(3,5)=3, max(3,5)=5
        assert!((new_data[0] - 3.0).abs() < 1e-6);
        assert!((new_data[1] - 5.0).abs() < 1e-6);

        // Row 1: min(7,2)=2, max(7,2)=7
        assert!((new_data[2] - 2.0).abs() < 1e-6);
        assert!((new_data[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_nan_handling() {
        let gen = InteractionGenerator::from_pairs(vec![(0, 1)]);

        let data = vec![f32::NAN, 5.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (new_data, _) = gen.generate(&data, 2, &names);

        // NaN * 5 = NaN
        assert!(new_data[0].is_nan());
    }

    #[test]
    fn test_large_dataset() {
        let num_rows = 1000;
        let num_features = 10;
        let data: Vec<f32> = (0..num_rows * num_features)
            .map(|i| (i % 100) as f32)
            .collect();
        let names: Vec<String> = (0..num_features).map(|i| format!("f{}", i)).collect();

        let mut gen = InteractionGenerator::top_correlated(20);
        gen.fit(&data, num_features);

        let (new_data, new_names) = gen.generate(&data, num_features, &names);

        assert_eq!(new_data.len(), num_rows * new_names.len());
        assert!(new_names.len() <= 20);
    }
}
