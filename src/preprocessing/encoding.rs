//! Standard categorical encoding transformations (sklearn-style)
//!
//! Provides **simple, general-purpose encoders** following the sklearn API pattern.
//! These are ideal for quick prototyping, low-cardinality categoricals, and standard
//! data preparation workflows.
//!
//! # Available Encoders
//!
//! - [`FrequencyEncoder`]: Maps category → count (optimal for trees)
//! - [`LabelEncoder`]: Maps string → u32 (essential for CSV loading)
//! - [`OneHotEncoder`]: Maps category → binary columns (for linear models)
//!
//! # When to Use This Module vs `encoding`
//!
//! | Scenario | Use This Module | Use `encoding` |
//! |----------|-----------------|----------------|
//! | Low-cardinality (< 100 categories) | ✅ | ⚠️ overkill |
//! | Quick prototyping | ✅ | ❌ |
//! | Simple label/frequency encoding | ✅ | ❌ |
//! | One-hot encoding for linear models | ✅ | ❌ |
//! | High-cardinality (100+ categories) | ⚠️ | ✅ |
//! | Production with unseen categories | ⚠️ | ✅ |
//! | Rare category filtering | ❌ | ✅ |
//! | Target-based encoding with smoothing | ❌ | ✅ |
//!
//! # GBDT vs Linear Model Considerations
//!
//! **For Trees (GBDT)**:
//! - Prefer FrequencyEncoder or LabelEncoder
//! - Trees can split on numerical magnitude (rare vs common)
//! - OneHot is detrimental (forces deep trees, memory waste)
//!
//! **For Linear Models**:
//! - Prefer OneHotEncoder
//! - Linear models need binary indicators for interpretable coefficients
//! - Frequency/Label encoding creates ordinal relationship that doesn't exist
//!
//! **For Mixed Ensembles (Linear + Tree)**:
//! - Use OneHot for linear component
//! - Use Frequency/Label for tree component
//! - Or use both and let the model select
//!
//! # See Also
//!
//! - [`crate::encoding`]: Production-grade encoders for high-cardinality features

use std::collections::HashMap;

use crate::{Result, TreeBoostError};

// =============================================================================
// FrequencyEncoder
// =============================================================================

/// FrequencyEncoder: Maps category → count (frequency) in training set
///
/// Optimal for GBDTs because trees can easily split on "rare vs common" categories.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::FrequencyEncoder;
///
/// let categories = vec!["A", "B", "A", "C", "A", "B"];
/// let mut encoder = FrequencyEncoder::new();
/// encoder.fit(&categories);
///
/// // A appears 3 times, B appears 2 times, C appears 1 time
/// assert_eq!(encoder.transform_single("A"), Some(3.0));
/// assert_eq!(encoder.transform_single("B"), Some(2.0));
/// assert_eq!(encoder.transform_single("C"), Some(1.0));
/// assert_eq!(encoder.transform_single("D"), Some(0.0)); // Unknown category → default value
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrequencyEncoder {
    /// Maps category string → count in training set
    counts: HashMap<String, usize>,
    /// Total number of samples seen during fit
    total_count: usize,
    /// Value to use for unknown categories (None = error, Some(x) = use x)
    unknown_value: Option<f32>,
    /// Whether to normalize counts to [0, 1] range
    normalize: bool,
    /// Whether fit() has been called
    fitted: bool,
}

impl FrequencyEncoder {
    /// Create a new unfitted FrequencyEncoder
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total_count: 0,
            unknown_value: Some(0.0), // Default: unknown categories get 0
            normalize: false,
            fitted: false,
        }
    }

    /// Set the value to use for unknown categories
    ///
    /// - `Some(x)`: Use value x for unknown categories
    /// - `None`: Return error for unknown categories
    pub fn with_unknown_value(mut self, value: Option<f32>) -> Self {
        self.unknown_value = value;
        self
    }

    /// Enable normalization (counts / total → [0, 1] range)
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Fit the encoder on training data
    pub fn fit(&mut self, categories: &[impl AsRef<str>]) {
        self.counts.clear();
        self.total_count = categories.len();

        for cat in categories {
            *self.counts.entry(cat.as_ref().to_string()).or_insert(0) += 1;
        }

        self.fitted = true;
    }

    /// Transform a single category to its frequency
    pub fn transform_single(&self, category: &str) -> Option<f32> {
        if !self.fitted {
            return None;
        }

        match self.counts.get(category) {
            Some(&count) => {
                let value = if self.normalize {
                    count as f32 / self.total_count as f32
                } else {
                    count as f32
                };
                Some(value)
            }
            None => self.unknown_value,
        }
    }

    /// Transform multiple categories to their frequencies
    pub fn transform(&self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "FrequencyEncoder not fitted. Call fit() first.".into(),
            ));
        }

        let mut result = Vec::with_capacity(categories.len());

        for (i, cat) in categories.iter().enumerate() {
            match self.transform_single(cat.as_ref()) {
                Some(value) => result.push(value),
                None => {
                    return Err(TreeBoostError::Data(format!(
                        "Unknown category '{}' at index {} and no unknown_value set",
                        cat.as_ref(),
                        i
                    )));
                }
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        self.fit(categories);
        self.transform(categories)
    }

    /// Check if encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the number of unique categories
    pub fn num_categories(&self) -> usize {
        self.counts.len()
    }

    /// Get the counts map (for inspection)
    pub fn counts(&self) -> &HashMap<String, usize> {
        &self.counts
    }
}

impl Default for FrequencyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// LabelEncoder
// =============================================================================

/// LabelEncoder: Maps string categories → integer labels (u32)
///
/// Essential for converting CSV categorical columns to numerical input for trees.
/// Categories are assigned labels in sorted order (alphabetically) for consistency.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::LabelEncoder;
///
/// let categories = vec!["red", "blue", "red", "green"];
/// let mut encoder = LabelEncoder::new();
/// encoder.fit(&categories);
///
/// // Sorted alphabetically: blue=0, green=1, red=2
/// assert_eq!(encoder.transform_single("blue"), Some(0));
/// assert_eq!(encoder.transform_single("green"), Some(1));
/// assert_eq!(encoder.transform_single("red"), Some(2));
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LabelEncoder {
    /// Maps category string → integer label
    mapping: HashMap<String, u32>,
    /// Reverse mapping: label → category string
    inverse_mapping: Vec<String>,
    /// Value to use for unknown categories (None = error)
    unknown_label: Option<u32>,
    /// Whether fit() has been called
    fitted: bool,
}

impl LabelEncoder {
    /// Create a new unfitted LabelEncoder
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            inverse_mapping: Vec::new(),
            unknown_label: None,
            fitted: false,
        }
    }

    /// Set the label to use for unknown categories
    ///
    /// - `Some(x)`: Use label x for unknown categories
    /// - `None`: Return error for unknown categories (default)
    pub fn with_unknown_label(mut self, label: Option<u32>) -> Self {
        self.unknown_label = label;
        self
    }

    /// Fit the encoder on training data
    ///
    /// Categories are sorted alphabetically and assigned sequential labels.
    pub fn fit(&mut self, categories: &[impl AsRef<str>]) {
        // Collect unique categories
        let mut unique: Vec<String> = categories
            .iter()
            .map(|c| c.as_ref().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Sort for consistent ordering
        unique.sort();

        // Create mapping
        self.mapping.clear();
        self.inverse_mapping = unique.clone();

        for (label, category) in unique.into_iter().enumerate() {
            self.mapping.insert(category, label as u32);
        }

        self.fitted = true;
    }

    /// Transform a single category to its label
    pub fn transform_single(&self, category: &str) -> Option<u32> {
        if !self.fitted {
            return None;
        }

        match self.mapping.get(category) {
            Some(&label) => Some(label),
            None => self.unknown_label,
        }
    }

    /// Transform multiple categories to their labels
    pub fn transform(&self, categories: &[impl AsRef<str>]) -> Result<Vec<u32>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "LabelEncoder not fitted. Call fit() first.".into(),
            ));
        }

        let mut result = Vec::with_capacity(categories.len());

        for (i, cat) in categories.iter().enumerate() {
            match self.transform_single(cat.as_ref()) {
                Some(label) => result.push(label),
                None => {
                    return Err(TreeBoostError::Data(format!(
                        "Unknown category '{}' at index {} and no unknown_label set",
                        cat.as_ref(),
                        i
                    )));
                }
            }
        }

        Ok(result)
    }

    /// Transform labels as f32 (for direct use in feature arrays)
    pub fn transform_f32(&self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        self.transform(categories)
            .map(|labels| labels.into_iter().map(|l| l as f32).collect())
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, categories: &[impl AsRef<str>]) -> Result<Vec<u32>> {
        self.fit(categories);
        self.transform(categories)
    }

    /// Inverse transform: convert labels back to categories
    pub fn inverse_transform(&self, labels: &[u32]) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "LabelEncoder not fitted. Call fit() first.".into(),
            ));
        }

        let mut result = Vec::with_capacity(labels.len());

        for (i, &label) in labels.iter().enumerate() {
            if (label as usize) < self.inverse_mapping.len() {
                result.push(self.inverse_mapping[label as usize].clone());
            } else {
                return Err(TreeBoostError::Data(format!(
                    "Unknown label {} at index {}",
                    label, i
                )));
            }
        }

        Ok(result)
    }

    /// Check if encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the number of unique categories (classes)
    pub fn num_classes(&self) -> usize {
        self.mapping.len()
    }

    /// Get the category for a given label
    pub fn get_category(&self, label: u32) -> Option<&str> {
        self.inverse_mapping.get(label as usize).map(|s| s.as_str())
    }

    /// Get the label for a given category
    pub fn get_label(&self, category: &str) -> Option<u32> {
        self.mapping.get(category).copied()
    }

    /// Get all categories in label order
    pub fn classes(&self) -> &[String] {
        &self.inverse_mapping
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// OneHotEncoder
// =============================================================================

/// Strategy for handling unknown categories in test set
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum UnknownStrategy {
    /// All one-hot columns = 0 for unknown category
    AllZeros,
    /// Return error if unknown category encountered
    Error,
}

/// OneHotEncoder: Maps category → binary columns
///
/// **WARNING**: Not recommended for pure GBDTs! Use for linear models in mixed ensembles.
///
/// Creates N binary columns for N categories, where exactly one column is 1.0 and
/// the rest are 0.0 for each sample.
///
/// # Why This Hurts Trees
///
/// - Increases memory usage (N categories → N features)
/// - Forces trees to grow very deep to recover category information
/// - One split per category (inefficient)
///
/// # When to Use
///
/// - Linear models in mixed ensembles (need binary indicators)
/// - When interpretability requires explicit category coefficients
/// - Low-cardinality categoricals only (< 20 categories)
///
/// # Safety Limit
///
/// By default, OneHotEncoder has a `max_categories` limit of 100 to prevent
/// memory explosion with high-cardinality features. Use `with_max_categories()`
/// to adjust if needed, or consider using TargetEncoder for high-cardinality.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{OneHotEncoder, UnknownStrategy};
///
/// let categories = vec!["red", "blue", "red", "green"];
/// let mut encoder = OneHotEncoder::new();
/// encoder.fit(&categories);
///
/// // 3 categories → 3 columns: [blue, green, red] (sorted)
/// let encoded = encoder.transform(&categories).unwrap();
/// // "red"   → [0.0, 0.0, 1.0]
/// // "blue"  → [1.0, 0.0, 0.0]
/// // "green" → [0.0, 1.0, 0.0]
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OneHotEncoder {
    /// Sorted list of categories
    categories: Vec<String>,
    /// Maps category → column index
    category_to_idx: HashMap<String, usize>,
    /// How to handle unknown categories
    handle_unknown: UnknownStrategy,
    /// Whether to drop first category (avoid multicollinearity for linear models)
    drop_first: bool,
    /// Maximum allowed categories (0 = unlimited, default = 100)
    max_categories: usize,
    /// Whether fit() has been called
    fitted: bool,
}

impl OneHotEncoder {
    /// Create a new unfitted OneHotEncoder
    ///
    /// Default `max_categories` is 100 to prevent memory explosion.
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            category_to_idx: HashMap::new(),
            handle_unknown: UnknownStrategy::AllZeros,
            drop_first: false,
            max_categories: 100, // Default safety limit
            fitted: false,
        }
    }

    /// Set the strategy for handling unknown categories
    pub fn with_unknown_strategy(mut self, strategy: UnknownStrategy) -> Self {
        self.handle_unknown = strategy;
        self
    }

    /// Drop the first category column (for linear models to avoid multicollinearity)
    pub fn with_drop_first(mut self, drop: bool) -> Self {
        self.drop_first = drop;
        self
    }

    /// Set the maximum allowed number of categories
    ///
    /// Default is 100. Set to 0 for unlimited (not recommended).
    ///
    /// **Warning**: High-cardinality one-hot encoding can cause memory explosion.
    /// For features with >100 categories, consider using `TargetEncoder` instead.
    pub fn with_max_categories(mut self, max: usize) -> Self {
        self.max_categories = max;
        self
    }

    /// Get the maximum allowed categories setting
    pub fn max_categories(&self) -> usize {
        self.max_categories
    }

    /// Fit the encoder on training data
    ///
    /// # Errors
    ///
    /// Returns an error if the number of unique categories exceeds `max_categories`.
    pub fn fit(&mut self, categories: &[impl AsRef<str>]) -> Result<()> {
        // Collect unique categories
        let mut unique: Vec<String> = categories
            .iter()
            .map(|c| c.as_ref().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Check category limit
        if self.max_categories > 0 && unique.len() > self.max_categories {
            return Err(TreeBoostError::Config(format!(
                "OneHotEncoder: {} unique categories exceeds max_categories limit of {}. \
                 High-cardinality one-hot encoding can cause memory explosion. \
                 Consider using TargetEncoder or increasing max_categories with with_max_categories().",
                unique.len(),
                self.max_categories
            )));
        }

        // Sort for consistent ordering
        unique.sort();

        // Create mapping
        self.category_to_idx.clear();
        for (idx, cat) in unique.iter().enumerate() {
            self.category_to_idx.insert(cat.clone(), idx);
        }

        self.categories = unique;
        self.fitted = true;
        Ok(())
    }

    /// Get the number of output columns
    pub fn num_columns(&self) -> usize {
        if self.drop_first && !self.categories.is_empty() {
            self.categories.len() - 1
        } else {
            self.categories.len()
        }
    }

    /// Get the output feature names
    pub fn get_feature_names(&self, prefix: &str) -> Vec<String> {
        let start_idx = if self.drop_first { 1 } else { 0 };
        self.categories[start_idx..]
            .iter()
            .map(|cat| format!("{}_{}", prefix, cat))
            .collect()
    }

    /// Transform a single category to one-hot vector
    pub fn transform_single(&self, category: &str) -> Result<Vec<f32>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "OneHotEncoder not fitted. Call fit() first.".into(),
            ));
        }

        let num_cols = self.num_columns();
        let mut result = vec![0.0; num_cols];

        match self.category_to_idx.get(category) {
            Some(&idx) => {
                let adjusted_idx = if self.drop_first {
                    idx.saturating_sub(1)
                } else {
                    idx
                };
                // If drop_first and this is the first category, all zeros (it's the reference)
                if !(self.drop_first && idx == 0) && adjusted_idx < num_cols {
                    result[adjusted_idx] = 1.0;
                }
            }
            None => {
                // Unknown category
                match self.handle_unknown {
                    UnknownStrategy::AllZeros => {
                        // result is already all zeros
                    }
                    UnknownStrategy::Error => {
                        return Err(TreeBoostError::Data(format!(
                            "Unknown category '{}'",
                            category
                        )));
                    }
                }
            }
        }

        Ok(result)
    }

    /// Transform multiple categories to one-hot matrix (flattened row-major)
    ///
    /// Returns a flat array where each row of `num_columns()` represents one sample.
    pub fn transform(&self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        if !self.fitted {
            return Err(TreeBoostError::Data(
                "OneHotEncoder not fitted. Call fit() first.".into(),
            ));
        }

        let num_cols = self.num_columns();
        let mut result = Vec::with_capacity(categories.len() * num_cols);

        for cat in categories {
            let row = self.transform_single(cat.as_ref())?;
            result.extend(row);
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, categories: &[impl AsRef<str>]) -> Result<Vec<f32>> {
        self.fit(categories)?;
        self.transform(categories)
    }

    /// Check if encoder has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the categories (in order)
    pub fn categories(&self) -> &[String] {
        &self.categories
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // FrequencyEncoder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_frequency_encoder_basic() {
        let categories = vec!["A", "B", "A", "C", "A", "B"];
        let mut encoder = FrequencyEncoder::new();
        encoder.fit(&categories);

        assert!(encoder.is_fitted());
        assert_eq!(encoder.num_categories(), 3);

        // A appears 3 times, B appears 2 times, C appears 1 time
        assert_eq!(encoder.transform_single("A"), Some(3.0));
        assert_eq!(encoder.transform_single("B"), Some(2.0));
        assert_eq!(encoder.transform_single("C"), Some(1.0));
    }

    #[test]
    fn test_frequency_encoder_unknown() {
        let categories = vec!["A", "B", "A"];
        let mut encoder = FrequencyEncoder::new();
        encoder.fit(&categories);

        // Default unknown_value is 0.0
        assert_eq!(encoder.transform_single("D"), Some(0.0));

        // With unknown_value = None, should return None
        let mut encoder2 = FrequencyEncoder::new().with_unknown_value(None);
        encoder2.fit(&categories);
        assert_eq!(encoder2.transform_single("D"), None);
    }

    #[test]
    fn test_frequency_encoder_normalize() {
        let categories = vec!["A", "B", "A", "A", "B"];
        let mut encoder = FrequencyEncoder::new().with_normalize(true);
        encoder.fit(&categories);

        // A: 3/5 = 0.6, B: 2/5 = 0.4
        assert!((encoder.transform_single("A").unwrap() - 0.6).abs() < 1e-6);
        assert!((encoder.transform_single("B").unwrap() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_encoder_transform_batch() {
        let categories = vec!["A", "B", "A", "C", "A"];
        let mut encoder = FrequencyEncoder::new();
        encoder.fit(&categories);

        let result = encoder.transform(&["A", "B", "C"]).unwrap();
        assert_eq!(result, vec![3.0, 1.0, 1.0]);
    }

    // -------------------------------------------------------------------------
    // LabelEncoder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_encoder_basic() {
        let categories = vec!["red", "blue", "red", "green"];
        let mut encoder = LabelEncoder::new();
        encoder.fit(&categories);

        assert!(encoder.is_fitted());
        assert_eq!(encoder.num_classes(), 3);

        // Sorted alphabetically: blue=0, green=1, red=2
        assert_eq!(encoder.transform_single("blue"), Some(0));
        assert_eq!(encoder.transform_single("green"), Some(1));
        assert_eq!(encoder.transform_single("red"), Some(2));
    }

    #[test]
    fn test_label_encoder_unknown() {
        let categories = vec!["A", "B"];
        let mut encoder = LabelEncoder::new();
        encoder.fit(&categories);

        // Default: unknown returns None
        assert_eq!(encoder.transform_single("C"), None);

        // With unknown_label set
        let mut encoder2 = LabelEncoder::new().with_unknown_label(Some(999));
        encoder2.fit(&categories);
        assert_eq!(encoder2.transform_single("C"), Some(999));
    }

    #[test]
    fn test_label_encoder_inverse_transform() {
        let categories = vec!["red", "blue", "green"];
        let mut encoder = LabelEncoder::new();
        encoder.fit(&categories);

        let labels = encoder.transform(&["red", "blue", "green"]).unwrap();
        let reversed = encoder.inverse_transform(&labels).unwrap();

        assert_eq!(reversed, vec!["red", "blue", "green"]);
    }

    #[test]
    fn test_label_encoder_classes() {
        let categories = vec!["C", "A", "B", "A"];
        let mut encoder = LabelEncoder::new();
        encoder.fit(&categories);

        // Should be sorted alphabetically
        assert_eq!(encoder.classes(), &["A", "B", "C"]);
    }

    // -------------------------------------------------------------------------
    // OneHotEncoder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_onehot_encoder_basic() {
        let categories = vec!["red", "blue", "green"];
        let mut encoder = OneHotEncoder::new();
        encoder.fit(&categories).unwrap();

        assert!(encoder.is_fitted());
        assert_eq!(encoder.num_columns(), 3);

        // Sorted: blue, green, red
        let blue = encoder.transform_single("blue").unwrap();
        let green = encoder.transform_single("green").unwrap();
        let red = encoder.transform_single("red").unwrap();

        assert_eq!(blue, vec![1.0, 0.0, 0.0]);
        assert_eq!(green, vec![0.0, 1.0, 0.0]);
        assert_eq!(red, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_onehot_encoder_drop_first() {
        let categories = vec!["red", "blue", "green"];
        let mut encoder = OneHotEncoder::new().with_drop_first(true);
        encoder.fit(&categories).unwrap();

        // 3 categories, drop first → 2 columns
        assert_eq!(encoder.num_columns(), 2);

        // Sorted: blue (dropped), green, red
        let blue = encoder.transform_single("blue").unwrap();
        let green = encoder.transform_single("green").unwrap();
        let red = encoder.transform_single("red").unwrap();

        // blue is reference category (all zeros)
        assert_eq!(blue, vec![0.0, 0.0]);
        assert_eq!(green, vec![1.0, 0.0]);
        assert_eq!(red, vec![0.0, 1.0]);
    }

    #[test]
    fn test_onehot_encoder_unknown_allzeros() {
        let categories = vec!["A", "B"];
        let mut encoder = OneHotEncoder::new().with_unknown_strategy(UnknownStrategy::AllZeros);
        encoder.fit(&categories).unwrap();

        let unknown = encoder.transform_single("C").unwrap();
        assert_eq!(unknown, vec![0.0, 0.0]);
    }

    #[test]
    fn test_onehot_encoder_unknown_error() {
        let categories = vec!["A", "B"];
        let mut encoder = OneHotEncoder::new().with_unknown_strategy(UnknownStrategy::Error);
        encoder.fit(&categories).unwrap();

        let result = encoder.transform_single("C");
        assert!(result.is_err());
    }

    #[test]
    fn test_onehot_encoder_feature_names() {
        let categories = vec!["red", "blue", "green"];
        let mut encoder = OneHotEncoder::new();
        encoder.fit(&categories).unwrap();

        let names = encoder.get_feature_names("color");
        assert_eq!(names, vec!["color_blue", "color_green", "color_red"]);
    }

    #[test]
    fn test_onehot_encoder_batch_transform() {
        let categories = vec!["A", "B", "C"];
        let mut encoder = OneHotEncoder::new();
        encoder.fit(&categories).unwrap();

        let result = encoder.transform(&["A", "B", "C"]).unwrap();
        // 3 samples × 3 columns = 9 values
        assert_eq!(result.len(), 9);
        // A: [1, 0, 0], B: [0, 1, 0], C: [0, 0, 1]
        assert_eq!(result, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_onehot_encoder_max_categories_limit() {
        // Create categories exceeding the default limit
        let categories: Vec<String> = (0..150).map(|i| format!("cat_{}", i)).collect();

        // Default max_categories is 100, should fail
        let mut encoder = OneHotEncoder::new();
        let result = encoder.fit(&categories);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("150 unique categories"));
        assert!(err_msg.contains("max_categories limit of 100"));

        // With increased limit, should succeed
        let mut encoder2 = OneHotEncoder::new().with_max_categories(200);
        assert!(encoder2.fit(&categories).is_ok());
        assert_eq!(encoder2.num_columns(), 150);

        // With unlimited (0), should succeed
        let mut encoder3 = OneHotEncoder::new().with_max_categories(0);
        assert!(encoder3.fit(&categories).is_ok());

        // Small category count should always succeed
        let small_categories = vec!["A", "B", "C"];
        let mut encoder4 = OneHotEncoder::new();
        assert!(encoder4.fit(&small_categories).is_ok());
    }
}
