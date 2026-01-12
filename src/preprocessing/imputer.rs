//! Missing value imputation strategies
//!
//! This module provides imputation methods for handling missing values:
//!
//! ## SimpleImputer
//! - **Mean**: Replace with column mean (numerical)
//! - **Median**: Replace with column median (robust to outliers)
//! - **Mode**: Replace with most frequent value (categorical)
//! - **Constant**: Replace with a fixed value
//!
//! ## IndicatorImputer
//! - Creates binary indicator columns (1 if was missing, 0 otherwise)
//! - Missingness can be informative (e.g., income missing → likely unemployed)
//!
//! ## Design Philosophy
//!
//! For GBDTs, TreeBoost already handles missing values via bin 0 (implicit indicator).
//! These imputers provide explicit control when needed, especially for:
//! - Mixed ensembles with linear models (which can't handle NaN)
//! - Data export/interchange with other systems
//! - Explicit missingness features

use crate::{Result, TreeBoostError};
use std::collections::HashMap;

// =============================================================================
// Imputation Strategy
// =============================================================================

/// Strategy for imputing missing values
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ImputeStrategy {
    /// Replace with column mean (numerical features)
    #[default]
    Mean,
    /// Replace with column median (robust to outliers)
    Median,
    /// Replace with most frequent value (categorical/discrete)
    Mode,
    /// Replace with a constant value
    Constant(f32),
}

// =============================================================================
// SimpleImputer
// =============================================================================

/// Simple imputer with configurable strategy
///
/// Handles missing values (NaN) by replacing them with computed statistics
/// from the training data.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::{SimpleImputer, ImputeStrategy};
///
/// let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
///
/// // Data with missing values (NaN)
/// let mut data = vec![1.0, f32::NAN, 3.0, 4.0, f32::NAN, 6.0]; // 2 rows × 3 features
///
/// imputer.fit(&data, 3).unwrap();
/// imputer.transform(&mut data, 3).unwrap();
/// // NaN values are now replaced with column means
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimpleImputer {
    /// Imputation strategy
    strategy: ImputeStrategy,
    /// Fill values per feature (computed during fit)
    fill_values: Vec<f32>,
    /// Whether the imputer has been fitted
    fitted: bool,
}

impl SimpleImputer {
    /// Create a new SimpleImputer with the given strategy
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill_values: Vec::new(),
            fitted: false,
        }
    }

    /// Create a mean imputer (most common for numerical)
    pub fn mean() -> Self {
        Self::new(ImputeStrategy::Mean)
    }

    /// Create a median imputer (robust to outliers)
    pub fn median() -> Self {
        Self::new(ImputeStrategy::Median)
    }

    /// Create a mode imputer (for categorical/discrete)
    pub fn mode() -> Self {
        Self::new(ImputeStrategy::Mode)
    }

    /// Create a constant imputer
    pub fn constant(value: f32) -> Self {
        Self::new(ImputeStrategy::Constant(value))
    }

    /// Fit the imputer on training data
    ///
    /// Computes fill values for each feature based on the strategy.
    /// Data is in row-major format: `data[row * num_features + col]`
    pub fn fit(&mut self, data: &[f32], num_features: usize) -> Result<()> {
        if data.is_empty() {
            return Err(TreeBoostError::Data(
                "SimpleImputer::fit() received empty data. Provide at least 1 data point.".into(),
            ));
        }

        let num_rows = data.len() / num_features;
        if data.len() != num_rows * num_features {
            return Err(TreeBoostError::Data(format!(
                "SimpleImputer::fit() received invalid data layout: {} elements not divisible by {} features. \
                 Ensure data is row-major: num_rows × num_features.",
                data.len(),
                num_features
            )));
        }

        self.fill_values = Vec::with_capacity(num_features);

        for col in 0..num_features {
            // Extract non-NaN values for this column
            let values: Vec<f32> = (0..num_rows)
                .map(|row| data[row * num_features + col])
                .filter(|v| !v.is_nan())
                .collect();

            let fill_value = if values.is_empty() {
                // All values are NaN - use 0.0 as fallback
                0.0
            } else {
                match self.strategy {
                    ImputeStrategy::Mean => values.iter().sum::<f32>() / values.len() as f32,
                    ImputeStrategy::Median => compute_median(&values),
                    ImputeStrategy::Mode => compute_mode(&values),
                    ImputeStrategy::Constant(c) => c,
                }
            };

            self.fill_values.push(fill_value);
        }

        self.fitted = true;
        Ok(())
    }

    /// Transform data by replacing NaN values with fitted fill values
    ///
    /// Data is modified in-place in row-major format.
    pub fn transform(&self, data: &mut [f32], num_features: usize) -> Result<()> {
        if !self.fitted {
            return Err(TreeBoostError::Config(
                "SimpleImputer::transform() called before fitting. Call fit() first to learn fill values."
                    .into(),
            ));
        }

        if self.fill_values.len() != num_features {
            return Err(TreeBoostError::Config(format!(
                "SimpleImputer::transform() feature count mismatch: fitted with {} features, but transform() called with {} features. \
                 Ensure transform data has same feature count as training data.",
                self.fill_values.len(),
                num_features
            )));
        }

        let num_rows = data.len() / num_features;

        for row in 0..num_rows {
            for col in 0..num_features {
                let idx = row * num_features + col;
                if data[idx].is_nan() {
                    data[idx] = self.fill_values[col];
                }
            }
        }

        Ok(())
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &mut [f32], num_features: usize) -> Result<()> {
        self.fit(data, num_features)?;
        self.transform(data, num_features)?;
        Ok(())
    }

    /// Check if the imputer has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the fill values (for inspection/debugging)
    pub fn fill_values(&self) -> &[f32] {
        &self.fill_values
    }

    /// Get the strategy
    pub fn strategy(&self) -> ImputeStrategy {
        self.strategy
    }
}

// =============================================================================
// IndicatorImputer
// =============================================================================

/// Creates binary indicator columns for missing values
///
/// Adds new columns indicating whether the original value was missing (NaN).
/// This is useful when missingness itself is informative.
///
/// # Example
///
/// ```rust
/// use treeboost::preprocessing::IndicatorImputer;
///
/// let imputer = IndicatorImputer::new();
///
/// // Data with missing values
/// let data = vec![1.0, f32::NAN, 3.0, 4.0]; // 2 rows × 2 features
/// let feature_names = vec!["age".to_string(), "income".to_string()];
///
/// let (indicators, indicator_names) = imputer.create_indicators(&data, 2, &feature_names);
/// // indicators: [0.0, 1.0, 0.0, 0.0] (income was missing in row 0)
/// // indicator_names: ["age_missing", "income_missing"]
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IndicatorImputer {
    /// Suffix to append to feature names for indicator columns
    suffix: String,
    /// Only create indicators for columns that have missing values
    only_if_missing: bool,
}

impl IndicatorImputer {
    /// Create a new IndicatorImputer with default settings
    pub fn new() -> Self {
        Self {
            suffix: "_missing".to_string(),
            only_if_missing: true,
        }
    }

    /// Set the suffix for indicator column names
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Create indicators for all columns, even if they have no missing values
    pub fn for_all_columns(mut self) -> Self {
        self.only_if_missing = false;
        self
    }

    /// Create missing value indicators
    ///
    /// Returns (indicator_data, indicator_names) where:
    /// - indicator_data: Row-major f32 data (1.0 if missing, 0.0 otherwise)
    /// - indicator_names: Names for the indicator columns
    pub fn create_indicators(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>) {
        let num_rows = data.len() / num_features;

        // First pass: determine which columns have missing values
        let mut has_missing: Vec<bool> = vec![false; num_features];
        if self.only_if_missing {
            for col in 0..num_features {
                for row in 0..num_rows {
                    if data[row * num_features + col].is_nan() {
                        has_missing[col] = true;
                        break;
                    }
                }
            }
        } else {
            has_missing.fill(true);
        }

        // Count output columns
        let num_indicators: usize = has_missing.iter().filter(|&&x| x).count();

        if num_indicators == 0 {
            return (Vec::new(), Vec::new());
        }

        // Create indicator data and names
        let mut indicators = Vec::with_capacity(num_rows * num_indicators);
        let mut names = Vec::with_capacity(num_indicators);

        for col in 0..num_features {
            if !has_missing[col] {
                continue;
            }

            // Add name
            let name = if col < feature_names.len() {
                format!("{}{}", feature_names[col], self.suffix)
            } else {
                format!("f{}{}", col, self.suffix)
            };
            names.push(name);
        }

        // Fill indicator data (row-major, but grouped by indicator column)
        // We need to output in row-major format for consistency
        for row in 0..num_rows {
            for col in 0..num_features {
                if !has_missing[col] {
                    continue;
                }
                let is_missing = data[row * num_features + col].is_nan();
                indicators.push(if is_missing { 1.0 } else { 0.0 });
            }
        }

        (indicators, names)
    }

    /// Create indicators and append them to the original data
    ///
    /// Returns (combined_data, combined_names) with original + indicator columns
    pub fn transform_with_indicators(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>) {
        let num_rows = data.len() / num_features;
        let (indicators, indicator_names) =
            self.create_indicators(data, num_features, feature_names);

        if indicator_names.is_empty() {
            // No missing values, return original
            return (data.to_vec(), feature_names.to_vec());
        }

        let num_indicators = indicator_names.len();
        let total_features = num_features + num_indicators;

        // Combine original data with indicators
        let mut combined = Vec::with_capacity(num_rows * total_features);
        for row in 0..num_rows {
            // Original features
            for col in 0..num_features {
                combined.push(data[row * num_features + col]);
            }
            // Indicator features
            for ind in 0..num_indicators {
                combined.push(indicators[row * num_indicators + ind]);
            }
        }

        // Combine names
        let mut combined_names = feature_names.to_vec();
        combined_names.extend(indicator_names);

        (combined, combined_names)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute median of a slice (requires sorted data internally)
fn compute_median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = sorted.len();
    if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Compute mode (most frequent value) of a slice
/// For continuous values, rounds to 2 decimal places for binning
fn compute_mode(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    // For numerical data, bin values to find mode
    // Round to 2 decimal places
    let mut counts: HashMap<i64, (usize, f32)> = HashMap::new();

    for &v in values {
        let key = (v * 100.0).round() as i64;
        let entry = counts.entry(key).or_insert((0, v));
        entry.0 += 1;
    }

    counts
        .into_values()
        .max_by_key(|(count, _)| *count)
        .map(|(_, value)| value)
        .unwrap_or(0.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_imputer_mean() {
        let mut imputer = SimpleImputer::mean();

        // 3 rows × 2 features, with NaN values
        let mut data = vec![1.0, 2.0, f32::NAN, 4.0, 3.0, f32::NAN];

        imputer.fit(&data, 2).unwrap();
        assert!(imputer.is_fitted());

        // Mean of col 0: (1 + 3) / 2 = 2.0
        // Mean of col 1: (2 + 4) / 2 = 3.0
        assert!((imputer.fill_values()[0] - 2.0).abs() < 0.01);
        assert!((imputer.fill_values()[1] - 3.0).abs() < 0.01);

        imputer.transform(&mut data, 2).unwrap();

        // Check NaN values are replaced
        assert!((data[2] - 2.0).abs() < 0.01); // row 1, col 0
        assert!((data[5] - 3.0).abs() < 0.01); // row 2, col 1
    }

    #[test]
    fn test_simple_imputer_median() {
        let mut imputer = SimpleImputer::median();

        // 4 rows × 1 feature
        let mut data = vec![1.0, 3.0, f32::NAN, 5.0];

        imputer.fit(&data, 1).unwrap();

        // Median of [1, 3, 5] = 3.0
        assert!((imputer.fill_values()[0] - 3.0).abs() < 0.01);

        imputer.transform(&mut data, 1).unwrap();
        assert!((data[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_imputer_mode() {
        let mut imputer = SimpleImputer::mode();

        // 5 rows × 1 feature (mode should be 2.0)
        let mut data = vec![1.0, 2.0, 2.0, f32::NAN, 3.0];

        imputer.fit(&data, 1).unwrap();

        // Mode of [1, 2, 2, 3] = 2.0
        assert!((imputer.fill_values()[0] - 2.0).abs() < 0.01);

        imputer.transform(&mut data, 1).unwrap();
        assert!((data[3] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_simple_imputer_constant() {
        let mut imputer = SimpleImputer::constant(-999.0);

        let mut data = vec![1.0, f32::NAN, 3.0];

        imputer.fit(&data, 1).unwrap();
        assert!((imputer.fill_values()[0] - (-999.0)).abs() < 0.01);

        imputer.transform(&mut data, 1).unwrap();
        assert!((data[1] - (-999.0)).abs() < 0.01);
    }

    #[test]
    fn test_simple_imputer_fit_transform() {
        let mut imputer = SimpleImputer::mean();
        let mut data = vec![1.0, f32::NAN, 3.0];

        imputer.fit_transform(&mut data, 1).unwrap();

        assert!(imputer.is_fitted());
        assert!((data[1] - 2.0).abs() < 0.01); // Mean of [1, 3] = 2
    }

    #[test]
    fn test_simple_imputer_all_nan_column() {
        let mut imputer = SimpleImputer::mean();

        // All NaN in column 0
        let data = vec![f32::NAN, 1.0, f32::NAN, 2.0];

        imputer.fit(&data, 2).unwrap();

        // Column 0 all NaN → fallback to 0.0
        assert!((imputer.fill_values()[0] - 0.0).abs() < 0.01);
        // Column 1 mean = 1.5
        assert!((imputer.fill_values()[1] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_simple_imputer_not_fitted_error() {
        let imputer = SimpleImputer::mean();
        let mut data = vec![1.0, 2.0];

        let result = imputer.transform(&mut data, 2);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("fit") && err_msg.contains("SimpleImputer"),
            "Expected error message to mention fitting and component name, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_indicator_imputer_basic() {
        let imputer = IndicatorImputer::new();

        // 2 rows × 2 features
        let data = vec![1.0, f32::NAN, 3.0, 4.0];
        let names = vec!["age".to_string(), "income".to_string()];

        let (indicators, indicator_names) = imputer.create_indicators(&data, 2, &names);

        // Only income has missing values
        assert_eq!(indicator_names.len(), 1);
        assert_eq!(indicator_names[0], "income_missing");

        // Row 0: income missing (1.0), Row 1: income not missing (0.0)
        assert_eq!(indicators, vec![1.0, 0.0]);
    }

    #[test]
    fn test_indicator_imputer_all_columns() {
        let imputer = IndicatorImputer::new().for_all_columns();

        let data = vec![1.0, 2.0, 3.0, 4.0]; // No NaN
        let names = vec!["a".to_string(), "b".to_string()];

        let (indicators, indicator_names) = imputer.create_indicators(&data, 2, &names);

        // All columns get indicators even without NaN
        assert_eq!(indicator_names.len(), 2);
        assert_eq!(indicators, vec![0.0, 0.0, 0.0, 0.0]); // All zeros
    }

    #[test]
    fn test_indicator_imputer_custom_suffix() {
        let imputer = IndicatorImputer::new().with_suffix("_is_null");

        let data = vec![f32::NAN, 2.0];
        let names = vec!["x".to_string()];

        let (_, indicator_names) = imputer.create_indicators(&data, 1, &names);

        assert_eq!(indicator_names[0], "x_is_null");
    }

    #[test]
    fn test_indicator_imputer_transform_with_indicators() {
        let imputer = IndicatorImputer::new();

        // 2 rows × 2 features
        let data = vec![1.0, f32::NAN, 3.0, 4.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (combined, combined_names) = imputer.transform_with_indicators(&data, 2, &names);

        // 3 features now: a, b, b_missing
        assert_eq!(combined_names.len(), 3);
        assert_eq!(combined_names, vec!["a", "b", "b_missing"]);

        // 2 rows × 3 features = 6 values
        assert_eq!(combined.len(), 6);

        // Row 0: 1.0, NaN, 1.0 (b was missing)
        assert!((combined[0] - 1.0).abs() < 0.01);
        assert!(combined[1].is_nan());
        assert!((combined[2] - 1.0).abs() < 0.01);

        // Row 1: 3.0, 4.0, 0.0 (b not missing)
        assert!((combined[3] - 3.0).abs() < 0.01);
        assert!((combined[4] - 4.0).abs() < 0.01);
        assert!((combined[5] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_indicator_imputer_no_missing() {
        let imputer = IndicatorImputer::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let names = vec!["a".to_string(), "b".to_string()];

        let (indicators, indicator_names) = imputer.create_indicators(&data, 2, &names);

        // No missing values → no indicators
        assert!(indicators.is_empty());
        assert!(indicator_names.is_empty());
    }

    #[test]
    fn test_compute_median() {
        assert!((compute_median(&[1.0, 2.0, 3.0]) - 2.0).abs() < 0.01);
        assert!((compute_median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 0.01);
        assert!((compute_median(&[5.0]) - 5.0).abs() < 0.01);
        assert!((compute_median(&[]) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_mode() {
        assert!((compute_mode(&[1.0, 2.0, 2.0, 3.0]) - 2.0).abs() < 0.01);
        assert!((compute_mode(&[5.0]) - 5.0).abs() < 0.01);
        assert!((compute_mode(&[]) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_imputer_serialization() {
        let mut imputer = SimpleImputer::mean();
        imputer.fit(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();

        let json = serde_json::to_string(&imputer).unwrap();
        let loaded: SimpleImputer = serde_json::from_str(&json).unwrap();

        assert!(loaded.is_fitted());
        assert_eq!(loaded.fill_values(), imputer.fill_values());
    }
}
