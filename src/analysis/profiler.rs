//! Column and DataFrame Profiling for AutoML
//!
//! This module provides the "blood test" for each column - analyzing data characteristics
//! to inform smart preprocessing, feature engineering, and model selection decisions.
//!
//! # Design Philosophy
//!
//! **Two-Pass Analysis for Efficiency:**
//! 1. **Pass 1 (Cheap)**: Use Polars native stats (`null_count`, `n_unique`, `min`, `max`)
//!    to identify columns to DROP (constant, ID-like, text) - ~1ms per column
//! 2. **Pass 2 (Expensive)**: Only for kept columns, compute skewness, kurtosis,
//!    target correlation - these require full scans
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::profiler::{DataFrameProfile, TaskType};
//!
//! let profile = DataFrameProfile::analyze(&df, "target")?;
//! println!("{}", profile.report());
//!
//! // Check what to drop
//! for col in &profile.drop_columns {
//!     println!("Dropping: {} (reason: {})", col.name, col.reason);
//! }
//!
//! // Get task type
//! match profile.task_type {
//!     TaskType::Regression => println!("Regression task"),
//!     TaskType::BinaryClassification => println!("Binary classification"),
//!     TaskType::MultiClassification { num_classes } => println!("{}-class classification", num_classes),
//! }
//! ```

use crate::preprocessing::polars_ext::{is_categorical, is_numeric};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::collections::HashSet;

// Use alias to avoid conflict with Polars DataType
use polars::prelude::DataType as PolarsDataType;

/// Data type classification for a column
///
/// Text vs Categorical distinction is critical:
/// - Categorical: Low cardinality strings that can be encoded (city names, categories)
/// - Text: High cardinality + long strings that need embeddings (reviews, descriptions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDataType {
    /// Numeric column (int or float)
    Numeric,
    /// Categorical column - low cardinality strings that can be encoded
    Categorical,
    /// Text column - high cardinality + long strings (DROP for v1, needs NLP)
    Text,
    /// DateTime column - can extract seasonal features
    DateTime,
    /// Boolean column - binary indicator
    Boolean,
}

impl ColumnDataType {
    /// Detect data type from a Polars column
    ///
    /// Heuristic for Text vs Categorical:
    /// - String + cardinality > 100 + avg_length > 20 → Text
    /// - String + cardinality <= 100 OR avg_length <= 20 → Categorical
    pub fn detect(column: &Column, num_rows: usize) -> Self {
        let dtype = column.dtype();

        // Check for boolean first
        if matches!(dtype, PolarsDataType::Boolean) {
            return ColumnDataType::Boolean;
        }

        // Check for datetime
        if matches!(
            dtype,
            PolarsDataType::Date | PolarsDataType::Datetime(_, _) | PolarsDataType::Time | PolarsDataType::Duration(_)
        ) {
            return ColumnDataType::DateTime;
        }

        // Check for numeric
        if is_numeric(column) {
            return ColumnDataType::Numeric;
        }

        // Check for string/categorical
        if is_categorical(column) {
            // Detect Text vs Categorical
            let cardinality = column.n_unique().unwrap_or(0);

            // For high cardinality, check average string length
            if cardinality > 100 {
                let avg_len = Self::estimate_avg_string_length(column);
                if avg_len > 20.0 {
                    return ColumnDataType::Text;
                }
            }

            // Also check if cardinality ratio is very high (potential ID)
            let cardinality_ratio = cardinality as f32 / num_rows.max(1) as f32;
            if cardinality_ratio > 0.9 && cardinality > 50 {
                // High cardinality ratio suggests ID-like or text
                let avg_len = Self::estimate_avg_string_length(column);
                if avg_len > 15.0 {
                    return ColumnDataType::Text;
                }
            }

            return ColumnDataType::Categorical;
        }

        // Default to categorical for unknown types
        ColumnDataType::Categorical
    }

    /// Estimate average string length by sampling
    fn estimate_avg_string_length(column: &Column) -> f32 {
        let Ok(str_col) = column.str() else {
            return 0.0;
        };

        let mut total_len = 0usize;
        let mut count = 0usize;
        let sample_size = 100.min(str_col.len());

        for i in 0..sample_size {
            if let Some(s) = str_col.get(i) {
                total_len += s.len();
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_len as f32 / count as f32
        }
    }
}

/// Reason why a column should be dropped
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropReason {
    /// Column has only one unique value
    Constant,
    /// Column appears to be an ID (high cardinality + monotonic)
    IdLike,
    /// Column is text that requires NLP (not supported in v1)
    Text,
    /// Column is the target (excluded from features)
    Target,
}

impl std::fmt::Display for DropReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DropReason::Constant => write!(f, "constant (single value)"),
            DropReason::IdLike => write!(f, "ID-like (unique per row)"),
            DropReason::Text => write!(f, "text (requires NLP, not supported)"),
            DropReason::Target => write!(f, "target column"),
        }
    }
}

/// Column marked for dropping with reason
#[derive(Debug, Clone)]
pub struct DroppedColumn {
    pub name: String,
    pub reason: DropReason,
}

/// Profile of a single column
///
/// Contains both cheap stats (Pass 1) and expensive stats (Pass 2).
/// Expensive stats are `Option` because they're only computed for kept columns.
#[derive(Debug, Clone)]
pub struct ColumnProfile {
    /// Column name
    pub name: String,
    /// Detected data type
    pub dtype: ColumnDataType,

    // === Pass 1: Cheap stats (Polars native) ===
    /// Fraction of missing values [0, 1]
    pub missing_ratio: f32,
    /// Number of unique values
    pub cardinality: usize,
    /// Cardinality / total rows
    pub cardinality_ratio: f32,
    /// True if column has only one unique value
    pub is_constant: bool,
    /// True if column appears to be an ID (high cardinality + monotonic)
    pub is_id_like: bool,

    // === Pass 2: Expensive stats (only for kept columns) ===
    /// Skewness (only for Numeric)
    pub skewness: Option<f32>,
    /// Kurtosis (only for Numeric)
    pub kurtosis: Option<f32>,
    /// Minimum value (only for Numeric)
    pub min: Option<f64>,
    /// Maximum value (only for Numeric)
    pub max: Option<f64>,
    /// True if column contains negative values
    pub has_negative: bool,
    /// True if column has exactly 2 unique values
    pub is_binary: bool,
    /// Correlation with target (computed with target)
    pub target_correlation: Option<f32>,
}

impl ColumnProfile {
    /// Run Pass 1 analysis (cheap stats using Polars native functions)
    fn analyze_pass1(column: &Column, num_rows: usize) -> Self {
        let name = column.name().to_string();
        let dtype = ColumnDataType::detect(column, num_rows);

        // Cheap stats using Polars native
        let null_count = column.null_count();
        let missing_ratio = null_count as f32 / num_rows.max(1) as f32;

        let cardinality = column.n_unique().unwrap_or(0);
        let cardinality_ratio = cardinality as f32 / num_rows.max(1) as f32;

        let is_constant = cardinality <= 1;

        // ID-like detection: high cardinality + monotonic (for numeric)
        let is_id_like = if dtype == ColumnDataType::Numeric && cardinality_ratio > 0.9 {
            Self::check_monotonic(column)
        } else {
            cardinality_ratio > 0.95 // For non-numeric, just use very high cardinality
        };

        // Quick binary check
        let is_binary = cardinality == 2;

        // Quick negative check for numeric
        let has_negative = if dtype == ColumnDataType::Numeric {
            Self::extract_scalar_f64(&column.min_reduce().ok())
                .is_some_and(|v| v < 0.0)
        } else {
            false
        };

        // Min/max from Polars (cheap)
        let (min, max) = if dtype == ColumnDataType::Numeric {
            let min_val = Self::extract_scalar_f64(&column.min_reduce().ok());
            let max_val = Self::extract_scalar_f64(&column.max_reduce().ok());
            (min_val, max_val)
        } else {
            (None, None)
        };

        Self {
            name,
            dtype,
            missing_ratio,
            cardinality,
            cardinality_ratio,
            is_constant,
            is_id_like,
            skewness: None,
            kurtosis: None,
            min,
            max,
            has_negative,
            is_binary,
            target_correlation: None,
        }
    }

    /// Extract f64 from Polars Scalar
    fn extract_scalar_f64(scalar: &Option<Scalar>) -> Option<f64> {
        scalar.as_ref().and_then(|s| {
            // Get the AnyValue from the scalar and convert to f64
            let av = s.value();
            match av {
                AnyValue::Float64(v) => Some(*v),
                AnyValue::Float32(v) => Some(*v as f64),
                AnyValue::Int64(v) => Some(*v as f64),
                AnyValue::Int32(v) => Some(*v as f64),
                AnyValue::Int16(v) => Some(*v as f64),
                AnyValue::Int8(v) => Some(*v as f64),
                AnyValue::UInt64(v) => Some(*v as f64),
                AnyValue::UInt32(v) => Some(*v as f64),
                AnyValue::UInt16(v) => Some(*v as f64),
                AnyValue::UInt8(v) => Some(*v as f64),
                _ => None,
            }
        })
    }

    /// Run Pass 2 analysis (expensive stats - only for kept columns)
    fn analyze_pass2(&mut self, column: &Column, target: Option<&[f32]>) {
        if self.dtype != ColumnDataType::Numeric {
            return;
        }

        // Compute skewness and kurtosis
        let values: Vec<f64> = column
            .cast(&PolarsDataType::Float64)
            .ok()
            .and_then(|c| c.f64().ok().map(|ca| ca.into_iter().flatten().collect()))
            .unwrap_or_default();

        if values.len() < 3 {
            return;
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std > 1e-10 {
            // Skewness: E[(X - μ)³] / σ³
            let m3 = values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
            self.skewness = Some(m3 as f32);

            // Kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
            let m4 = values.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
            self.kurtosis = Some((m4 - 3.0) as f32);
        }

        // Compute target correlation if target provided
        if let Some(target) = target {
            if values.len() == target.len() {
                self.target_correlation = Some(crate::analysis::compute_correlation_mixed(&values, target));
            }
        }
    }

    /// Check if column values are monotonically increasing
    fn check_monotonic(column: &Column) -> bool {
        let Ok(values) = column.cast(&PolarsDataType::Float64) else {
            return false;
        };
        let Ok(ca) = values.f64() else {
            return false;
        };

        let mut prev: Option<f64> = None;
        let mut increasing = true;
        let mut sample_count = 0;
        let max_samples = 1000; // Sample for efficiency

        for opt_val in ca.into_iter() {
            if sample_count >= max_samples {
                break;
            }
            if let Some(val) = opt_val {
                if let Some(p) = prev {
                    if val < p {
                        increasing = false;
                        break;
                    }
                }
                prev = Some(val);
                sample_count += 1;
            }
        }

        increasing && sample_count > 10
    }

    /// Check if this column should be dropped
    pub fn should_drop(&self) -> Option<DropReason> {
        if self.is_constant {
            Some(DropReason::Constant)
        } else if self.is_id_like {
            Some(DropReason::IdLike)
        } else if self.dtype == ColumnDataType::Text {
            Some(DropReason::Text)
        } else {
            None
        }
    }
}

/// Task type detected from target column
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Continuous target variable
    Regression,
    /// Binary classification (2 classes)
    BinaryClassification,
    /// Multi-class classification
    MultiClassification { num_classes: usize },
}

impl TaskType {
    /// Detect task type from target values
    ///
    /// Heuristic:
    /// - 2 unique values → Binary classification
    /// - 3-20 unique values → Multi-class classification
    /// - >20 unique values → Regression
    pub fn detect(target: &[f32]) -> Self {
        // Count unique values (quantize to handle float comparison)
        let unique: HashSet<i64> = target.iter().map(|v| (v * 1000.0) as i64).collect();

        match unique.len() {
            2 => TaskType::BinaryClassification,
            3..=20 => TaskType::MultiClassification {
                num_classes: unique.len(),
            },
            _ => TaskType::Regression,
        }
    }

    /// Get default loss function name for this task type
    pub fn default_loss_name(&self) -> &'static str {
        match self {
            TaskType::Regression => "mse",
            TaskType::BinaryClassification => "logloss",
            TaskType::MultiClassification { .. } => "softmax",
        }
    }

    /// Get default evaluation metric name for this task type
    pub fn default_metric_name(&self) -> &'static str {
        match self {
            TaskType::Regression => "rmse",
            TaskType::BinaryClassification => "auc",
            TaskType::MultiClassification { .. } => "mlogloss",
        }
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskType::Regression => write!(f, "Regression"),
            TaskType::BinaryClassification => write!(f, "Binary Classification"),
            TaskType::MultiClassification { num_classes } => {
                write!(f, "{}-Class Classification", num_classes)
            }
        }
    }
}

/// Complete profile of a DataFrame
#[derive(Debug, Clone)]
pub struct DataFrameProfile {
    /// Profiles for all columns (excluding dropped)
    pub columns: Vec<ColumnProfile>,
    /// Columns marked for dropping
    pub drop_columns: Vec<DroppedColumn>,
    /// Total number of rows
    pub num_rows: usize,
    /// Number of numeric features (after drops)
    pub num_numeric: usize,
    /// Number of categorical features (after drops)
    pub num_categorical: usize,
    /// Profile of target column
    pub target_profile: ColumnProfile,
    /// Detected task type
    pub task_type: TaskType,
    /// Target column name
    pub target_name: String,
}

impl DataFrameProfile {
    /// Analyze a DataFrame with two-pass strategy
    ///
    /// Pass 1: Cheap stats to identify drops (~1ms per column)
    /// Pass 2: Expensive stats only for kept columns
    pub fn analyze(df: &DataFrame, target_col: &str) -> Result<Self> {
        let num_rows = df.height();

        // Get target column and values
        let target_column = df
            .column(target_col)
            .map_err(|e| TreeBoostError::Data(format!("Target column '{}' not found: {}", target_col, e)))?;

        let target_values: Vec<f32> = target_column
            .cast(&PolarsDataType::Float64)
            .map_err(|e| TreeBoostError::Data(format!("Cannot convert target to numeric: {}", e)))?
            .f64()
            .map_err(|e| TreeBoostError::Data(format!("Target column error: {}", e)))?
            .into_iter()
            .map(|opt| opt.map(|v| v as f32).unwrap_or(f32::NAN))
            .collect();

        // Detect task type
        let task_type = TaskType::detect(&target_values);

        // Profile target column
        let mut target_profile = ColumnProfile::analyze_pass1(target_column, num_rows);
        target_profile.analyze_pass2(target_column, None);

        // === Pass 1: Cheap stats for all columns ===
        let mut columns = Vec::new();
        let mut drop_columns = Vec::new();

        for name in df.get_column_names() {
            let name_str = name.to_string();

            // Skip target column
            if name_str == target_col {
                drop_columns.push(DroppedColumn {
                    name: name_str,
                    reason: DropReason::Target,
                });
                continue;
            }

            let column = df.column(name).unwrap();
            let profile = ColumnProfile::analyze_pass1(column, num_rows);

            // Check if should drop
            if let Some(reason) = profile.should_drop() {
                drop_columns.push(DroppedColumn {
                    name: name_str,
                    reason,
                });
            } else {
                columns.push(profile);
            }
        }

        // === Pass 2: Expensive stats for kept columns ===
        for profile in &mut columns {
            let column = df.column(&profile.name).unwrap();
            profile.analyze_pass2(column, Some(&target_values));
        }

        // Count by type
        let num_numeric = columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Numeric)
            .count();
        let num_categorical = columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Categorical)
            .count();

        Ok(Self {
            columns,
            drop_columns,
            num_rows,
            num_numeric,
            num_categorical,
            target_profile,
            task_type,
            target_name: target_col.to_string(),
        })
    }

    /// Get names of columns to keep (not dropped)
    pub fn kept_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|p| p.name.as_str()).collect()
    }

    /// Get names of numeric columns
    pub fn numeric_column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Numeric)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Get names of categorical columns
    pub fn categorical_column_names(&self) -> Vec<&str> {
        self.columns
            .iter()
            .filter(|p| p.dtype == ColumnDataType::Categorical)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Get column profile by name
    pub fn get_column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|p| p.name == name)
    }

    /// Get columns with high skewness (candidates for YeoJohnson)
    pub fn high_skew_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| p.skewness.map(|s| s.abs() > threshold).unwrap_or(false))
            .collect()
    }

    /// Get columns with high missing ratio
    pub fn high_missing_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| p.missing_ratio > threshold)
            .collect()
    }

    /// Get columns strongly correlated with target
    pub fn correlated_columns(&self, threshold: f32) -> Vec<&ColumnProfile> {
        self.columns
            .iter()
            .filter(|p| {
                p.target_correlation
                    .map(|c| c.abs() > threshold)
                    .unwrap_or(false)
            })
            .collect()
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("┌─────────────────────────────────────────────────────────────────┐\n");
        report.push_str("│                    DataFrame Profile                            │\n");
        report.push_str("├─────────────────────────────────────────────────────────────────┤\n");

        // Summary
        report.push_str(&format!(
            "│ Rows: {:>10}    Features: {:>3} ({} numeric, {} categorical)    │\n",
            self.num_rows,
            self.columns.len(),
            self.num_numeric,
            self.num_categorical
        ));
        report.push_str(&format!(
            "│ Target: {} ({})                                     │\n",
            self.target_name, self.task_type
        ));

        // Dropped columns
        if !self.drop_columns.is_empty() {
            report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ Dropped Columns                                                 │\n");
            for col in &self.drop_columns {
                if col.reason != DropReason::Target {
                    report.push_str(&format!("│   • {} ({})                              │\n", col.name, col.reason));
                }
            }
        }

        // High skewness warning
        let skewed = self.high_skew_columns(2.0);
        if !skewed.is_empty() {
            report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ High Skewness (consider YeoJohnson)                            │\n");
            for col in skewed.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} (skew={:.2})                                     │\n",
                    col.name,
                    col.skewness.unwrap_or(0.0)
                ));
            }
        }

        // High missing warning
        let missing = self.high_missing_columns(0.05);
        if !missing.is_empty() {
            report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ High Missing Values (>5%)                                      │\n");
            for col in missing.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} ({:.1}% missing)                                 │\n",
                    col.name,
                    col.missing_ratio * 100.0
                ));
            }
        }

        // Top correlated features
        let mut correlated: Vec<_> = self
            .columns
            .iter()
            .filter(|p| p.target_correlation.is_some())
            .collect();
        correlated.sort_by(|a, b| {
            b.target_correlation
                .unwrap_or(0.0)
                .abs()
                .partial_cmp(&a.target_correlation.unwrap_or(0.0).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !correlated.is_empty() {
            report.push_str("├─────────────────────────────────────────────────────────────────┤\n");
            report.push_str("│ Top Correlated Features                                        │\n");
            for col in correlated.iter().take(5) {
                report.push_str(&format!(
                    "│   • {} (r={:.3})                                       │\n",
                    col.name,
                    col.target_correlation.unwrap_or(0.0)
                ));
            }
        }

        report.push_str("└─────────────────────────────────────────────────────────────────┘\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_detect_numeric() {
        let col: Column = Series::new("test".into(), vec![1.0f64, 2.0, 3.0]).into();
        assert_eq!(ColumnDataType::detect(&col, 3), ColumnDataType::Numeric);
    }

    #[test]
    fn test_data_type_detect_categorical() {
        let col: Column = Series::new("test".into(), vec!["a", "b", "c", "a", "b"]).into();
        assert_eq!(ColumnDataType::detect(&col, 5), ColumnDataType::Categorical);
    }

    #[test]
    fn test_task_type_regression() {
        let target = vec![1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.1, 8.9, 9.0, 10.1];
        let mut more_values = target.clone();
        for i in 11..100 {
            more_values.push(i as f32 * 0.1);
        }
        assert_eq!(TaskType::detect(&more_values), TaskType::Regression);
    }

    #[test]
    fn test_task_type_binary() {
        let target = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(TaskType::detect(&target), TaskType::BinaryClassification);
    }

    #[test]
    fn test_task_type_multiclass() {
        let target = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            TaskType::detect(&target),
            TaskType::MultiClassification { num_classes: 5 }
        );
    }

    #[test]
    fn test_column_profile_constant() {
        let col: Column = Series::new("test".into(), vec![5.0f64, 5.0, 5.0, 5.0]).into();
        let profile = ColumnProfile::analyze_pass1(&col, 4);
        assert!(profile.is_constant);
        assert_eq!(profile.should_drop(), Some(DropReason::Constant));
    }

    #[test]
    fn test_column_profile_id_like() {
        let col: Column = Series::new("id".into(), vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]).into();
        let profile = ColumnProfile::analyze_pass1(&col, 10);
        // High cardinality ratio + monotonic = ID-like
        assert!(profile.cardinality_ratio > 0.9);
    }

    #[test]
    fn test_dataframe_profile() {
        let df = DataFrame::new(vec![
            Series::new("feature1".into(), vec![1.0f64, 2.0, 3.0, 4.0, 5.0]).into(),
            Series::new("feature2".into(), vec!["a", "b", "a", "b", "a"]).into(),
            Series::new("constant".into(), vec![1.0f64, 1.0, 1.0, 1.0, 1.0]).into(),
            Series::new("target".into(), vec![0.0f64, 1.0, 0.0, 1.0, 0.0]).into(),
        ])
        .unwrap();

        let profile = DataFrameProfile::analyze(&df, "target").unwrap();

        // Should have 2 features (constant dropped, target excluded)
        assert_eq!(profile.columns.len(), 2);
        assert_eq!(profile.num_numeric, 1);
        assert_eq!(profile.num_categorical, 1);

        // Task type should be binary classification
        assert_eq!(profile.task_type, TaskType::BinaryClassification);

        // Constant column should be dropped
        assert!(profile
            .drop_columns
            .iter()
            .any(|d| d.name == "constant" && d.reason == DropReason::Constant));
    }

    #[test]
    fn test_pearson_correlation() {
        use crate::analysis::compute_correlation_mixed;

        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
        let r = compute_correlation_mixed(&x, &y);
        assert!((r - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = vec![10.0f32, 8.0, 6.0, 4.0, 2.0];
        let r_neg = compute_correlation_mixed(&x, &y_neg);
        assert!((r_neg + 1.0).abs() < 0.001);
    }
}
