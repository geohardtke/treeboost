//! Time-series feature generation for GBDT models
//!
//! This module provides specialized feature generators for time-series data:
//!
//! - [`LagGenerator`] - Create lagged versions of features (x_{t-1}, x_{t-2}, etc.)
//! - [`RollingGenerator`] - Compute rolling statistics (mean, std, min, max, sum, ewma)
//! - [`SeasonalGenerator`] - Extract seasonal components from timestamps
//!
//! # When to Use
//!
//! | Generator | Use Case |
//! |-----------|----------|
//! | `LagGenerator` | Autoregressive patterns, momentum features |
//! | `RollingGenerator` | Smoothing, trend detection, volatility |
//! | `SeasonalGenerator` | Day-of-week effects, hourly patterns |
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{LagGenerator, RollingGenerator, RollingStat};
//!
//! // Create lag features
//! let lag_gen = LagGenerator::new(vec![1, 7, 14]); // t-1, t-7, t-14
//! let lagged = lag_gen.transform(&data, num_features);
//!
//! // Create rolling features
//! let roll_gen = RollingGenerator::new(7)
//!     .with_stats(vec![RollingStat::Mean, RollingStat::Std]);
//! let rolled = roll_gen.transform(&data, num_features);
//! ```
//!
//! # Design Philosophy
//!
//! These generators are optimized for **tree-based models**:
//! - NaN values at series boundaries (trees handle NaN natively)
//! - No normalization applied (trees are scale-invariant)
//! - Efficient single-pass algorithms where possible

use crate::{Result, TreeBoostError};

/// Strategy for handling NaN values at series boundaries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum NaNStrategy {
    /// Keep NaN values (default, trees handle this well)
    #[default]
    Keep,
    /// Forward fill from last valid value
    ForwardFill,
    /// Fill with a constant value
    Constant(i32), // Using i32 to allow f32::from_bits for actual constant
}

impl NaNStrategy {
    /// Create a constant fill strategy
    pub fn constant(value: f32) -> Self {
        Self::Constant(value.to_bits() as i32)
    }

    /// Get the constant value (if applicable)
    fn get_constant(&self) -> Option<f32> {
        match self {
            Self::Constant(bits) => Some(f32::from_bits(*bits as u32)),
            _ => None,
        }
    }

    /// Apply the NaN strategy to produce a fallback value
    ///
    /// # Arguments
    /// * `fallback` - Value to use for ForwardFill strategy
    ///
    /// # Returns
    /// The value to use based on the strategy
    pub fn apply(&self, fallback: f32) -> f32 {
        match self {
            Self::Keep => f32::NAN,
            Self::ForwardFill => fallback,
            Self::Constant(_) => self.get_constant().unwrap_or(0.0),
        }
    }
}

// ============================================================================
// Lag Generator
// ============================================================================

/// Generates lagged features from time-series data
///
/// Creates new features representing past values: x_{t-1}, x_{t-2}, ..., x_{t-k}
///
/// # Example
///
/// ```ignore
/// let gen = LagGenerator::new(vec![1, 7]); // Create t-1 and t-7 lags
/// let lagged = gen.transform(&data, num_features)?;
/// // Result has original features + lag features
/// // For 3 features with lags [1, 7]: output has 3 + 3*2 = 9 features
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LagGenerator {
    /// Lag periods to generate (e.g., [1, 2, 7] for t-1, t-2, t-7)
    lags: Vec<usize>,
    /// How to handle NaN values at series start
    nan_strategy: NaNStrategy,
    /// Optional feature names (for output naming)
    feature_names: Option<Vec<String>>,
}

impl LagGenerator {
    /// Create a new lag generator with specified lag periods
    ///
    /// # Arguments
    /// * `lags` - Lag periods to generate (e.g., `vec![1, 7, 14]`)
    pub fn new(lags: Vec<usize>) -> Self {
        Self {
            lags,
            nan_strategy: NaNStrategy::default(),
            feature_names: None,
        }
    }

    /// Create with a range of lags
    ///
    /// # Arguments
    /// * `max_lag` - Maximum lag (creates lags 1..=max_lag)
    pub fn range(max_lag: usize) -> Self {
        Self::new((1..=max_lag).collect())
    }

    /// Set the NaN handling strategy
    pub fn with_nan_strategy(mut self, strategy: NaNStrategy) -> Self {
        self.nan_strategy = strategy;
        self
    }

    /// Set feature names for output column naming
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get the number of lag features that will be created per input feature
    pub fn num_lags(&self) -> usize {
        self.lags.len()
    }

    /// Get output feature names
    pub fn output_names(&self) -> Vec<String> {
        let base_names: Vec<String> = self
            .feature_names
            .clone()
            .unwrap_or_else(|| vec!["feature".to_string()]);

        let mut names = Vec::new();
        for base in &base_names {
            for lag in &self.lags {
                names.push(format!("{}_lag_{}", base, lag));
            }
        }
        names
    }

    /// Transform data by adding lag features
    ///
    /// # Arguments
    /// * `data` - Row-major data matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// New data with original + lagged features (num_rows × (num_features + num_features * num_lags))
    pub fn transform(&self, data: &[f32], num_features: usize) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let num_rows = data.len() / num_features;
        if num_rows * num_features != data.len() {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_new_features = num_features * self.lags.len();
        let total_features = num_features + num_new_features;
        let mut result = vec![f32::NAN; num_rows * total_features];

        for row in 0..num_rows {
            // Copy original features
            let src_start = row * num_features;
            let dst_start = row * total_features;
            result[dst_start..dst_start + num_features]
                .copy_from_slice(&data[src_start..src_start + num_features]);

            // Generate lag features
            let mut lag_offset = num_features;
            for &lag in &self.lags {
                for feat in 0..num_features {
                    let dst_idx = dst_start + lag_offset + feat;
                    if row >= lag {
                        // We have enough history
                        let src_row = row - lag;
                        let src_idx = src_row * num_features + feat;
                        result[dst_idx] = data[src_idx];
                    } else {
                        // Not enough history - apply NaN strategy
                        result[dst_idx] = self.nan_strategy.apply(data[feat]);
                    }
                }
                lag_offset += num_features;
            }
        }

        Ok(result)
    }

    /// Transform a single feature column (column-major)
    ///
    /// # Arguments
    /// * `column` - Single feature column (length = num_rows)
    ///
    /// # Returns
    /// Lagged columns concatenated (length = num_rows * num_lags)
    pub fn transform_column(&self, column: &[f32]) -> Vec<f32> {
        let num_rows = column.len();
        let mut result = vec![f32::NAN; num_rows * self.lags.len()];

        for (lag_idx, &lag) in self.lags.iter().enumerate() {
            let offset = lag_idx * num_rows;
            for row in 0..num_rows {
                if row >= lag {
                    result[offset + row] = column[row - lag];
                } else {
                    result[offset + row] = self.nan_strategy.apply(column[0]);
                }
            }
        }

        result
    }
}

// ============================================================================
// Rolling Generator
// ============================================================================

/// Statistics to compute over rolling windows
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RollingStat {
    /// Rolling mean (average)
    Mean,
    /// Rolling standard deviation
    Std,
    /// Rolling minimum
    Min,
    /// Rolling maximum
    Max,
    /// Rolling sum
    Sum,
    /// Rolling count of non-NaN values
    Count,
    /// Rolling median (requires sorting)
    Median,
    /// Rolling variance
    Var,
}

impl RollingStat {
    /// Get the suffix for naming output features
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Std => "std",
            Self::Min => "min",
            Self::Max => "max",
            Self::Sum => "sum",
            Self::Count => "count",
            Self::Median => "median",
            Self::Var => "var",
        }
    }
}

/// Generates rolling window statistics from time-series data
///
/// Computes statistics (mean, std, min, max, etc.) over a sliding window.
///
/// # Example
///
/// ```ignore
/// let gen = RollingGenerator::new(7)
///     .with_stats(vec![RollingStat::Mean, RollingStat::Std]);
/// let rolled = gen.transform(&data, num_features)?;
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RollingGenerator {
    /// Window size for rolling calculations
    window: usize,
    /// Statistics to compute
    stats: Vec<RollingStat>,
    /// Minimum number of observations required
    min_periods: usize,
    /// Center the window (vs right-aligned)
    center: bool,
    /// Optional feature names
    feature_names: Option<Vec<String>>,
}

impl RollingGenerator {
    /// Create a new rolling generator with specified window size
    ///
    /// # Arguments
    /// * `window` - Number of periods in the rolling window
    pub fn new(window: usize) -> Self {
        Self {
            window,
            stats: vec![RollingStat::Mean],
            min_periods: 1,
            center: false,
            feature_names: None,
        }
    }

    /// Set statistics to compute
    pub fn with_stats(mut self, stats: Vec<RollingStat>) -> Self {
        self.stats = stats;
        self
    }

    /// Set minimum periods required for calculation
    pub fn with_min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }

    /// Center the window (default is right-aligned)
    pub fn centered(mut self) -> Self {
        self.center = true;
        self
    }

    /// Set feature names for output column naming
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get output feature names
    pub fn output_names(&self) -> Vec<String> {
        let base_names: Vec<String> = self
            .feature_names
            .clone()
            .unwrap_or_else(|| vec!["feature".to_string()]);

        let mut names = Vec::new();
        for base in &base_names {
            for stat in &self.stats {
                names.push(format!("{}_roll_{}_{}", base, self.window, stat.suffix()));
            }
        }
        names
    }

    /// Transform data by adding rolling features
    ///
    /// # Arguments
    /// * `data` - Row-major data matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// New data with original + rolling features
    pub fn transform(&self, data: &[f32], num_features: usize) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let num_rows = data.len() / num_features;
        if num_rows * num_features != data.len() {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_new_features = num_features * self.stats.len();
        let total_features = num_features + num_new_features;
        let mut result = vec![f32::NAN; num_rows * total_features];

        // For each row
        for row in 0..num_rows {
            // Copy original features
            let src_start = row * num_features;
            let dst_start = row * total_features;
            result[dst_start..dst_start + num_features]
                .copy_from_slice(&data[src_start..src_start + num_features]);

            // Compute rolling stats for each feature
            let mut stat_offset = num_features;
            for stat in &self.stats {
                for feat in 0..num_features {
                    let dst_idx = dst_start + stat_offset + feat;

                    // Determine window bounds
                    let (start_row, end_row) = if self.center {
                        let half = self.window / 2;
                        let start = row.saturating_sub(half);
                        let end = (row + half + 1).min(num_rows);
                        (start, end)
                    } else {
                        // Right-aligned (lookback only)
                        let start = row.saturating_sub(self.window - 1);
                        (start, row + 1)
                    };

                    // Collect window values
                    let mut window_vals: Vec<f32> = Vec::with_capacity(end_row - start_row);
                    for r in start_row..end_row {
                        let val = data[r * num_features + feat];
                        if !val.is_nan() {
                            window_vals.push(val);
                        }
                    }

                    // Check min_periods
                    if window_vals.len() < self.min_periods {
                        result[dst_idx] = f32::NAN;
                        continue;
                    }

                    // Compute statistic
                    result[dst_idx] = self.compute_stat(*stat, &window_vals);
                }
                stat_offset += num_features;
            }
        }

        Ok(result)
    }

    /// Compute a single statistic over a window
    fn compute_stat(&self, stat: RollingStat, values: &[f32]) -> f32 {
        if values.is_empty() {
            return f32::NAN;
        }

        match stat {
            RollingStat::Mean => {
                let sum: f32 = values.iter().sum();
                sum / values.len() as f32
            }
            RollingStat::Std => {
                if values.len() < 2 {
                    return f32::NAN;
                }
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / (values.len() - 1) as f32;
                variance.sqrt()
            }
            RollingStat::Var => {
                if values.len() < 2 {
                    return f32::NAN;
                }
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32
            }
            RollingStat::Min => values.iter().cloned().fold(f32::INFINITY, f32::min),
            RollingStat::Max => values.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            RollingStat::Sum => values.iter().sum(),
            RollingStat::Count => values.len() as f32,
            RollingStat::Median => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len().is_multiple_of(2) {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                }
            }
        }
    }

    /// Transform a single feature column (column-major)
    pub fn transform_column(&self, column: &[f32]) -> Vec<f32> {
        let num_rows = column.len();
        let mut result = vec![f32::NAN; num_rows * self.stats.len()];

        for (stat_idx, stat) in self.stats.iter().enumerate() {
            let offset = stat_idx * num_rows;
            for row in 0..num_rows {
                // Determine window bounds (right-aligned)
                let start_row = row.saturating_sub(self.window - 1);
                let end_row = row + 1;

                // Collect window values
                let mut window_vals: Vec<f32> = Vec::with_capacity(end_row - start_row);
                for &val in &column[start_row..end_row] {
                    if !val.is_nan() {
                        window_vals.push(val);
                    }
                }

                // Check min_periods
                if window_vals.len() < self.min_periods {
                    result[offset + row] = f32::NAN;
                    continue;
                }

                result[offset + row] = self.compute_stat(*stat, &window_vals);
            }
        }

        result
    }
}

// ============================================================================
// Exponential Weighted Moving Average
// ============================================================================

/// Generates Exponentially Weighted Moving Average (EWMA) features
///
/// EWMA gives more weight to recent observations, useful for trend detection.
///
/// Formula: EWMA_t = α × x_t + (1 - α) × EWMA_{t-1}
///
/// # Example
///
/// ```ignore
/// let ewma = EwmaGenerator::new(0.3); // alpha = 0.3
/// let smoothed = ewma.transform(&data, num_features)?;
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EwmaGenerator {
    /// Smoothing factor (0 < alpha ≤ 1)
    alpha: f32,
    /// Optional feature names
    feature_names: Option<Vec<String>>,
    /// Adjust for bias in early observations
    adjust: bool,
}

impl EwmaGenerator {
    /// Create a new EWMA generator with specified alpha
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0 < alpha ≤ 1). Higher = more weight on recent.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "Alpha must be in (0, 1]");
        Self {
            alpha,
            feature_names: None,
            adjust: true,
        }
    }

    /// Create from span (common alternative parameterization)
    ///
    /// alpha = 2 / (span + 1)
    pub fn from_span(span: usize) -> Self {
        assert!(span >= 1, "Span must be >= 1");
        Self::new(2.0 / (span as f32 + 1.0))
    }

    /// Create from halflife (time for weight to decay by half)
    ///
    /// alpha = 1 - exp(ln(0.5) / halflife)
    pub fn from_halflife(halflife: f32) -> Self {
        assert!(halflife > 0.0, "Halflife must be > 0");
        let alpha = 1.0 - (0.5_f32.ln() / halflife).exp();
        Self::new(alpha)
    }

    /// Disable bias adjustment
    pub fn without_adjust(mut self) -> Self {
        self.adjust = false;
        self
    }

    /// Set feature names for output column naming
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Transform data by computing EWMA
    ///
    /// # Arguments
    /// * `data` - Row-major data matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// EWMA features (same shape as input)
    pub fn transform(&self, data: &[f32], num_features: usize) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let num_rows = data.len() / num_features;
        if num_rows * num_features != data.len() {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let mut result = vec![f32::NAN; data.len()];
        let mut ewma = vec![0.0f32; num_features];
        let mut sum_weights = vec![0.0f32; num_features];

        for row in 0..num_rows {
            let row_start = row * num_features;

            for feat in 0..num_features {
                let val = data[row_start + feat];

                if val.is_nan() {
                    // Propagate last EWMA value
                    result[row_start + feat] = if row > 0 {
                        result[(row - 1) * num_features + feat]
                    } else {
                        f32::NAN
                    };
                    continue;
                }

                if row == 0 || ewma[feat] == 0.0 && sum_weights[feat] == 0.0 {
                    // Initialize
                    ewma[feat] = val;
                    sum_weights[feat] = 1.0;
                } else {
                    // Update EWMA
                    ewma[feat] = self.alpha * val + (1.0 - self.alpha) * ewma[feat];
                    sum_weights[feat] = self.alpha + (1.0 - self.alpha) * sum_weights[feat];
                }

                // Apply bias adjustment if enabled
                result[row_start + feat] = if self.adjust {
                    ewma[feat] / sum_weights[feat]
                } else {
                    ewma[feat]
                };
            }
        }

        Ok(result)
    }

    /// Transform a single column
    pub fn transform_column(&self, column: &[f32]) -> Vec<f32> {
        let mut result = vec![f32::NAN; column.len()];
        let mut ewma = 0.0f32;
        let mut sum_weights = 0.0f32;

        for (i, &val) in column.iter().enumerate() {
            if val.is_nan() {
                result[i] = if i > 0 { result[i - 1] } else { f32::NAN };
                continue;
            }

            if i == 0 || (ewma == 0.0 && sum_weights == 0.0) {
                ewma = val;
                sum_weights = 1.0;
            } else {
                ewma = self.alpha * val + (1.0 - self.alpha) * ewma;
                sum_weights = self.alpha + (1.0 - self.alpha) * sum_weights;
            }

            result[i] = if self.adjust {
                ewma / sum_weights
            } else {
                ewma
            };
        }

        result
    }
}

// ============================================================================
// Momentum Generator
// ============================================================================

/// Minimum absolute value for denominator in momentum calculation to prevent
/// division-by-near-zero numerical artifacts (e.g., momentum = (x_t - x_{t-k}) / x_{t-k})
const MIN_DENOMINATOR_VALUE: f32 = 1e-10;

/// Generates momentum/return features from time-series data
///
/// Computes percentage changes over different periods:
/// momentum_t = (x_t - x_{t-lag}) / x_{t-lag}
///
/// This is fundamental for financial time-series (stock returns, etc.)
/// where relative changes are more predictive than absolute values.
///
/// # Example
///
/// ```ignore
/// let gen = MomentumGenerator::new(vec![1, 7, 14]); // 1-day, 7-day, 14-day returns
/// let momentum = gen.transform(&data, num_features)?;
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MomentumGenerator {
    /// Lag periods for computing momentum (e.g., [1, 7, 14])
    lags: Vec<usize>,
    /// Optional feature names
    feature_names: Option<Vec<String>>,
}

impl MomentumGenerator {
    /// Create a new momentum generator with specified lag periods
    ///
    /// # Arguments
    /// * `lags` - Lag periods for momentum calculation (e.g., `vec![1, 7, 14]`)
    pub fn new(lags: Vec<usize>) -> Self {
        Self {
            lags,
            feature_names: None,
        }
    }

    /// Set feature names for output column naming
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get output feature names
    pub fn output_names(&self) -> Vec<String> {
        let base_names: Vec<String> = self
            .feature_names
            .clone()
            .unwrap_or_else(|| vec!["feature".to_string()]);

        let mut names = Vec::new();
        for base in &base_names {
            for lag in &self.lags {
                names.push(format!("{}_momentum_{}", base, lag));
            }
        }
        names
    }

    /// Transform data by adding momentum features
    ///
    /// # Arguments
    /// * `data` - Row-major data matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// New data with original + momentum features (num_rows × (num_features + num_features * num_lags))
    pub fn transform(&self, data: &[f32], num_features: usize) -> Result<Vec<f32>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let num_rows = data.len() / num_features;
        if num_rows * num_features != data.len() {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let num_new_features = num_features * self.lags.len();
        let total_features = num_features + num_new_features;
        let mut result = vec![f32::NAN; num_rows * total_features];

        for row in 0..num_rows {
            // Copy original features
            let src_start = row * num_features;
            let dst_start = row * total_features;
            result[dst_start..dst_start + num_features]
                .copy_from_slice(&data[src_start..src_start + num_features]);

            // Generate momentum features
            let mut momentum_offset = num_features;
            for &lag in &self.lags {
                for feat in 0..num_features {
                    let dst_idx = dst_start + momentum_offset + feat;
                    if row >= lag {
                        // We have enough history
                        let current = data[src_start + feat];
                        let lagged = data[(row - lag) * num_features + feat];

                        // Compute percentage change: (current - lagged) / lagged
                        if !current.is_nan()
                            && !lagged.is_nan()
                            && lagged.abs() > MIN_DENOMINATOR_VALUE
                        {
                            result[dst_idx] = (current - lagged) / lagged;
                        } else {
                            result[dst_idx] = f32::NAN;
                        }
                    } else {
                        // Not enough history
                        result[dst_idx] = f32::NAN;
                    }
                }
                momentum_offset += num_features;
            }
        }

        Ok(result)
    }

    /// Transform a single feature column (column-major)
    ///
    /// # Arguments
    /// * `column` - Single feature column (length = num_rows)
    ///
    /// # Returns
    /// Momentum columns concatenated (length = num_rows * num_lags)
    pub fn transform_column(&self, column: &[f32]) -> Vec<f32> {
        let num_rows = column.len();
        let mut result = vec![f32::NAN; num_rows * self.lags.len()];

        for (lag_idx, &lag) in self.lags.iter().enumerate() {
            let offset = lag_idx * num_rows;
            for row in 0..num_rows {
                if row >= lag {
                    let current = column[row];
                    let lagged = column[row - lag];

                    if !current.is_nan() && !lagged.is_nan() && lagged.abs() > MIN_DENOMINATOR_VALUE
                    {
                        result[offset + row] = (current - lagged) / lagged;
                    } else {
                        result[offset + row] = f32::NAN;
                    }
                } else {
                    result[offset + row] = f32::NAN;
                }
            }
        }

        result
    }
}

// ============================================================================
// Seasonal Generator
// ============================================================================

/// Components to extract from timestamps
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SeasonalComponent {
    /// Hour of day (0-23)
    Hour,
    /// Day of week (0=Monday, 6=Sunday)
    DayOfWeek,
    /// Day of month (1-31)
    DayOfMonth,
    /// Day of year (1-366)
    DayOfYear,
    /// Week of year (1-53)
    WeekOfYear,
    /// Month (1-12)
    Month,
    /// Quarter (1-4)
    Quarter,
    /// Year
    Year,
    /// Is weekend (0/1)
    IsWeekend,
}

impl SeasonalComponent {
    /// Get the suffix for naming output features
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Hour => "hour",
            Self::DayOfWeek => "dow",
            Self::DayOfMonth => "dom",
            Self::DayOfYear => "doy",
            Self::WeekOfYear => "woy",
            Self::Month => "month",
            Self::Quarter => "quarter",
            Self::Year => "year",
            Self::IsWeekend => "is_weekend",
        }
    }

    /// Get the maximum value for this component (for cyclical encoding)
    pub fn max_value(&self) -> f32 {
        match self {
            Self::Hour => 24.0,
            Self::DayOfWeek => 7.0,
            Self::DayOfMonth => 31.0,
            Self::DayOfYear => 366.0,
            Self::WeekOfYear => 53.0,
            Self::Month => 12.0,
            Self::Quarter => 4.0,
            Self::Year => 1.0,      // Not cyclical
            Self::IsWeekend => 1.0, // Binary
        }
    }

    /// Whether this component is cyclical
    pub fn is_cyclical(&self) -> bool {
        !matches!(self, Self::Year | Self::IsWeekend)
    }
}

/// Generates seasonal features from Unix timestamps
///
/// Extracts calendar components and optionally encodes them as cyclical (sin/cos).
///
/// # Example
///
/// ```ignore
/// let gen = SeasonalGenerator::new(vec![
///     SeasonalComponent::Hour,
///     SeasonalComponent::DayOfWeek,
///     SeasonalComponent::Month,
/// ])
/// .with_cyclical(true);
///
/// let features = gen.transform_timestamps(&timestamps)?;
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SeasonalGenerator {
    /// Components to extract
    components: Vec<SeasonalComponent>,
    /// Use cyclical encoding (sin/cos) for periodic components
    cyclical: bool,
}

impl SeasonalGenerator {
    /// Create a new seasonal generator
    pub fn new(components: Vec<SeasonalComponent>) -> Self {
        Self {
            components,
            cyclical: false,
        }
    }

    /// Create with common datetime components
    pub fn datetime() -> Self {
        Self::new(vec![
            SeasonalComponent::Hour,
            SeasonalComponent::DayOfWeek,
            SeasonalComponent::DayOfMonth,
            SeasonalComponent::Month,
        ])
    }

    /// Create with date-only components
    pub fn date_only() -> Self {
        Self::new(vec![
            SeasonalComponent::DayOfWeek,
            SeasonalComponent::DayOfMonth,
            SeasonalComponent::Month,
            SeasonalComponent::Quarter,
        ])
    }

    /// Enable cyclical encoding (sin/cos) for periodic components
    pub fn with_cyclical(mut self, cyclical: bool) -> Self {
        self.cyclical = cyclical;
        self
    }

    /// Get output feature names
    pub fn output_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        for comp in &self.components {
            if self.cyclical && comp.is_cyclical() {
                names.push(format!("{}_{}_sin", prefix, comp.suffix()));
                names.push(format!("{}_{}_cos", prefix, comp.suffix()));
            } else {
                names.push(format!("{}_{}", prefix, comp.suffix()));
            }
        }
        names
    }

    /// Get number of output features per timestamp
    pub fn num_features(&self) -> usize {
        self.components
            .iter()
            .map(|c| {
                if self.cyclical && c.is_cyclical() {
                    2
                } else {
                    1
                }
            })
            .sum()
    }

    /// Transform Unix timestamps (seconds since epoch) to seasonal features
    ///
    /// # Arguments
    /// * `timestamps` - Unix timestamps (f64 for sub-second precision)
    ///
    /// # Returns
    /// Seasonal features (row-major: num_timestamps × num_features)
    pub fn transform_timestamps(&self, timestamps: &[f64]) -> Vec<f32> {
        use std::f64::consts::PI;

        let num_features = self.num_features();
        let mut result = Vec::with_capacity(timestamps.len() * num_features);

        for &ts in timestamps {
            // Convert timestamp to datetime components
            // Using manual calculation (no external datetime crate)
            let secs = ts as i64;

            // Days since Unix epoch (1970-01-01)
            let days = secs / 86400;
            let time_of_day = (secs % 86400 + 86400) % 86400; // Handle negative

            // Time components
            let hour = (time_of_day / 3600) as f32;
            let _minute = ((time_of_day % 3600) / 60) as f32;

            // Day of week (1970-01-01 was Thursday = 3)
            let day_of_week = ((days % 7 + 3 + 7) % 7) as f32; // 0=Mon, 6=Sun

            // Convert days to year/month/day using a simplified algorithm
            let (year, month, day_of_month, day_of_year) = days_to_ymd(days);

            // Week of year (ISO week approximation)
            let week_of_year = (day_of_year / 7 + 1).min(53) as f32;

            // Quarter
            let quarter = ((month - 1) / 3 + 1) as f32;

            // Is weekend
            let is_weekend = if day_of_week >= 5.0 { 1.0 } else { 0.0 };

            for comp in &self.components {
                let value = match comp {
                    SeasonalComponent::Hour => hour,
                    SeasonalComponent::DayOfWeek => day_of_week,
                    SeasonalComponent::DayOfMonth => day_of_month as f32,
                    SeasonalComponent::DayOfYear => day_of_year as f32,
                    SeasonalComponent::WeekOfYear => week_of_year,
                    SeasonalComponent::Month => month as f32,
                    SeasonalComponent::Quarter => quarter,
                    SeasonalComponent::Year => year as f32,
                    SeasonalComponent::IsWeekend => is_weekend,
                };

                if self.cyclical && comp.is_cyclical() {
                    // Cyclical encoding: sin and cos
                    let max = comp.max_value() as f64;
                    let angle = 2.0 * PI * (value as f64) / max;
                    result.push(angle.sin() as f32);
                    result.push(angle.cos() as f32);
                } else {
                    result.push(value);
                }
            }
        }

        result
    }
}

/// Convert days since Unix epoch to (year, month, day_of_month, day_of_year)
fn days_to_ymd(days: i64) -> (i32, i32, i32, i32) {
    // Algorithm based on Howard Hinnant's date algorithms
    // http://howardhinnant.github.io/date_algorithms.html

    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32; // day of era
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = (y + (m <= 2) as i64) as i32;

    // Calculate actual day of year
    let is_leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let day_of_year = month_days[m as usize - 1] + d as i32 + if m > 2 && is_leap { 1 } else { 0 };

    (year, m as i32, d as i32, day_of_year)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // LagGenerator Tests
    // ========================================

    #[test]
    fn test_lag_generator_basic() {
        let gen = LagGenerator::new(vec![1, 2]);

        // 5 rows × 2 features
        let data = vec![
            1.0, 10.0, // row 0
            2.0, 20.0, // row 1
            3.0, 30.0, // row 2
            4.0, 40.0, // row 3
            5.0, 50.0, // row 4
        ];

        let result = gen.transform(&data, 2).unwrap();

        // Output: 5 rows × 6 features (2 original + 2 lag1 + 2 lag2)
        assert_eq!(result.len(), 30);

        // Row 0: original [1, 10], lag1 [NaN, NaN], lag2 [NaN, NaN]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 10.0);
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        assert!(result[4].is_nan());
        assert!(result[5].is_nan());

        // Row 2: original [3, 30], lag1 [2, 20], lag2 [1, 10]
        let row2_start = 2 * 6;
        assert_eq!(result[row2_start], 3.0);
        assert_eq!(result[row2_start + 1], 30.0);
        assert_eq!(result[row2_start + 2], 2.0); // lag1 feat0
        assert_eq!(result[row2_start + 3], 20.0); // lag1 feat1
        assert_eq!(result[row2_start + 4], 1.0); // lag2 feat0
        assert_eq!(result[row2_start + 5], 10.0); // lag2 feat1
    }

    #[test]
    fn test_lag_generator_range() {
        let gen = LagGenerator::range(3);
        assert_eq!(gen.lags, vec![1, 2, 3]);
    }

    #[test]
    fn test_lag_generator_forward_fill() {
        let gen = LagGenerator::new(vec![1, 2]).with_nan_strategy(NaNStrategy::ForwardFill);

        let data = vec![5.0, 6.0, 7.0, 8.0];
        let result = gen.transform(&data, 1).unwrap();

        // With forward fill, row 0 lag1 should be 5.0 (first value)
        assert_eq!(result[1], 5.0); // lag1 at row 0
        assert_eq!(result[2], 5.0); // lag2 at row 0
    }

    #[test]
    fn test_lag_generator_column() {
        let gen = LagGenerator::new(vec![1, 2]);
        let column = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = gen.transform_column(&column);

        // Output: 5 × 2 lags = 10 values
        assert_eq!(result.len(), 10);

        // Lag 1
        assert!(result[0].is_nan());
        assert_eq!(result[1], 10.0);
        assert_eq!(result[2], 20.0);
        assert_eq!(result[3], 30.0);
        assert_eq!(result[4], 40.0);

        // Lag 2
        assert!(result[5].is_nan());
        assert!(result[6].is_nan());
        assert_eq!(result[7], 10.0);
        assert_eq!(result[8], 20.0);
        assert_eq!(result[9], 30.0);
    }

    // ========================================
    // RollingGenerator Tests
    // ========================================

    #[test]
    fn test_rolling_mean() {
        let gen = RollingGenerator::new(3).with_stats(vec![RollingStat::Mean]);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = gen.transform(&data, 1).unwrap();

        // Output: 5 rows × 2 features (1 original + 1 rolling mean)
        assert_eq!(result.len(), 10);

        // Row 0: window=[1], mean=1
        assert_eq!(result[0], 1.0); // original
        assert_eq!(result[1], 1.0); // rolling mean

        // Row 2: window=[1,2,3], mean=2
        assert_eq!(result[4], 3.0); // original
        assert_eq!(result[5], 2.0); // rolling mean

        // Row 4: window=[3,4,5], mean=4
        assert_eq!(result[8], 5.0); // original
        assert_eq!(result[9], 4.0); // rolling mean
    }

    #[test]
    fn test_rolling_multiple_stats() {
        let gen = RollingGenerator::new(3).with_stats(vec![
            RollingStat::Min,
            RollingStat::Max,
            RollingStat::Sum,
        ]);

        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = gen.transform(&data, 1).unwrap();

        // Output: 5 rows × 4 features (1 original + 3 stats)
        assert_eq!(result.len(), 20);

        // Row 4: window=[3,7,2]
        let row4_start = 4 * 4;
        assert_eq!(result[row4_start], 2.0); // original
        assert_eq!(result[row4_start + 1], 2.0); // min
        assert_eq!(result[row4_start + 2], 7.0); // max
        assert_eq!(result[row4_start + 3], 12.0); // sum
    }

    #[test]
    fn test_rolling_min_periods() {
        let gen = RollingGenerator::new(3)
            .with_stats(vec![RollingStat::Mean])
            .with_min_periods(3);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = gen.transform(&data, 1).unwrap();

        // First 2 rows should be NaN (not enough observations)
        assert!(result[1].is_nan()); // row 0 rolling mean
        assert!(result[3].is_nan()); // row 1 rolling mean
        assert!(!result[5].is_nan()); // row 2 rolling mean (has 3 observations)
    }

    #[test]
    fn test_rolling_std() {
        let gen = RollingGenerator::new(3).with_stats(vec![RollingStat::Std]);

        let data = vec![1.0, 2.0, 3.0];
        let result = gen.transform(&data, 1).unwrap();

        // Row 2: window=[1,2,3], std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / 2) = 1.0
        assert!((result[5] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rolling_median() {
        let gen = RollingGenerator::new(3).with_stats(vec![RollingStat::Median]);

        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = gen.transform(&data, 1).unwrap();

        // Row 2: window=[1,5,3], median=3
        assert_eq!(result[5], 3.0);

        // Row 4: window=[3,7,2], sorted=[2,3,7], median=3
        assert_eq!(result[9], 3.0);
    }

    // ========================================
    // EwmaGenerator Tests
    // ========================================

    #[test]
    fn test_ewma_basic() {
        let gen = EwmaGenerator::new(0.5);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = gen.transform(&data, 1).unwrap();

        assert_eq!(result.len(), 4);

        // First value = first observation
        assert_eq!(result[0], 1.0);

        // Subsequent values are smoothed
        // EWMA_1 = 0.5 * 2 + 0.5 * 1 = 1.5 (with adjustment)
        // After adjustment: varies based on cumulative weights
        assert!(result[1] > 1.0 && result[1] < 2.0);
    }

    #[test]
    fn test_ewma_from_span() {
        let gen = EwmaGenerator::from_span(3);
        // alpha = 2 / (3 + 1) = 0.5
        assert!((gen.alpha - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_ewma_handles_nan() {
        let gen = EwmaGenerator::new(0.5);

        let data = vec![1.0, f32::NAN, 3.0, 4.0];
        let result = gen.transform(&data, 1).unwrap();

        // NaN should propagate last EWMA
        assert_eq!(result[1], result[0]);
    }

    // ========================================
    // MomentumGenerator Tests
    // ========================================

    #[test]
    fn test_momentum_basic() {
        let gen = MomentumGenerator::new(vec![1, 2]);

        // 5 rows × 1 feature
        let data = vec![100.0, 110.0, 121.0, 133.1, 146.41];

        let result = gen.transform(&data, 1).unwrap();

        // Output: 5 rows × 3 features (1 original + 1 momentum_1 + 1 momentum_2)
        assert_eq!(result.len(), 15);

        // Row 2: original=121, momentum_1=(121-110)/110=0.1, momentum_2=(121-100)/100=0.21
        let row2_start = 2 * 3;
        assert_eq!(result[row2_start], 121.0);
        assert!((result[row2_start + 1] - 0.1).abs() < 0.001); // momentum lag 1
        assert!((result[row2_start + 2] - 0.21).abs() < 0.001); // momentum lag 2

        // Row 4: original=146.41, momentum_1=(146.41-133.1)/133.1=0.1
        let row4_start = 4 * 3;
        assert_eq!(result[row4_start], 146.41);
        assert!((result[row4_start + 1] - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_momentum_handles_zero() {
        let gen = MomentumGenerator::new(vec![1]);

        // Include a zero value
        let data = vec![100.0, 0.0, 50.0];
        let result = gen.transform(&data, 1).unwrap();

        // Row 1: momentum = (0 - 100) / 100 = -1.0
        assert!((result[3] + 1.0).abs() < 0.001);

        // Row 2: momentum = (50 - 0) / 0 = NaN (division by zero)
        assert!(result[5].is_nan());
    }

    #[test]
    fn test_momentum_column() {
        let gen = MomentumGenerator::new(vec![1, 3]);
        let column = vec![100.0, 110.0, 121.0, 133.1, 146.41];
        let result = gen.transform_column(&column);

        // Output: 5 rows × 2 lags = 10 values
        assert_eq!(result.len(), 10);

        // Lag 1 at row 1: (110 - 100) / 100 = 0.1
        assert!((result[1] - 0.1).abs() < 0.001);

        // Lag 3 at row 3: (133.1 - 100) / 100 = 0.331
        assert!((result[8] - 0.331).abs() < 0.001);
    }

    // ========================================
    // SeasonalGenerator Tests
    // ========================================

    #[test]
    fn test_seasonal_basic() {
        let gen = SeasonalGenerator::new(vec![
            SeasonalComponent::Hour,
            SeasonalComponent::DayOfWeek,
            SeasonalComponent::Month,
        ]);

        // 2024-01-15 12:00:00 UTC (Monday)
        let ts = 1705320000.0;
        let result = gen.transform_timestamps(&[ts]);

        assert_eq!(result.len(), 3);

        // Hour should be 12
        assert_eq!(result[0], 12.0);

        // Day of week should be 0 (Monday)
        assert_eq!(result[1], 0.0);

        // Month should be 1 (January)
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_seasonal_cyclical() {
        let gen = SeasonalGenerator::new(vec![SeasonalComponent::Hour]).with_cyclical(true);

        // Hour 0 (midnight)
        let ts_midnight = 1705276800.0; // 2024-01-15 00:00:00
        let result_midnight = gen.transform_timestamps(&[ts_midnight]);

        // Hour 12 (noon)
        let ts_noon = 1705320000.0; // 2024-01-15 12:00:00
        let result_noon = gen.transform_timestamps(&[ts_noon]);

        // At hour 0: sin(0) = 0, cos(0) = 1
        assert!(result_midnight[0].abs() < 0.001); // sin
        assert!((result_midnight[1] - 1.0).abs() < 0.001); // cos

        // At hour 12: sin(π) ≈ 0, cos(π) ≈ -1
        assert!(result_noon[0].abs() < 0.001); // sin
        assert!((result_noon[1] + 1.0).abs() < 0.001); // cos
    }

    #[test]
    fn test_seasonal_weekend() {
        let gen = SeasonalGenerator::new(vec![SeasonalComponent::IsWeekend]);

        // 2024-01-15 (Monday)
        let monday = 1705320000.0;
        let monday_result = gen.transform_timestamps(&[monday]);
        assert_eq!(monday_result[0], 0.0);

        // 2024-01-13 (Saturday)
        let saturday = 1705147200.0;
        let saturday_result = gen.transform_timestamps(&[saturday]);
        assert_eq!(saturday_result[0], 1.0);
    }

    #[test]
    fn test_seasonal_output_names() {
        let gen =
            SeasonalGenerator::new(vec![SeasonalComponent::Hour, SeasonalComponent::DayOfWeek])
                .with_cyclical(true);

        let names = gen.output_names("timestamp");
        assert_eq!(names.len(), 4); // 2 components × 2 (sin/cos each)
        assert_eq!(names[0], "timestamp_hour_sin");
        assert_eq!(names[1], "timestamp_hour_cos");
        assert_eq!(names[2], "timestamp_dow_sin");
        assert_eq!(names[3], "timestamp_dow_cos");
    }

    #[test]
    fn test_days_to_ymd() {
        // Test known dates
        // 2024-01-15 = 19737 days since 1970-01-01
        let (year, month, day, _) = days_to_ymd(19737);
        assert_eq!(year, 2024);
        assert_eq!(month, 1);
        assert_eq!(day, 15);

        // 1970-01-01 = 0 days
        let (year, month, day, doy) = days_to_ymd(0);
        assert_eq!(year, 1970);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(doy, 1);
    }

    // ========================================
    // Serialization Tests
    // ========================================

    #[test]
    fn test_lag_generator_serialization() {
        let gen = LagGenerator::new(vec![1, 7, 14]).with_nan_strategy(NaNStrategy::ForwardFill);

        let json = serde_json::to_string(&gen).unwrap();
        let loaded: LagGenerator = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.lags, vec![1, 7, 14]);
        assert_eq!(loaded.nan_strategy, NaNStrategy::ForwardFill);
    }

    #[test]
    fn test_rolling_generator_serialization() {
        let gen = RollingGenerator::new(7)
            .with_stats(vec![RollingStat::Mean, RollingStat::Std])
            .with_min_periods(3);

        let json = serde_json::to_string(&gen).unwrap();
        let loaded: RollingGenerator = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.window, 7);
        assert_eq!(loaded.stats.len(), 2);
        assert_eq!(loaded.min_periods, 3);
    }

    #[test]
    fn test_ewma_generator_serialization() {
        let gen = EwmaGenerator::new(0.3).without_adjust();

        let json = serde_json::to_string(&gen).unwrap();
        let loaded: EwmaGenerator = serde_json::from_str(&json).unwrap();

        assert!((loaded.alpha - 0.3).abs() < 1e-6);
        assert!(!loaded.adjust);
    }

    #[test]
    fn test_seasonal_generator_serialization() {
        let gen = SeasonalGenerator::new(vec![
            SeasonalComponent::Hour,
            SeasonalComponent::DayOfWeek,
            SeasonalComponent::Month,
        ])
        .with_cyclical(true);

        let json = serde_json::to_string(&gen).unwrap();
        let loaded: SeasonalGenerator = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.components.len(), 3);
        assert!(loaded.cyclical);
    }
}
