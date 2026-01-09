//! Group-aware time-series feature generation
//!
//! This module provides time-series feature generators that process data WITHIN groups,
//! which is essential for panel data (e.g., multiple stocks over time).
//!
//! # Problem with Non-Grouped Generators
//!
//! Without group awareness, lag features would leak across entities:
//! ```text
//! Row  Stock  Date   Price  Lag-1 (WRONG)  Lag-1 (CORRECT)
//! 0    AAPL   Day1   100    NaN            NaN
//! 1    GOOG   Day1   200    100 (LEAK!)    NaN
//! 2    AAPL   Day2   105    200 (LEAK!)    100
//! 3    GOOG   Day2   210    105 (LEAK!)    200
//! ```
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{GroupedTimeSeriesGenerator, GroupedTimeSeriesConfig};
//!
//! let config = GroupedTimeSeriesConfig::daily();
//! let mut gen = GroupedTimeSeriesGenerator::new(config);
//!
//! // Fit with group IDs and timestamps
//! gen.fit(&group_ids, &timestamps, vec![0, 1, 2])?;
//!
//! // Transform generates features WITHIN each group
//! let (features, names) = gen.transform(&data, num_features)?;
//! ```

use super::timeseries::{EwmaGenerator, LagGenerator, NaNStrategy, RollingGenerator, RollingStat};
use crate::{Result, TreeBoostError};
use std::collections::HashMap;

/// Configuration for group-aware time-series features
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GroupedTimeSeriesConfig {
    /// Lag periods to generate [1, 3, 7, 14, ...]
    pub lag_periods: Vec<usize>,
    /// Rolling window sizes [7, 14, 28, ...]
    pub rolling_windows: Vec<usize>,
    /// Statistics for rolling windows
    pub rolling_stats: Vec<RollingStat>,
    /// EWMA alpha values [0.1, 0.3]
    pub ewma_alphas: Vec<f32>,
    /// Momentum periods for return calculation [1, 3, 7, 14, ...]
    pub momentum_periods: Vec<usize>,
    /// NaN handling strategy
    pub nan_strategy: NaNStrategy,
    /// Minimum periods for rolling statistics
    pub min_periods: usize,
}

impl Default for GroupedTimeSeriesConfig {
    fn default() -> Self {
        Self::daily()
    }
}

impl GroupedTimeSeriesConfig {
    /// Configuration for daily data
    pub fn daily() -> Self {
        Self {
            lag_periods: vec![1, 3, 7, 14],
            rolling_windows: vec![7, 14, 28],
            rolling_stats: vec![RollingStat::Mean, RollingStat::Std],
            ewma_alphas: vec![0.1, 0.3],
            momentum_periods: vec![1, 3, 7, 14],
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    }

    /// Configuration for hourly data
    pub fn hourly() -> Self {
        Self {
            lag_periods: vec![1, 6, 12, 24],
            rolling_windows: vec![12, 24, 48],
            rolling_stats: vec![RollingStat::Mean, RollingStat::Std],
            ewma_alphas: vec![0.1, 0.3],
            momentum_periods: vec![1, 6, 12, 24],
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    }

    /// Configuration for weekly data
    pub fn weekly() -> Self {
        Self {
            lag_periods: vec![1, 4, 12],
            rolling_windows: vec![4, 12, 26],
            rolling_stats: vec![RollingStat::Mean, RollingStat::Std],
            ewma_alphas: vec![0.1, 0.3],
            momentum_periods: vec![1, 4, 12],
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    }

    /// Minimal configuration for quick testing
    pub fn minimal() -> Self {
        Self {
            lag_periods: vec![1],
            rolling_windows: vec![7],
            rolling_stats: vec![RollingStat::Mean],
            ewma_alphas: vec![],
            momentum_periods: vec![],
            nan_strategy: NaNStrategy::Keep,
            min_periods: 1,
        }
    }

    /// Builder: set lag periods
    pub fn with_lag_periods(mut self, periods: Vec<usize>) -> Self {
        self.lag_periods = periods;
        self
    }

    /// Builder: set rolling windows
    pub fn with_rolling_windows(mut self, windows: Vec<usize>) -> Self {
        self.rolling_windows = windows;
        self
    }

    /// Builder: set rolling statistics
    pub fn with_rolling_stats(mut self, stats: Vec<RollingStat>) -> Self {
        self.rolling_stats = stats;
        self
    }

    /// Builder: set EWMA alphas
    pub fn with_ewma_alphas(mut self, alphas: Vec<f32>) -> Self {
        self.ewma_alphas = alphas;
        self
    }

    /// Builder: set NaN strategy
    pub fn with_nan_strategy(mut self, strategy: NaNStrategy) -> Self {
        self.nan_strategy = strategy;
        self
    }

    /// Builder: set minimum periods for rolling
    pub fn with_min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }

    /// Calculate number of features generated per input column
    pub fn features_per_column(&self) -> usize {
        let n_lags = self.lag_periods.len();
        let n_rolling = self.rolling_windows.len() * self.rolling_stats.len();
        let n_ewma = self.ewma_alphas.len();
        let n_momentum = self.momentum_periods.len();
        n_lags + n_rolling + n_ewma + n_momentum
    }
}

/// Group-aware time-series feature generator
///
/// Computes lag, rolling, and EWMA features WITHIN each group (e.g., per stock code).
/// This prevents data leakage in panel data scenarios.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GroupedTimeSeriesGenerator {
    config: GroupedTimeSeriesConfig,
    /// Group indices: group_id -> row indices sorted by timestamp
    group_indices: Option<Vec<Vec<usize>>>,
    /// Feature column indices to transform
    feature_columns: Vec<usize>,
    /// Feature column names (for output naming)
    feature_names: Vec<String>,
}

impl GroupedTimeSeriesGenerator {
    /// Create a new group-aware time-series generator
    pub fn new(config: GroupedTimeSeriesConfig) -> Self {
        Self {
            config,
            group_indices: None,
            feature_columns: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    /// Create with daily configuration
    pub fn daily() -> Self {
        Self::new(GroupedTimeSeriesConfig::daily())
    }

    /// Create with hourly configuration
    pub fn hourly() -> Self {
        Self::new(GroupedTimeSeriesConfig::hourly())
    }

    /// Create with weekly configuration
    pub fn weekly() -> Self {
        Self::new(GroupedTimeSeriesConfig::weekly())
    }

    /// Fit the generator with group assignments and timestamps
    ///
    /// # Arguments
    /// * `group_ids` - Group ID for each row (e.g., stock code encoded as u32)
    /// * `timestamps` - Unix timestamp for each row (for sorting within groups)
    /// * `feature_indices` - Which column indices to generate features for
    /// * `feature_names` - Names for those columns (for output naming)
    ///
    /// # Returns
    /// Ok(()) if successful, Err if data is invalid
    pub fn fit(
        &mut self,
        group_ids: &[u32],
        timestamps: &[f64],
        feature_indices: Vec<usize>,
        feature_names: Vec<String>,
    ) -> Result<()> {
        if group_ids.len() != timestamps.len() {
            return Err(TreeBoostError::Data(format!(
                "group_ids length ({}) != timestamps length ({})",
                group_ids.len(),
                timestamps.len()
            )));
        }

        if feature_indices.len() != feature_names.len() {
            return Err(TreeBoostError::Data(format!(
                "feature_indices length ({}) != feature_names length ({})",
                feature_indices.len(),
                feature_names.len()
            )));
        }

        // Build group_id -> [(row_idx, timestamp), ...] mapping
        let mut groups: HashMap<u32, Vec<(usize, f64)>> = HashMap::new();

        for (row_idx, (&gid, &ts)) in group_ids.iter().zip(timestamps.iter()).enumerate() {
            groups.entry(gid).or_default().push((row_idx, ts));
        }

        // Sort each group by timestamp
        let mut group_indices = Vec::with_capacity(groups.len());
        for (_, mut rows) in groups {
            rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            group_indices.push(rows.into_iter().map(|(idx, _)| idx).collect());
        }

        self.group_indices = Some(group_indices);
        self.feature_columns = feature_indices;
        self.feature_names = feature_names;

        Ok(())
    }

    /// Get the number of groups detected
    pub fn num_groups(&self) -> usize {
        self.group_indices.as_ref().map(|g| g.len()).unwrap_or(0)
    }

    /// Get the number of output features that will be generated
    pub fn num_output_features(&self) -> usize {
        self.feature_columns.len() * self.config.features_per_column()
    }

    /// Get output feature names
    pub fn output_feature_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.num_output_features());

        for col_name in &self.feature_names {
            // Lag features
            for lag in &self.config.lag_periods {
                names.push(format!("{}_lag_{}", col_name, lag));
            }

            // Rolling features
            for window in &self.config.rolling_windows {
                for stat in &self.config.rolling_stats {
                    names.push(format!("{}_roll_{}_{}", col_name, window, stat.suffix()));
                }
            }

            // EWMA features
            for alpha in &self.config.ewma_alphas {
                names.push(format!("{}_ewma_{:.2}", col_name, alpha));
            }

            // Momentum features
            for period in &self.config.momentum_periods {
                names.push(format!("{}_momentum_{}", col_name, period));
            }
        }

        names
    }

    /// Transform data by generating group-aware time-series features
    ///
    /// # Arguments
    /// * `data` - Row-major data matrix (num_rows × num_features)
    /// * `num_features` - Number of features per row
    ///
    /// # Returns
    /// (new_features, feature_names) - Only the generated features (not original data)
    pub fn transform(
        &self,
        data: &[f32],
        num_features: usize,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        let group_indices = self.group_indices.as_ref().ok_or_else(|| {
            TreeBoostError::Config("Generator not fitted - call fit() first".into())
        })?;

        let num_rows = data.len() / num_features;
        if num_rows * num_features != data.len() {
            return Err(TreeBoostError::Data(format!(
                "Data length {} not divisible by num_features {}",
                data.len(),
                num_features
            )));
        }

        let total_new_features = self.num_output_features();
        if total_new_features == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Initialize output with NaN
        let mut new_data = vec![f32::NAN; num_rows * total_new_features];
        let feature_names = self.output_feature_names();

        // Create generators (reusable across groups)
        let lag_gen = LagGenerator::new(self.config.lag_periods.clone())
            .with_nan_strategy(self.config.nan_strategy);

        let rolling_gens: Vec<RollingGenerator> = self
            .config
            .rolling_windows
            .iter()
            .map(|&w| {
                RollingGenerator::new(w)
                    .with_stats(self.config.rolling_stats.clone())
                    .with_min_periods(self.config.min_periods)
            })
            .collect();

        let ewma_gens: Vec<EwmaGenerator> = self
            .config
            .ewma_alphas
            .iter()
            .map(|&alpha| EwmaGenerator::new(alpha))
            .collect();

        let momentum_gen = crate::features::MomentumGenerator::new(self.config.momentum_periods.clone());

        // Process each group independently
        for group_rows in group_indices {
            if group_rows.is_empty() {
                continue;
            }

            // Process each feature column
            for (feat_idx, &col_idx) in self.feature_columns.iter().enumerate() {
                // Extract column for this group (in time order)
                let column: Vec<f32> = group_rows
                    .iter()
                    .map(|&row| {
                        if row * num_features + col_idx < data.len() {
                            data[row * num_features + col_idx]
                        } else {
                            f32::NAN
                        }
                    })
                    .collect();

                let features_per_col = self.config.features_per_column();
                let base_offset = feat_idx * features_per_col;

                // Generate lag features
                let lagged = lag_gen.transform_column(&column);
                let n_lags = self.config.lag_periods.len();

                for (local_idx, &global_row) in group_rows.iter().enumerate() {
                    for lag_idx in 0..n_lags {
                        let src_idx = lag_idx * column.len() + local_idx;
                        let dst_idx = global_row * total_new_features + base_offset + lag_idx;
                        if src_idx < lagged.len() && dst_idx < new_data.len() {
                            new_data[dst_idx] = lagged[src_idx];
                        }
                    }
                }

                // Generate rolling features
                let mut roll_offset = n_lags;
                for rolling_gen in &rolling_gens {
                    let rolled = rolling_gen.transform_column(&column);
                    let n_stats = self.config.rolling_stats.len();

                    for (local_idx, &global_row) in group_rows.iter().enumerate() {
                        for stat_idx in 0..n_stats {
                            let src_idx = stat_idx * column.len() + local_idx;
                            let dst_idx =
                                global_row * total_new_features + base_offset + roll_offset + stat_idx;
                            if src_idx < rolled.len() && dst_idx < new_data.len() {
                                new_data[dst_idx] = rolled[src_idx];
                            }
                        }
                    }
                    roll_offset += n_stats;
                }

                // Generate EWMA features
                for (ewma_idx, ewma_gen) in ewma_gens.iter().enumerate() {
                    let ewma = ewma_gen.transform_column(&column);

                    for (local_idx, &global_row) in group_rows.iter().enumerate() {
                        let dst_idx =
                            global_row * total_new_features + base_offset + roll_offset + ewma_idx;
                        if local_idx < ewma.len() && dst_idx < new_data.len() {
                            new_data[dst_idx] = ewma[local_idx];
                        }
                    }
                }

                // Generate momentum features
                let momentum = momentum_gen.transform_column(&column);
                let n_momentum = self.config.momentum_periods.len();
                let n_ewma = self.config.ewma_alphas.len();
                let momentum_offset = roll_offset + n_ewma;

                for (local_idx, &global_row) in group_rows.iter().enumerate() {
                    for mom_idx in 0..n_momentum {
                        let src_idx = mom_idx * column.len() + local_idx;
                        let dst_idx =
                            global_row * total_new_features + base_offset + momentum_offset + mom_idx;
                        if src_idx < momentum.len() && dst_idx < new_data.len() {
                            new_data[dst_idx] = momentum[src_idx];
                        }
                    }
                }
            }
        }

        Ok((new_data, feature_names))
    }

    /// Get the configuration
    pub fn config(&self) -> &GroupedTimeSeriesConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grouped_lag_basic() {
        // Two groups (A=0, B=1), each with 5 time-ordered rows
        // Data is interleaved: A0, B0, A1, B1, A2, B2, A3, B3, A4, B4
        let group_ids = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let timestamps = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];

        // Single feature column with values:
        // Group A: 10, 20, 30, 40, 50 (at rows 0, 2, 4, 6, 8)
        // Group B: 100, 200, 300, 400, 500 (at rows 1, 3, 5, 7, 9)
        let data = vec![
            10.0, 100.0, 20.0, 200.0, 30.0, 300.0, 40.0, 400.0, 50.0, 500.0,
        ];

        let config = GroupedTimeSeriesConfig::minimal()
            .with_lag_periods(vec![1])
            .with_rolling_windows(vec![])
            .with_ewma_alphas(vec![]);

        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(&group_ids, &timestamps, vec![0], vec!["price".to_string()])
            .unwrap();

        assert_eq!(gen.num_groups(), 2);

        let (features, names) = gen.transform(&data, 1).unwrap();

        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "price_lag_1");
        assert_eq!(features.len(), 10);

        // Group A lag-1: [NaN, 10, 20, 30, 40] at global rows [0, 2, 4, 6, 8]
        assert!(features[0].is_nan()); // Row 0 (Group A, t=1)
        assert_eq!(features[2], 10.0); // Row 2 (Group A, t=2)
        assert_eq!(features[4], 20.0); // Row 4 (Group A, t=3)
        assert_eq!(features[6], 30.0); // Row 6 (Group A, t=4)
        assert_eq!(features[8], 40.0); // Row 8 (Group A, t=5)

        // Group B lag-1: [NaN, 100, 200, 300, 400] at global rows [1, 3, 5, 7, 9]
        assert!(features[1].is_nan()); // Row 1 (Group B, t=1) - should be NaN, NOT 10!
        assert_eq!(features[3], 100.0); // Row 3 (Group B, t=2)
        assert_eq!(features[5], 200.0); // Row 5 (Group B, t=3)
        assert_eq!(features[7], 300.0); // Row 7 (Group B, t=4)
        assert_eq!(features[9], 400.0); // Row 9 (Group B, t=5)
    }

    #[test]
    fn test_grouped_rolling_mean() {
        // Simple case: one group with 5 rows
        let group_ids = vec![0, 0, 0, 0, 0];
        let timestamps = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let config = GroupedTimeSeriesConfig::minimal()
            .with_lag_periods(vec![])
            .with_rolling_windows(vec![3])
            .with_rolling_stats(vec![RollingStat::Mean])
            .with_ewma_alphas(vec![]);

        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(&group_ids, &timestamps, vec![0], vec!["val".to_string()])
            .unwrap();

        let (features, _names) = gen.transform(&data, 1).unwrap();

        // Rolling mean with window=3:
        // Row 0: [10] -> 10
        // Row 1: [10, 20] -> 15
        // Row 2: [10, 20, 30] -> 20
        // Row 3: [20, 30, 40] -> 30
        // Row 4: [30, 40, 50] -> 40
        assert!((features[0] - 10.0).abs() < 0.01);
        assert!((features[1] - 15.0).abs() < 0.01);
        assert!((features[2] - 20.0).abs() < 0.01);
        assert!((features[3] - 30.0).abs() < 0.01);
        assert!((features[4] - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_grouped_multiple_features() {
        let group_ids = vec![0, 0, 0];
        let timestamps = vec![1.0, 2.0, 3.0];

        // Two feature columns
        let data = vec![
            10.0, 100.0, // row 0
            20.0, 200.0, // row 1
            30.0, 300.0, // row 2
        ];

        let config = GroupedTimeSeriesConfig::minimal()
            .with_lag_periods(vec![1])
            .with_rolling_windows(vec![])
            .with_ewma_alphas(vec![]);

        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(
            &group_ids,
            &timestamps,
            vec![0, 1],
            vec!["feat_a".to_string(), "feat_b".to_string()],
        )
        .unwrap();

        let (features, names) = gen.transform(&data, 2).unwrap();

        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "feat_a_lag_1");
        assert_eq!(names[1], "feat_b_lag_1");

        // 3 rows × 2 output features
        assert_eq!(features.len(), 6);

        // Row 0: NaN for both lags
        assert!(features[0].is_nan());
        assert!(features[1].is_nan());

        // Row 1: lag-1 = [10, 100]
        assert_eq!(features[2], 10.0);
        assert_eq!(features[3], 100.0);

        // Row 2: lag-1 = [20, 200]
        assert_eq!(features[4], 20.0);
        assert_eq!(features[5], 200.0);
    }

    #[test]
    fn test_full_config() {
        let group_ids = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let timestamps: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let data: Vec<f32> = (1..=8).map(|i| i as f32 * 10.0).collect();

        let config = GroupedTimeSeriesConfig::daily();
        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(&group_ids, &timestamps, vec![0], vec!["price".to_string()])
            .unwrap();

        let (features, names) = gen.transform(&data, 1).unwrap();

        // Verify feature count
        // Lags: 4 (1, 3, 7, 14)
        // Rolling: 3 windows × 2 stats = 6
        // EWMA: 2
        // Total: 4 + 6 + 2 = 12 per column
        let expected_per_col = gen.config().features_per_column();
        assert_eq!(expected_per_col, 12);
        assert_eq!(names.len(), 12);
        assert_eq!(features.len(), 8 * 12);

        // Verify name patterns
        assert!(names.iter().any(|n| n.contains("lag_1")));
        assert!(names.iter().any(|n| n.contains("roll_7_mean")));
        assert!(names.iter().any(|n| n.contains("ewma_0.10")));
    }

    #[test]
    fn test_unsorted_input() {
        // Data provided in wrong order should still work (sorted internally)
        let group_ids = vec![0, 0, 0];
        let timestamps = vec![3.0, 1.0, 2.0]; // Out of order!
        let data = vec![30.0, 10.0, 20.0]; // Corresponding values

        let config = GroupedTimeSeriesConfig::minimal()
            .with_lag_periods(vec![1])
            .with_rolling_windows(vec![])
            .with_ewma_alphas(vec![]);

        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(&group_ids, &timestamps, vec![0], vec!["val".to_string()])
            .unwrap();

        let (features, _) = gen.transform(&data, 1).unwrap();

        // After sorting by timestamp: [10, 20, 30]
        // Lag-1: [NaN, 10, 20]
        // Original rows: 1, 2, 0 (sorted by timestamp)
        // So features[1] = NaN (first in time order)
        // features[2] = 10 (second in time order, lag from first)
        // features[0] = 20 (third in time order, lag from second)
        assert!(features[1].is_nan()); // Global row 1 was t=1
        assert_eq!(features[2], 10.0); // Global row 2 was t=2
        assert_eq!(features[0], 20.0); // Global row 0 was t=3
    }

    #[test]
    fn test_single_row_group() {
        // Groups with only one row should produce all NaN for lags
        let group_ids = vec![0, 1, 2];
        let timestamps = vec![1.0, 1.0, 1.0];
        let data = vec![10.0, 20.0, 30.0];

        let config = GroupedTimeSeriesConfig::minimal()
            .with_lag_periods(vec![1])
            .with_rolling_windows(vec![])
            .with_ewma_alphas(vec![]);

        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(&group_ids, &timestamps, vec![0], vec!["val".to_string()])
            .unwrap();

        let (features, _) = gen.transform(&data, 1).unwrap();

        // All groups have single observation, so lag-1 is NaN for all
        assert!(features[0].is_nan());
        assert!(features[1].is_nan());
        assert!(features[2].is_nan());
    }

    #[test]
    fn test_output_names() {
        let config = GroupedTimeSeriesConfig::daily();
        let mut gen = GroupedTimeSeriesGenerator::new(config);
        gen.fit(
            &[0],
            &[1.0],
            vec![0, 1],
            vec!["price".to_string(), "volume".to_string()],
        )
        .unwrap();

        let names = gen.output_feature_names();

        // 12 features per column × 2 columns = 24 names
        assert_eq!(names.len(), 24);

        // Check some expected names
        assert!(names.contains(&"price_lag_1".to_string()));
        assert!(names.contains(&"price_roll_7_mean".to_string()));
        assert!(names.contains(&"volume_lag_1".to_string()));
        assert!(names.contains(&"volume_ewma_0.10".to_string()));
    }

    #[test]
    fn test_serialization() {
        let config = GroupedTimeSeriesConfig::daily()
            .with_lag_periods(vec![1, 7])
            .with_rolling_windows(vec![7, 14]);

        let json = serde_json::to_string(&config).unwrap();
        let loaded: GroupedTimeSeriesConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.lag_periods, vec![1, 7]);
        assert_eq!(loaded.rolling_windows, vec![7, 14]);
    }
}
