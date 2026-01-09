//! Cross-sectional features for panel data
//!
//! Transforms features to be relative to their cross-section (e.g., all stocks on the same date).
//! Essential for ranking models where relative position matters more than absolute values.

use polars::prelude::*;

/// Apply cross-sectional transformations to numeric columns
///
/// For panel data (e.g., stocks × dates), transforms features to be relative
/// to their cross-section. This is critical for ranking models (Rank IC).
///
/// # Transformations
///
/// For each numeric column, generates:
/// - `{col}_rank`: Percentile rank within cross-section [0, 1]
/// - `{col}_zscore`: Standardized value (mean=0, std=1) within cross-section
/// - `{col}_vs_median`: Distance from cross-sectional median
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data
/// * `group_col` - Column defining cross-sections (e.g., "date")
/// * `exclude_cols` - Columns to skip (e.g., IDs, targets)
///
/// # Example
///
/// ```ignore
/// // For stock data: transform features relative to same-day distribution
/// let df = apply_crosssectional_features(
///     &df,
///     "date",
///     &["code", "date", "y"]
/// )?;
///
/// // Now each stock's features are ranked against other stocks that day
/// // f_0_rank: where does this stock's f_0 rank among all stocks today?
/// // f_0_zscore: how many std devs from today's mean?
/// ```
pub fn apply_crosssectional_features(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
) -> PolarsResult<DataFrame> {
    apply_crosssectional_features_with_include(df, group_col, exclude_cols, None)
}

/// Apply cross-sectional transformations to specific columns
///
/// Same as `apply_crosssectional_features` but allows specifying which columns to include.
///
/// # Arguments
///
/// * `df` - Input DataFrame with panel data
/// * `group_col` - Column defining cross-sections (e.g., "date")
/// * `exclude_cols` - Columns to skip (e.g., IDs, targets)
/// * `include_cols` - Optional list of columns to process. If None, processes all numeric columns.
pub fn apply_crosssectional_features_with_include(
    df: &DataFrame,
    group_col: &str,
    exclude_cols: &[&str],
    include_cols: Option<&[&str]>,
) -> PolarsResult<DataFrame> {
    // Get numeric columns
    let numeric_cols: Vec<String> = if let Some(includes) = include_cols {
        // If include_cols specified, use only those
        includes
            .iter()
            .filter(|&&name| {
                if let Ok(col) = df.column(name) {
                    col.dtype().is_numeric() && name != group_col && !exclude_cols.contains(&name)
                } else {
                    false
                }
            })
            .map(|&s| s.to_string())
            .collect()
    } else {
        // Otherwise, process all numeric columns (excluding group and exclusion list)
        df.get_columns()
            .iter()
            .filter(|s| {
                let name = s.name().as_str();
                s.dtype().is_numeric()
                    && name != group_col
                    && !exclude_cols.contains(&name)
            })
            .map(|s| s.name().to_string())
            .collect()
    };

    if numeric_cols.is_empty() {
        return Ok(df.clone());
    }

    // Start with original dataframe
    let mut result = df.clone();

    // For each numeric column, compute transformations
    for col_name_str in &numeric_cols {
        let col_name = col_name_str.as_str();

        // Get group column and feature column
        let group_series = result.column(group_col)?;
        let feature_series = result.column(col_name)?;
        let feature_f64 = feature_series.cast(&DataType::Float64)?;
        let feature_vals = feature_f64.f64()?;

        let n_rows = result.height();

        // Initialize output arrays
        let mut rank_vals = vec![0.0; n_rows];
        let mut zscore_vals = vec![0.0; n_rows];
        let mut median_diff_vals = vec![0.0; n_rows];

        // Get unique groups
        let unique_groups = group_series.unique()?;
        let n_groups = unique_groups.len();

        // Process each group
        for i in 0..n_groups {
            let group_val = unique_groups.get(i).unwrap();

            // Find rows in this group by comparing values directly
            let indices: Vec<usize> = (0..n_rows)
                .filter(|&row_idx| {
                    if let Ok(val) = group_series.get(row_idx) {
                        val == group_val
                    } else {
                        false
                    }
                })
                .collect();

            if indices.is_empty() {
                continue;
            }

            // Extract group values
            let group_vals: Vec<f64> = indices
                .iter()
                .filter_map(|&idx| feature_vals.get(idx))
                .collect();

            if group_vals.is_empty() {
                continue;
            }

            // Compute statistics
            let mean: f64 = group_vals.iter().sum::<f64>() / group_vals.len() as f64;
            let var: f64 = group_vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / group_vals.len() as f64;
            let std = var.sqrt();

            let mut sorted_vals = group_vals.clone();
            sorted_vals.sort_by(|a, b| {
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater, // NaN goes to end
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => a.partial_cmp(b).unwrap(),
                }
            });
            let median = if sorted_vals.len() % 2 == 0 {
                let mid = sorted_vals.len() / 2;
                (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
            } else {
                sorted_vals[sorted_vals.len() / 2]
            };

            // Compute ranks for this group
            let mut indexed_vals: Vec<(usize, f64)> = indices
                .iter()
                .filter_map(|&idx| feature_vals.get(idx).map(|val| (idx, val)))
                .collect();
            indexed_vals.sort_by(|a, b| {
                match (a.1.is_nan(), b.1.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater, // NaN goes to end
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => a.1.partial_cmp(&b.1).unwrap(),
                }
            });

            // Assign percentile ranks
            for (rank, (orig_idx, _)) in indexed_vals.iter().enumerate() {
                let percentile = if indexed_vals.len() > 1 {
                    rank as f64 / (indexed_vals.len() - 1) as f64
                } else {
                    0.5
                };
                rank_vals[*orig_idx] = percentile;
            }

            // Assign zscore and median diff
            for &idx in &indices {
                if let Some(val) = feature_vals.get(idx) {
                    zscore_vals[idx] = if std > 0.0 { (val - mean) / std } else { 0.0 };
                    median_diff_vals[idx] = val - median;
                }
            }
        }

        // Add new columns
        let rank_series = Series::new(format!("{}_rank", col_name_str).as_str().into(), rank_vals);
        let zscore_series = Series::new(format!("{}_zscore", col_name_str).as_str().into(), zscore_vals);
        let median_diff_series = Series::new(format!("{}_vs_median", col_name_str).as_str().into(), median_diff_vals);

        result.with_column(rank_series)?;
        result.with_column(zscore_series)?;
        result.with_column(median_diff_series)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosssectional_features() {
        // Create sample panel data (3 stocks × 2 dates)
        let df = df!(
            "date" => &[1, 1, 1, 2, 2, 2],
            "stock" => &[1, 2, 3, 1, 2, 3],
            "price" => &[100.0, 200.0, 150.0, 110.0, 190.0, 160.0],
        )
        .unwrap();

        let result = apply_crosssectional_features(&df, "date", &["stock", "date"]).unwrap();

        // Check new columns exist
        assert!(result.column("price_rank").is_ok());
        assert!(result.column("price_zscore").is_ok());
        assert!(result.column("price_vs_median").is_ok());

        // Check ranks for date=1: [100, 200, 150] -> ranks [0.0, 1.0, 0.5]
        let ranks = result
            .column("price_rank")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();

        // 100 < 150 < 200, so ranks: 0/2, 2/2, 1/2 = 0.0, 1.0, 0.5
        assert!((ranks[0].unwrap() - 0.0).abs() < 0.01);
        assert!((ranks[1].unwrap() - 1.0).abs() < 0.01);
        assert!((ranks[2].unwrap() - 0.5).abs() < 0.01);
    }
}
