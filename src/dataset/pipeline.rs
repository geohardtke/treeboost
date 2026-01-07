//! Dirty Data Pipeline
//!
//! Implements the full preprocessing pipeline for handling messy real-world data:
//! 1. Count-Min Sketch filtering for rare categories → "unknown"
//! 2. Ordered Target Encoding with M-Estimate smoothing
//! 3. T-Digest quantile binning to u8
//!
//! This pipeline prevents target leakage and handles high-cardinality categoricals
//! with typos and rare values gracefully.

use crate::dataset::binner::QuantileBinner;
use crate::dataset::{BinnedDataset, FeatureInfo, FeatureType};
use crate::encoding::{CategoryFilter, CategoryMapping, EncodingMap, OrderedTargetEncoder};
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::path::Path;

/// Configuration for the dirty data pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of bins for numeric features (default: 255)
    pub num_bins: usize,
    /// CMS error tolerance (default: 0.001 = 0.1%)
    pub cms_eps: f64,
    /// CMS confidence level (default: 0.99)
    pub cms_confidence: f64,
    /// Minimum count for a category to be kept (default: 5)
    pub min_category_count: u64,
    /// M-estimate smoothing parameter for target encoding (default: 10.0)
    pub target_encoding_smoothing: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_bins: 255,
            cms_eps: 0.001,
            cms_confidence: 0.99,
            min_category_count: 5,
            target_encoding_smoothing: 10.0,
        }
    }
}

impl PipelineConfig {
    /// Create a new pipeline config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of bins for numeric features
    pub fn with_num_bins(mut self, num_bins: usize) -> Self {
        self.num_bins = num_bins;
        self
    }

    /// Set CMS parameters for rare category filtering
    pub fn with_cms_params(mut self, eps: f64, confidence: f64, min_count: u64) -> Self {
        self.cms_eps = eps;
        self.cms_confidence = confidence;
        self.min_category_count = min_count;
        self
    }

    /// Set target encoding smoothing parameter
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.target_encoding_smoothing = smoothing;
        self
    }
}

/// Learned encoding state for a single categorical column
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct CategoricalEncodingState {
    /// Column name
    pub name: String,
    /// Category to index mapping (for filtering rare categories)
    pub category_mapping: CategoryMapping,
    /// Target encoding map (category → encoded value)
    pub encoding_map: EncodingMap,
    /// Bin boundaries for the encoded values
    pub bin_boundaries: Vec<f64>,
}

/// Complete pipeline state for inference
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct PipelineState {
    /// Feature info for all columns (ordering matters)
    pub feature_info: Vec<FeatureInfo>,
    /// Encoding state for categorical columns (by column name)
    pub categorical_encodings: Vec<CategoricalEncodingState>,
    /// Column names in order
    pub column_order: Vec<String>,
    /// Which columns are categorical (by index)
    pub categorical_indices: Vec<usize>,
}

/// Dirty data pipeline for training
///
/// Processes data through:
/// 1. CMS filtering (rare categories → "unknown")
/// 2. Ordered Target Encoding (with M-Estimate smoothing)
/// 3. Quantile binning (T-Digest → u8)
pub struct DataPipeline {
    config: PipelineConfig,
    binner: QuantileBinner,
}

impl DataPipeline {
    /// Create a new data pipeline with the given configuration
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            binner: QuantileBinner::new(config.num_bins),
            config,
        }
    }

    /// Create a new data pipeline with default configuration
    ///
    /// This is a convenience method equivalent to `DataPipeline::new(PipelineConfig::default())`.
    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Load and process a CSV file for training
    pub fn load_csv_for_training(
        &self,
        path: impl AsRef<Path>,
        target_column: &str,
        categorical_columns: Option<&[&str]>,
    ) -> Result<(BinnedDataset, PipelineState, DataFrame)> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        self.process_for_training(df, target_column, categorical_columns)
    }

    /// Load and process a Parquet file for training
    pub fn load_parquet_for_training(
        &self,
        path: impl AsRef<Path>,
        target_column: &str,
        categorical_columns: Option<&[&str]>,
    ) -> Result<(BinnedDataset, PipelineState, DataFrame)> {
        let pl_path = PlPath::new(&path.as_ref().to_string_lossy());
        let df = LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?;
        self.process_for_training(df, target_column, categorical_columns)
    }

    /// Process a DataFrame for training
    ///
    /// Returns the binned dataset and the learned pipeline state for inference.
    pub fn process_for_training(
        &self,
        df: DataFrame,
        target_column: &str,
        categorical_columns: Option<&[&str]>,
    ) -> Result<(BinnedDataset, PipelineState, DataFrame)> {
        eprintln!(
            "[PHASE1 DataPipeline] Input DataFrame: {} rows, {} cols",
            df.height(),
            df.width()
        );
        eprintln!(
            "[PHASE1 DataPipeline] Column names: {:?}",
            df.get_column_names()
        );

        let _num_rows = df.height();

        // Extract target column
        let target_col = df.column(target_column).map_err(|e| {
            TreeBoostError::Data(format!(
                "Target column '{}' not found: {}",
                target_column, e
            ))
        })?;
        let targets: Vec<f64> = self.series_to_f64(target_col.as_materialized_series())?;

        // Fill NaN targets with 0 instead of filtering rows
        // This preserves all data and allows the model to learn from patterns
        let mut targets_filled: Vec<f64> = targets;
        for t in targets_filled.iter_mut() {
            if t.is_nan() {
                *t = 0.0;
            }
        }

        // Convert to f32 for training
        let targets_f32: Vec<f32> = targets_filled.iter().map(|&t| t as f32).collect();

        // Keep f64 version for categorical encoding
        let targets_filtered: Vec<f64> = targets_filled;

        let num_rows = targets_f32.len();

        // Determine feature columns (excluding target)
        let feature_names: Vec<String> = df
            .get_column_names()
            .into_iter()
            .filter(|name| *name != target_column)
            .map(|s| s.to_string())
            .collect();

        // Identify categorical columns
        let categorical_set: std::collections::HashSet<&str> = match categorical_columns {
            Some(cols) => cols.iter().copied().collect(),
            None => {
                // Auto-detect: String columns are categorical
                feature_names
                    .iter()
                    .filter(|name| {
                        matches!(
                            df.column(name.as_str()).map(|c| c.dtype().clone()),
                            Ok(DataType::String) | Ok(DataType::Categorical(_, _))
                        )
                    })
                    .map(|s| s.as_str())
                    .collect()
            }
        };

        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(feature_names.len());
        let mut all_info: Vec<FeatureInfo> = Vec::with_capacity(feature_names.len());
        let mut categorical_encodings: Vec<CategoricalEncodingState> = Vec::new();
        let mut categorical_indices: Vec<usize> = Vec::new();

        for (col_idx, name) in feature_names.iter().enumerate() {
            let col = df.column(name.as_str()).map_err(|e| {
                TreeBoostError::Data(format!("Feature column '{}' not found: {}", name, e))
            })?;
            let series = col.as_materialized_series();

            if categorical_set.contains(name.as_str()) {
                // Categorical column: CMS filter → Target Encode → Bin
                let (binned, info, encoding_state) =
                    self.process_categorical_column(name.clone(), series, &targets_filtered)?;
                all_binned.push(binned);
                all_info.push(info);
                categorical_encodings.push(encoding_state);
                categorical_indices.push(col_idx);
            } else {
                // Numeric column: Quantile bin directly
                let (binned, info) = self.process_numeric_column(name.clone(), series)?;
                all_binned.push(binned);
                all_info.push(info);
            }
        }

        // Flatten to column-major layout
        let mut features = Vec::with_capacity(num_rows * all_binned.len());
        for binned_col in all_binned {
            features.extend(binned_col);
        }

        let dataset = BinnedDataset::new(num_rows, features, targets_f32.clone(), all_info.clone());

        eprintln!(
            "[PHASE2 DataPipeline] After binning: {} rows, {} features",
            num_rows,
            all_info.len()
        );
        eprintln!("[PHASE2 DataPipeline] Feature names: {:?}", feature_names);
        eprintln!(
            "[PHASE2 DataPipeline] Targets: len={}, mean={:.4}, first 5: {:?}",
            targets_f32.len(),
            targets_f32.iter().sum::<f32>() / targets_f32.len() as f32,
            &targets_f32[..targets_f32.len().min(5)]
        );

        let state = PipelineState {
            feature_info: all_info,
            categorical_encodings,
            column_order: feature_names,
            categorical_indices,
        };

        Ok((dataset, state, df))
    }

    /// Load and process a CSV file for inference using learned state
    pub fn load_csv_for_inference(
        &self,
        path: impl AsRef<Path>,
        state: &PipelineState,
    ) -> Result<BinnedDataset> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        self.process_for_inference(df, state)
    }

    /// Load and process a Parquet file for inference using learned state
    pub fn load_parquet_for_inference(
        &self,
        path: impl AsRef<Path>,
        state: &PipelineState,
    ) -> Result<BinnedDataset> {
        let pl_path = PlPath::new(&path.as_ref().to_string_lossy());
        let df = LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?;
        self.process_for_inference(df, state)
    }

    /// Process a DataFrame for inference using learned state
    pub fn process_for_inference(
        &self,
        df: DataFrame,
        state: &PipelineState,
    ) -> Result<BinnedDataset> {
        eprintln!(
            "[DEBUG process_for_inference] Input df: {} rows, {} cols",
            df.height(),
            df.width()
        );
        eprintln!(
            "[DEBUG process_for_inference] State has {} features, {} categorical",
            state.feature_info.len(),
            state.categorical_indices.len()
        );

        let num_rows = df.height();

        // Build lookup for categorical encoding states
        let cat_state_map: std::collections::HashMap<&str, &CategoricalEncodingState> = state
            .categorical_encodings
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();

        let cat_indices_set: std::collections::HashSet<usize> =
            state.categorical_indices.iter().copied().collect();

        let mut all_binned: Vec<Vec<u8>> = Vec::with_capacity(state.column_order.len());

        for (col_idx, name) in state.column_order.iter().enumerate() {
            let col = df.column(name.as_str()).map_err(|e| {
                TreeBoostError::Data(format!("Feature column '{}' not found: {}", name, e))
            })?;
            let series = col.as_materialized_series();

            if cat_indices_set.contains(&col_idx) {
                eprintln!(
                    "[DEBUG process_for_inference] Column {}: '{}' - CATEGORICAL",
                    col_idx, name
                );
                // Categorical: use learned encoding
                let encoding_state = cat_state_map.get(name.as_str()).ok_or_else(|| {
                    TreeBoostError::Data(format!(
                        "Missing encoding state for categorical column '{}'",
                        name
                    ))
                })?;
                let binned = self.apply_categorical_encoding(series, encoding_state)?;
                all_binned.push(binned);
            } else {
                eprintln!(
                    "[DEBUG process_for_inference] Column {}: '{}' - NUMERIC",
                    col_idx, name
                );
                // Numeric: use learned bin boundaries
                let info = &state.feature_info[col_idx];
                let binned = self.apply_numeric_binning(series, info)?;
                all_binned.push(binned);
            }
        }

        // Flatten to column-major layout
        let mut features = Vec::with_capacity(num_rows * all_binned.len());
        for binned_col in all_binned {
            features.extend(binned_col);
        }

        // Dummy targets for inference
        let targets = vec![0.0f32; num_rows];

        Ok(BinnedDataset::new(
            num_rows,
            features,
            targets,
            state.feature_info.clone(),
        ))
    }

    /// Process a numeric column: quantile binning
    fn process_numeric_column(
        &self,
        name: String,
        series: &Series,
    ) -> Result<(Vec<u8>, FeatureInfo)> {
        let mut values = self.series_to_f64(series)?;

        if name == "id" {
            eprintln!(
                "[DEBUG process_numeric_column TRAINING] Feature 'id': first 5 values: {:?}",
                &values[..values.len().min(5)]
            );
        }

        // Impute NaN values with median (more robust than mean for outliers)
        let non_nan_values: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        let impute_value = if !non_nan_values.is_empty() {
            // Compute median
            let mut sorted = non_nan_values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        } else {
            0.0 // If all values are NaN, use 0
        };

        // Replace NaN with imputed value
        for v in values.iter_mut() {
            if v.is_nan() {
                *v = impute_value;
            }
        }

        // Compute bin boundaries using T-Digest
        let boundaries = self.binner.compute_boundaries(&values);

        // Bin the values
        let binned: Vec<u8> = values
            .iter()
            .map(|&v| QuantileBinner::bin_value(v, &boundaries))
            .collect();

        let info = FeatureInfo {
            name,
            feature_type: FeatureType::Numeric,
            num_bins: (boundaries.len() + 1).min(255) as u8,
            bin_boundaries: boundaries,
        };

        Ok((binned, info))
    }

    /// Process a categorical column: CMS filter → Target Encode → Bin
    fn process_categorical_column(
        &self,
        name: String,
        series: &Series,
        targets: &[f64],
    ) -> Result<(Vec<u8>, FeatureInfo, CategoricalEncodingState)> {
        // Extract string values
        let categories = self.series_to_strings(series)?;

        // Step 1: CMS filtering for rare categories
        let mut filter = CategoryFilter::new(
            self.config.cms_eps,
            self.config.cms_confidence,
            self.config.min_category_count,
        );

        // Count all categories
        for cat in &categories {
            filter.count(cat);
        }

        // Collect unique categories and finalize
        let unique: Vec<String> = categories
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        filter.finalize(unique);

        // Filter rare categories to "unknown"
        let filtered: Vec<String> = categories
            .iter()
            .map(|c| filter.filter(c).to_string())
            .collect();

        // Create category mapping for serialization
        let category_mapping = CategoryMapping::from_filter(&filter);

        // Step 2: Ordered Target Encoding
        let mut encoder = OrderedTargetEncoder::new(self.config.target_encoding_smoothing);
        let encoded = encoder.encode_column(&filtered, targets);

        // Get encoding map for inference
        let encoding_map = encoder.get_encoding_map();

        // Step 3: Quantile binning of encoded values
        let boundaries = self.binner.compute_boundaries(&encoded);
        let binned: Vec<u8> = encoded
            .iter()
            .map(|&v| QuantileBinner::bin_value(v, &boundaries))
            .collect();

        let info = FeatureInfo {
            name: name.clone(),
            feature_type: FeatureType::Categorical,
            num_bins: (boundaries.len() + 1).min(255) as u8,
            bin_boundaries: boundaries.clone(),
        };

        let encoding_state = CategoricalEncodingState {
            name,
            category_mapping,
            encoding_map,
            bin_boundaries: boundaries,
        };

        Ok((binned, info, encoding_state))
    }

    /// Apply learned categorical encoding for inference
    fn apply_categorical_encoding(
        &self,
        series: &Series,
        state: &CategoricalEncodingState,
    ) -> Result<Vec<u8>> {
        let categories = self.series_to_strings(series)?;

        // Map categories to encoded values using learned mapping
        let encoded: Vec<f64> = categories
            .iter()
            .map(|cat| {
                // Check if category is in the mapping, otherwise use default
                let idx = state.category_mapping.get_index(cat);
                if idx == state.category_mapping.unknown_idx {
                    // Unknown category: use default encoding value
                    state.encoding_map.default_value
                } else {
                    // Known category: use learned encoding
                    state.encoding_map.encode(cat)
                }
            })
            .collect();

        // Bin using learned boundaries
        let binned: Vec<u8> = encoded
            .iter()
            .map(|&v| QuantileBinner::bin_value(v, &state.bin_boundaries))
            .collect();

        eprintln!("[DEBUG apply_categorical_encoding] Feature '{}': {} bins, first 5 encoded: {:?}, first 5 bins: {:?}",
            state.name, state.bin_boundaries.len(), &encoded[..encoded.len().min(5)], &binned[..binned.len().min(5)]);

        Ok(binned)
    }

    /// Apply learned numeric binning for inference
    fn apply_numeric_binning(&self, series: &Series, info: &FeatureInfo) -> Result<Vec<u8>> {
        let values = self.series_to_f64(series)?;

        eprintln!(
            "[DEBUG apply_numeric_binning] Feature '{}': {} boundaries, first 5 values: {:?}",
            info.name,
            info.bin_boundaries.len(),
            &values[..values.len().min(5)]
        );

        let binned: Vec<u8> = values
            .iter()
            .map(|&v| QuantileBinner::bin_value(v, &info.bin_boundaries))
            .collect();

        eprintln!(
            "[DEBUG apply_numeric_binning] First 5 bins: {:?}",
            &binned[..binned.len().min(5)]
        );

        Ok(binned)
    }

    /// Convert series to f64 vec
    fn series_to_f64(&self, series: &Series) -> Result<Vec<f64>> {
        series
            .cast(&DataType::Float64)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast to f64: {}", e)))?
            .f64()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get f64 chunked: {}", e)))?
            .into_iter()
            .map(|opt| Ok(opt.unwrap_or(f64::NAN)))
            .collect()
    }

    /// Convert series to String vec
    fn series_to_strings(&self, series: &Series) -> Result<Vec<String>> {
        let str_series = series
            .cast(&DataType::String)
            .map_err(|e| TreeBoostError::Data(format!("Failed to cast to String: {}", e)))?;

        let str_chunked = str_series
            .str()
            .map_err(|e| TreeBoostError::Data(format!("Failed to get str chunked: {}", e)))?;

        Ok(str_chunked
            .into_iter()
            .map(|opt| opt.unwrap_or("").to_string())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_numeric_only() {
        let df = df! {
            "feature1" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature2" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();

        let pipeline = DataPipeline::new(PipelineConfig::new().with_num_bins(4));
        let (dataset, state, _filtered_df) =
            pipeline.process_for_training(df, "target", None).unwrap();

        assert_eq!(dataset.num_rows(), 10);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(state.column_order.len(), 2);
        assert!(state.categorical_indices.is_empty());
    }

    #[test]
    fn test_pipeline_with_categoricals() {
        let df = df! {
            "numeric" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "category" => &["a", "a", "b", "b", "c", "c", "a", "b", "c", "rare"],
            "target" => &[10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 11.0, 21.0, 31.0, 5.0]
        }
        .unwrap();

        let pipeline = DataPipeline::new(
            PipelineConfig::new()
                .with_num_bins(4)
                .with_cms_params(0.01, 0.99, 2) // min_count=2 to filter "rare"
                .with_smoothing(1.0),
        );

        let (dataset, state, _filtered_df) = pipeline
            .process_for_training(df, "target", Some(&["category"]))
            .unwrap();

        assert_eq!(dataset.num_rows(), 10);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(state.categorical_indices, vec![1]); // "category" is second column
        assert_eq!(state.categorical_encodings.len(), 1);

        // Check that "rare" was filtered (count=1 < min_count=2)
        let cat_state = &state.categorical_encodings[0];
        assert!(!cat_state
            .category_mapping
            .category_to_idx
            .iter()
            .any(|(name, _)| name == "rare"));
    }

    #[test]
    fn test_pipeline_inference() {
        // Training data
        let train_df = df! {
            "numeric" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "category" => &["a", "a", "b", "b", "c", "c", "a", "b", "c", "a"],
            "target" => &[10.0, 12.0, 20.0, 22.0, 30.0, 32.0, 11.0, 21.0, 31.0, 13.0]
        }
        .unwrap();

        let pipeline = DataPipeline::new(
            PipelineConfig::new()
                .with_num_bins(4)
                .with_cms_params(0.01, 0.99, 2)
                .with_smoothing(1.0),
        );

        let (_train_dataset, state, _filtered_df) = pipeline
            .process_for_training(train_df, "target", Some(&["category"]))
            .unwrap();

        // Inference data (different values, including unseen category)
        let test_df = df! {
            "numeric" => &[2.5, 5.5, 8.5],
            "category" => &["a", "b", "unseen"]
        }
        .unwrap();

        let test_dataset = pipeline.process_for_inference(test_df, &state).unwrap();

        assert_eq!(test_dataset.num_rows(), 3);
        assert_eq!(test_dataset.num_features(), 2);

        // Verify bins are in valid range
        for row in 0..3 {
            for feat in 0..2 {
                let bin = test_dataset.get_bin(row, feat);
                assert!(bin < state.feature_info[feat].num_bins);
            }
        }
    }

    #[test]
    fn test_target_encoding_ordering() {
        // Test that ordered target encoding prevents leakage
        let df = df! {
            "category" => &["a", "a", "a", "b", "b", "b"],
            "target" => &[10.0, 20.0, 30.0, 100.0, 200.0, 300.0]
        }
        .unwrap();

        let pipeline = DataPipeline::new(
            PipelineConfig::new()
                .with_num_bins(4)
                .with_cms_params(0.01, 0.99, 1) // Keep all categories
                .with_smoothing(0.0), // No smoothing for clearer testing
        );

        let (dataset, state, _filtered_df) = pipeline
            .process_for_training(df, "target", Some(&["category"]))
            .unwrap();

        assert_eq!(dataset.num_rows(), 6);
        assert_eq!(state.categorical_encodings.len(), 1);

        // With ordered encoding:
        // Row 0 (a, 10): encoded with global mean of empty = 0
        // Row 1 (a, 20): encoded with a's prior mean = 10
        // Row 2 (a, 30): encoded with a's prior mean = (10+20)/2 = 15
        // etc.
        // This prevents the target from leaking into its own encoding
    }
}
