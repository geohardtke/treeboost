//! Realistic mode configuration for target leakage prevention
//!
//! Provides proper train/validation encoding to prevent target leakage
//! during hyperparameter tuning.

use crate::dataset::{BinnedDataset, DataPipeline, PipelineConfig};
use crate::{Result, TreeBoostError};
use polars::prelude::*;

/// Configuration for realistic mode (encoding per split)
#[derive(Clone)]
pub struct RealisticModeConfig {
    /// Pipeline configuration for encoding
    pub pipeline_config: PipelineConfig,
    /// Target column name
    pub target_column: String,
    /// Categorical column names (None = auto-detect)
    pub categorical_columns: Option<Vec<String>>,
}

impl RealisticModeConfig {
    /// Create a new realistic mode configuration
    pub fn new(
        pipeline_config: PipelineConfig,
        target_column: impl Into<String>,
        categorical_columns: Option<Vec<String>>,
    ) -> Self {
        Self {
            pipeline_config,
            target_column: target_column.into(),
            categorical_columns,
        }
    }
}

// =============================================================================
// Helper Functions for Realistic Mode
// =============================================================================

/// Extract target column from DataFrame as Vec<f32>
///
/// Handles casting from various numeric types to f32.
/// Returns an error if any NULL values are found in the target column.
pub(crate) fn extract_targets_from_df(df: &DataFrame, target_column: &str) -> Result<Vec<f32>> {
    let col = df.column(target_column).map_err(|e| {
        TreeBoostError::Data(format!(
            "Target column '{}' not found: {}",
            target_column, e
        ))
    })?;

    col.cast(&DataType::Float64)
        .map_err(|e| TreeBoostError::Data(format!("Failed to cast target to f64: {}", e)))?
        .f64()
        .map_err(|e| TreeBoostError::Data(format!("Failed to get f64 values: {}", e)))?
        .iter()
        .enumerate()
        .map(|(idx, opt)| {
            opt.ok_or_else(|| {
                TreeBoostError::Data(format!(
                    "NULL value found in target column '{}' at row {}",
                    target_column, idx
                ))
            })
            .map(|v| v as f32)
        })
        .collect()
}

/// Split DataFrame by indices into train and validation sets
pub(crate) fn split_dataframe_by_indices(
    df: &DataFrame,
    train_indices: &[usize],
    val_indices: &[usize],
) -> Result<(DataFrame, DataFrame)> {
    let train_idx: Vec<u32> = train_indices.iter().map(|&i| i as u32).collect();
    let val_idx: Vec<u32> = val_indices.iter().map(|&i| i as u32).collect();

    let train_df = df.take(&IdxCa::from_vec("idx".into(), train_idx))?;
    let val_df = df.take(&IdxCa::from_vec("idx".into(), val_idx))?;

    Ok((train_df, val_df))
}

/// Process train/val DataFrames through pipeline for realistic mode
///
/// Returns encoded datasets and validation targets.
pub(crate) fn encode_train_val_split(
    train_df: DataFrame,
    val_df: DataFrame,
    realistic_cfg: &RealisticModeConfig,
) -> Result<(BinnedDataset, BinnedDataset, Vec<f32>)> {
    let pipeline = DataPipeline::new(realistic_cfg.pipeline_config.clone());
    let cat_cols: Option<Vec<&str>> = realistic_cfg
        .categorical_columns
        .as_ref()
        .map(|cols| cols.iter().map(|s| s.as_str()).collect());

    // Extract validation targets FIRST (before consuming val_df)
    let val_targets = extract_targets_from_df(&val_df, &realistic_cfg.target_column)?;

    // Fit encoder on TRAIN ONLY
    let (train_dataset, pipeline_state, _) = pipeline.process_for_training(
        train_df,
        &realistic_cfg.target_column,
        cat_cols.as_deref(),
    )?;

    // Apply encoder to validation data (using train's encoding)
    // No clone needed since we extracted targets first
    let val_dataset = pipeline.process_for_inference(val_df, &pipeline_state)?;

    Ok((train_dataset, val_dataset, val_targets))
}

/// Encode the full dataset for final model training
///
/// This is used after tuning to train the final model on all data.
pub(crate) fn encode_full_dataset(
    df: DataFrame,
    realistic_cfg: &RealisticModeConfig,
) -> Result<BinnedDataset> {
    let pipeline = DataPipeline::new(realistic_cfg.pipeline_config.clone());
    let cat_cols: Option<Vec<&str>> = realistic_cfg
        .categorical_columns
        .as_ref()
        .map(|cols| cols.iter().map(|s| s.as_str()).collect());

    let (dataset, _pipeline_state, _) =
        pipeline.process_for_training(df, &realistic_cfg.target_column, cat_cols.as_deref())?;

    Ok(dataset)
}
