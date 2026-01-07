//! Hyperparameter tuning methods for AutoBuilder
//!
//! This module contains all tuning logic for different boosting modes:
//! - Tree-based models (PureTree, RandomForest) via AutoTuner
//! - LinearThenTree via LttTuner
//! - Parameter extraction and configuration creation

use crate::analysis::{DataFrameProfile, TaskType};
use crate::booster::GBDTModel;
use crate::dataset::BinnedDataset;
use crate::learner::TreeConfig;
use crate::model::{BoostingMode, UniversalConfig};
use crate::model::config::{AutoConfig, TreeTunerConfig, TreeTuningResult, TuningLevel};
use crate::tuner::ltt::{LttTuner, LttTunerConfig, LttTuningResult};
use crate::tuner::{AutoTuner, EvalStrategy, GridStrategy, ParamBounds, ParameterSpace, SearchHistory, TunerConfig, TuningMode};
use crate::{Result, TreeBoostError};
use polars::prelude::*;

/// Tune hyperparameters for the selected boosting mode
pub(super) fn tune_hyperparameters(
    config: &AutoConfig,
    dataset: &BinnedDataset,
    mode: BoostingMode,
    df: &DataFrame,
    target_col: &str,
    profile: &DataFrameProfile,
) -> Result<(UniversalConfig, Option<LttTuningResult>, Option<TreeTuningResult>)> {
    match config.tuning_level {
        TuningLevel::None => {
            // No tuning - use defaults
            let univ_config = UniversalConfig::default().with_mode(mode);
            Ok((univ_config, None, None))
        }
        _ => {
            // Tune based on mode
            match mode {
                BoostingMode::LinearThenTree => {
                    let (univ_config, ltt_result) = tune_ltt(config, df, target_col, profile)?;
                    Ok((univ_config, ltt_result, None))
                }
                BoostingMode::PureTree | BoostingMode::RandomForest => {
                    // Run proper AutoTuner for tree-based models
                    let (univ_config, tree_result) = tune_tree_model(config, dataset, mode, profile)?;
                    Ok((univ_config, None, tree_result))
                }
            }
        }
    }
}

/// Tune tree-based models (PureTree, RandomForest) using AutoTuner
fn tune_tree_model(
    config: &AutoConfig,
    dataset: &BinnedDataset,
    mode: BoostingMode,
    profile: &DataFrameProfile,
) -> Result<(UniversalConfig, Option<TreeTuningResult>)> {
    if config.verbose {
        println!("  [Tuning] Running AutoTuner for {:?} mode...", mode);
    }

    // Get tuning configuration based on tuning level
    let tuner_cfg = match config.tuning_level {
        TuningLevel::Quick => TreeTunerConfig::quick(),
        TuningLevel::Standard => TreeTunerConfig::standard(),
        TuningLevel::Thorough => TreeTunerConfig::thorough(),
        TuningLevel::None => return Ok((UniversalConfig::default().with_mode(mode), None)),
    };

    // Define parameter space from config
    let param_space = ParameterSpace::new()
        .with_param(
            "max_depth",
            ParamBounds::discrete(tuner_cfg.max_depth_range.0, tuner_cfg.max_depth_range.1),
            6.0,
        )
        .with_param(
            "learning_rate",
            ParamBounds::log_continuous(tuner_cfg.learning_rate_range.0, tuner_cfg.learning_rate_range.1),
            0.1,
        )
        .with_param("subsample", ParamBounds::continuous(0.6, 1.0), 0.8)
        .with_param("lambda", ParamBounds::continuous(0.1, 5.0), 1.0)
        .with_param("entropy_weight", ParamBounds::continuous(0.0, 0.3), 0.1);

    // Base config for tuning - select loss function based on task type
    use crate::booster::GBDTConfig;
    let base_config = match &profile.task_type {
        TaskType::Regression => {
            GBDTConfig::new()
                .with_mse_loss()
                .with_min_samples_leaf(5)
                .with_seed(42)
        }
        TaskType::BinaryClassification => {
            GBDTConfig::new()
                .with_binary_logloss()
                .with_min_samples_leaf(5)
                .with_seed(42)
        }
        TaskType::MultiClassification { num_classes } => {
            GBDTConfig::new()
                .with_multiclass_logloss(*num_classes)
                .with_min_samples_leaf(5)
                .with_seed(42)
        }
    };

    // Configure tuner using TreeTunerConfig
    let tuner_config = TunerConfig::new()
        .with_iterations(tuner_cfg.n_iterations)
        .with_grid_strategy(GridStrategy::LatinHypercube {
            n_samples: tuner_cfg.n_samples,
        })
        .with_eval_strategy(EvalStrategy::conformal_90(tuner_cfg.validation_ratio))
        .with_tuning_mode(TuningMode::Optimistic) // Use pre-encoded data
        .with_num_rounds(tuner_cfg.max_rounds)
        .with_early_stopping(tuner_cfg.early_stopping_rounds, tuner_cfg.validation_ratio)
        .with_improvement_threshold(tuner_cfg.improvement_threshold)
        .with_min_f1_score(tuner_cfg.min_f1_score)
        .with_verbose(false); // Quiet internal logging

    let mut tuner = AutoTuner::<GBDTModel>::new(base_config)
        .with_config(tuner_config)
        .with_space(param_space)
        .with_seed(123);

    // Run tuning
    let (best_gbdt_config, history): (GBDTConfig, SearchHistory) = tuner.tune(dataset)?;

    // Convert GBDT config to UniversalConfig
    let tree_config = TreeConfig::default()
        .with_max_depth(best_gbdt_config.max_depth)
        .with_lambda(best_gbdt_config.lambda)
        .with_entropy_weight(best_gbdt_config.entropy_weight);

    let universal_config = UniversalConfig::default()
        .with_mode(mode)
        .with_num_rounds(best_gbdt_config.num_rounds)
        .with_learning_rate(best_gbdt_config.learning_rate)
        .with_subsample(best_gbdt_config.subsample)
        .with_tree_config(tree_config);

    // Extract tuning results
    let tuning_result = history.best().map(|best| TreeTuningResult {
        num_trials: history.len(),
        best_metric: best.val_metric,
        best_params: best.params.clone(),
    });

    if config.verbose {
        if let Some(ref result) = tuning_result {
            println!("  [Tuning] Best validation metric: {:.6}", result.best_metric);
            println!("  [Tuning] Best params: {:?}", result.best_params);
        }
    }

    Ok((universal_config, tuning_result))
}

/// Tune LinearThenTree mode
fn tune_ltt(
    config: &AutoConfig,
    df: &DataFrame,
    target_col: &str,
    profile: &DataFrameProfile,
) -> Result<(UniversalConfig, Option<LttTuningResult>)> {
    // Extract raw features for linear tuning
    let (features, targets, num_features) = extract_raw_features(df, target_col, profile)?;

    // Create tuner config based on tuning level
    let tuner_config = match config.tuning_level {
        TuningLevel::Quick => LttTunerConfig::quick(),
        TuningLevel::Standard => LttTunerConfig::default(),
        TuningLevel::Thorough => LttTunerConfig::thorough(),
        TuningLevel::None => LttTunerConfig::quick(), // Should not reach here
    };

    let tuner = LttTuner::new(tuner_config);
    let result = tuner.tune(&features, num_features, &targets)?;

    // Build UniversalConfig from tuning result
    let tree_config = TreeConfig::default()
        .with_max_depth(result.tree_params.max_depth as usize);

    let univ_config = UniversalConfig::default()
        .with_mode(BoostingMode::LinearThenTree)
        .with_learning_rate(result.tree_params.learning_rate)
        .with_num_rounds(result.tree_params.num_rounds as usize)
        .with_tree_config(tree_config)
        .with_linear_config(result.linear_params.to_config());

    Ok((univ_config, Some(result)))
}

/// Extract raw features from DataFrame for linear model tuning
///
/// NOTE: This is a simplified extraction for LTT hyperparameter tuning.
/// - Only numeric features are used (categorical features need encoding first)
/// - Columns marked for dropping in the profile are excluded
/// - For production prediction, use the full DataPipeline with preprocessing
fn extract_raw_features(
    df: &DataFrame,
    target_col: &str,
    profile: &DataFrameProfile,
) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let num_rows = df.height();

    // Build set of columns to drop based on profile
    let drop_cols: std::collections::HashSet<String> = profile
        .drop_columns
        .iter()
        .map(|d| d.name.clone())
        .collect();

    // Get numeric columns only, excluding dropped columns
    let numeric_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|&name| {
            let name_str = name.as_str();

            // Skip target column
            if name_str == target_col {
                return false;
            }

            // Skip columns marked for dropping
            if drop_cols.contains(name_str) {
                return false;
            }

            // Keep only numeric types
            match df.column(name_str) {
                Ok(col) => matches!(
                    col.dtype(),
                    DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
                ),
                Err(_) => false,
            }
        })
        .map(|s| s.to_string())
        .collect();

    let num_features = numeric_cols.len();
    if num_features == 0 {
        return Err(TreeBoostError::Data(
            format!(
                "No numeric features found for linear model tuning. \
                 Dataset has {} columns, {} dropped (constant/ID/text), {} remaining. \
                 Categorical features need encoding first - consider using DataPipeline.",
                df.width(),
                drop_cols.len(),
                df.width() - drop_cols.len() - 1 // -1 for target
            )
        ));
    }

    // Extract features (row-major)
    let mut features = Vec::with_capacity(num_rows * num_features);
    for row_idx in 0..num_rows {
        for col_name in &numeric_cols {
            let col = df.column(col_name)?;
            let val = col.get(row_idx).map_err(|e| TreeBoostError::Data(e.to_string()))?;
            let f_val = match val {
                AnyValue::Float32(v) => v,
                AnyValue::Float64(v) => v as f32,
                AnyValue::Int32(v) => v as f32,
                AnyValue::Int64(v) => v as f32,
                AnyValue::Null => 0.0, // Simple imputation
                _ => 0.0,
            };
            features.push(f_val);
        }
    }

    // Extract targets
    let target_col_series = df.column(target_col)?;
    let targets: Vec<f32> = (0..num_rows)
        .map(|i| {
            match target_col_series.get(i) {
                Ok(val) => match val {
                    AnyValue::Float32(v) => v,
                    AnyValue::Float64(v) => v as f32,
                    AnyValue::Int32(v) => v as f32,
                    AnyValue::Int64(v) => v as f32,
                    _ => 0.0,
                },
                Err(_) => 0.0,
            }
        })
        .collect();

    Ok((features, targets, num_features))
}

/// Create default config for non-LTT modes (fallback when tuning is disabled)
pub(super) fn create_config_for_mode(mode: BoostingMode, tuning_level: TuningLevel) -> UniversalConfig {
    let (num_rounds, learning_rate, max_depth) = match tuning_level {
        TuningLevel::Quick => (100, 0.1, 4),
        TuningLevel::Standard => (500, 0.1, 6),
        TuningLevel::Thorough => (1000, 0.05, 8),
        TuningLevel::None => (100, 0.1, 6),
    };

    let tree_config = TreeConfig::default()
        .with_max_depth(max_depth);

    UniversalConfig::default()
        .with_mode(mode)
        .with_num_rounds(num_rounds)
        .with_learning_rate(learning_rate)
        .with_tree_config(tree_config)
}
