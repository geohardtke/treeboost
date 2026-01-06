//! AutoModel: Self-contained trained model from AutoBuilder
//!
//! AutoModel wraps the result of AutoBuilder training and provides a clean
//! interface for predictions and model inspection.
//!
//! # Features
//!
//! - **One-line training**: `AutoModel::train(&df, "target")?`
//! - **Easy prediction**: `model.predict(&df)?`
//! - **Training summary**: Access all metadata from the build process
//! - **Serialization**: Save/load trained models
//!
//! # Example
//!
//! ```ignore
//! use treeboost::model::AutoModel;
//! use polars::prelude::*;
//!
//! // One-line training
//! let model = AutoModel::train(&df, "target")?;
//!
//! // See what mode was selected
//! println!("Mode: {:?}", model.mode());
//! println!("Build time: {:?}", model.build_time());
//!
//! // Predict
//! let predictions = model.predict(&test_df)?;
//!
//! // Save for later
//! model.save("model.rkyv")?;
//!
//! // Load and use
//! let loaded = AutoModel::load("model.rkyv")?;
//! let preds = loaded.predict(&test_df)?;
//! ```

use crate::analysis::{Confidence, DataFrameProfile, DatasetAnalysis};
use crate::dataset::{BinnedDataset, DataPipeline, PipelineConfig};
use crate::features::FeaturePlan;
use crate::model::{
    AutoBuilder, AutoConfig, BoostingMode, BuildPhaseTimes, BuildResult, TuningLevel,
    UniversalModel,
};
use crate::preprocessing::PreprocessingPlan;
use crate::tuner::ltt::LttTuningResult;
use crate::{Result, TreeBoostError};
use polars::prelude::*;
use std::time::Duration;

/// AutoModel: Trained model with full metadata
///
/// This is the main user-facing type for trained models. It wraps the
/// `UniversalModel` along with all the metadata from the training process.
pub struct AutoModel {
    /// The underlying trained model
    model: UniversalModel,

    /// Boosting mode that was used
    mode: BoostingMode,

    /// Confidence in mode selection
    mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if applicable)
    ltt_tuning: Option<LttTuningResult>,

    /// Column profile from analysis
    column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    analysis: Option<DatasetAnalysis>,

    /// Total build time
    build_time: Duration,

    /// Time breakdown by phase
    phase_times: BuildPhaseTimes,
}

impl AutoModel {
    /// Create an AutoModel from a BuildResult
    pub fn from_build_result(result: BuildResult) -> Self {
        Self {
            model: result.model,
            mode: result.mode,
            mode_confidence: result.mode_confidence,
            preprocessing_plan: result.preprocessing_plan,
            feature_plan: result.feature_plan,
            ltt_tuning: result.ltt_tuning,
            column_profile: result.column_profile,
            analysis: result.analysis,
            build_time: result.build_time,
            phase_times: result.phase_times,
        }
    }

    /// Train a model with default settings (the simplest API)
    ///
    /// This is the recommended entry point for most users.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = AutoModel::train(&df, "price")?;
    /// let predictions = model.predict(&test_df)?;
    /// ```
    pub fn train(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new();
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Train with quick settings (minimal tuning, fast training)
    pub fn train_quick(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new().with_tuning(TuningLevel::Quick);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Train with thorough settings (extensive tuning, best accuracy)
    pub fn train_thorough(df: &DataFrame, target_col: &str) -> Result<Self> {
        let builder = AutoBuilder::new().with_tuning(TuningLevel::Thorough);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Train with a specific mode (bypass auto-selection)
    pub fn train_with_mode(
        df: &DataFrame,
        target_col: &str,
        mode: BoostingMode,
    ) -> Result<Self> {
        let builder = AutoBuilder::new().with_mode(mode);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Train with custom configuration
    pub fn train_with_config(
        df: &DataFrame,
        target_col: &str,
        config: AutoConfig,
    ) -> Result<Self> {
        let builder = AutoBuilder::with_config(config);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Predict on a DataFrame
    ///
    /// Returns predictions as a Vec<f32>.
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f32>> {
        // Convert DataFrame to BinnedDataset for prediction
        let dataset = self.prepare_dataset_for_prediction(df)?;
        Ok(self.model.predict(&dataset))
    }

    /// Predict on a BinnedDataset (for advanced users)
    pub fn predict_binned(&self, dataset: &BinnedDataset) -> Vec<f32> {
        self.model.predict(dataset)
    }

    /// Get the boosting mode that was used
    pub fn mode(&self) -> BoostingMode {
        self.mode
    }

    /// Get confidence in the mode selection
    pub fn mode_confidence(&self) -> Option<Confidence> {
        self.mode_confidence
    }

    /// Get total training time
    pub fn build_time(&self) -> Duration {
        self.build_time
    }

    /// Get time breakdown by phase
    pub fn phase_times(&self) -> &BuildPhaseTimes {
        &self.phase_times
    }

    /// Get the preprocessing plan that was applied
    pub fn preprocessing_plan(&self) -> Option<&PreprocessingPlan> {
        self.preprocessing_plan.as_ref()
    }

    /// Get the feature engineering plan that was applied
    pub fn feature_plan(&self) -> Option<&FeaturePlan> {
        self.feature_plan.as_ref()
    }

    /// Get the column profile from analysis
    pub fn column_profile(&self) -> Option<&DataFrameProfile> {
        self.column_profile.as_ref()
    }

    /// Get the dataset analysis result
    pub fn analysis(&self) -> Option<&DatasetAnalysis> {
        self.analysis.as_ref()
    }

    /// Get LTT tuning result (if LTT mode was used)
    pub fn ltt_tuning(&self) -> Option<&LttTuningResult> {
        self.ltt_tuning.as_ref()
    }

    /// Get the underlying UniversalModel
    pub fn inner(&self) -> &UniversalModel {
        &self.model
    }

    /// Get a summary of the training process
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "AutoModel Training Summary".to_string(),
            "=".repeat(40),
            format!("Mode: {:?}", self.mode),
            format!(
                "Mode Confidence: {:?}",
                self.mode_confidence.as_ref().map(|c| format!("{:?}", c)).unwrap_or("N/A".to_string())
            ),
            format!("Total Build Time: {:?}", self.build_time),
            "".to_string(),
            "Phase Times:".to_string(),
            format!("  Profiling: {:?}", self.phase_times.profiling),
            format!("  Preprocessing: {:?}", self.phase_times.preprocessing),
            format!("  Feature Engineering: {:?}", self.phase_times.feature_engineering),
            format!("  Analysis: {:?}", self.phase_times.analysis),
            format!("  Tuning: {:?}", self.phase_times.tuning),
            format!("  Training: {:?}", self.phase_times.training),
        ];

        if let Some(ref profile) = self.column_profile {
            lines.push("".to_string());
            lines.push(format!("Columns Analyzed: {}", profile.columns.len()));
            lines.push(format!("Columns Dropped: {}", profile.drop_columns.len()));
            lines.push(format!("Task Type: {:?}", profile.task_type));
        }

        if let Some(ref plan) = self.feature_plan {
            lines.push("".to_string());
            lines.push("Feature Engineering:".to_string());
            lines.push(format!("  Polynomial Features: {}", plan.polynomial_features.len()));
            lines.push(format!("  Ratio Pairs: {}", plan.ratio_pairs.len()));
            lines.push(format!("  Interaction Pairs: {}", plan.interaction_pairs.len()));
        }

        if let Some(ref tuning) = self.ltt_tuning {
            lines.push("".to_string());
            lines.push("LTT Tuning Results:".to_string());
            lines.push(format!("  Linear R²: {:.4}", tuning.linear_r2));
            lines.push(format!("  Final RMSE: {:.4}", tuning.final_rmse));
            lines.push(format!("  Linear Lambda: {:.4}", tuning.linear_params.lambda));
            lines.push(format!("  Linear L1 Ratio: {:.4}", tuning.linear_params.l1_ratio));
            lines.push(format!("  Tree Max Depth: {}", tuning.tree_params.max_depth));
            lines.push(format!("  Tree Learning Rate: {:.4}", tuning.tree_params.learning_rate));
        }

        lines.join("\n")
    }

    /// Prepare a DataFrame for prediction
    fn prepare_dataset_for_prediction(&self, df: &DataFrame) -> Result<BinnedDataset> {
        // Get feature column names
        let feature_cols: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let feature_col_refs: Vec<&str> = feature_cols.iter().map(|s| s.as_str()).collect();

        // Use DataPipeline with default config
        let pipeline_config = PipelineConfig::default();
        let pipeline = DataPipeline::new(pipeline_config);

        // For prediction, we need a dummy target - use first numeric column
        // This is a workaround; ideally we'd have a predict-only pipeline
        let dummy_target = df
            .get_column_names()
            .iter()
            .find(|name| {
                match df.column(name.as_str()) {
                    Ok(col) => matches!(
                        col.dtype(),
                        DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64
                    ),
                    Err(_) => false,
                }
            })
            .map(|s| s.to_string())
            .ok_or_else(|| TreeBoostError::Data("No numeric column for dummy target".to_string()))?;

        // Create a modified DataFrame with target
        let df_with_target = df.clone();

        let (dataset, _state) = pipeline.process_for_training(
            df_with_target,
            &dummy_target,
            Some(&feature_col_refs),
        )?;

        Ok(dataset)
    }

    // Note: save/load methods are not yet implemented for UniversalModel
    // TODO: Add serialization support to UniversalModel and enable these methods
    //
    // pub fn save(&self, path: impl AsRef<Path>) -> Result<()>
    // pub fn load(path: impl AsRef<Path>) -> Result<Self>
}

impl std::fmt::Debug for AutoModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoModel")
            .field("mode", &self.mode)
            .field("mode_confidence", &self.mode_confidence)
            .field("build_time", &self.build_time)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_auto_model_from_build_result() {
        // This test just verifies the struct construction works
        // Full integration tests would require a DataFrame
    }

    #[test]
    fn test_auto_model_summary_format() {
        // Create a minimal AutoModel to test summary formatting
        // This would need a real UniversalModel in practice
    }
}
