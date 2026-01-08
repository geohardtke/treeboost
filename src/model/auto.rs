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
use crate::dataset::{BinnedDataset, DataPipeline};
use crate::features::FeaturePlan;
use crate::model::{
    AutoBuilder, AutoConfig, AutoTrainedModel, BoostingMode, BuildPhaseTimes, BuildResult,
    TreeTuningResult, TuningLevel, UniversalModel,
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
    model: AutoTrainedModel,

    /// Boosting mode that was used
    mode: BoostingMode,

    /// Target column name used during training
    target_column: String,

    /// Confidence in mode selection
    mode_confidence: Option<Confidence>,

    /// Preprocessing plan that was applied
    preprocessing_plan: Option<PreprocessingPlan>,

    /// Feature engineering plan that was applied
    feature_plan: Option<FeaturePlan>,

    /// LTT tuning result (if applicable)
    ltt_tuning: Option<LttTuningResult>,

    /// Tree tuning result (if PureTree/RandomForest mode was used)
    tree_tuning: Option<TreeTuningResult>,

    /// Column profile from analysis
    column_profile: Option<DataFrameProfile>,

    /// Dataset analysis result
    analysis: Option<DatasetAnalysis>,

    /// Fitted pipeline state (CRITICAL for inference!)
    pipeline_state: Option<crate::dataset::PipelineState>,

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
            target_column: result.target_column,
            mode_confidence: result.mode_confidence,
            preprocessing_plan: result.preprocessing_plan,
            feature_plan: result.feature_plan,
            ltt_tuning: result.ltt_tuning,
            tree_tuning: result.tree_tuning,
            column_profile: result.column_profile,
            pipeline_state: result.pipeline_state,
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
    pub fn train_with_mode(df: &DataFrame, target_col: &str, mode: BoostingMode) -> Result<Self> {
        let builder = AutoBuilder::new().with_mode(mode);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Train with custom configuration
    pub fn train_with_config(df: &DataFrame, target_col: &str, config: AutoConfig) -> Result<Self> {
        let builder = AutoBuilder::with_config(config);
        let result = builder.fit(df, target_col)?;
        Ok(Self::from_build_result(result))
    }

    /// Predict on a DataFrame
    ///
    /// Returns predictions as a Vec<f32>.
    pub fn predict(&self, df: &DataFrame) -> Result<Vec<f32>> {
        // Convert DataFrame to BinnedDataset for prediction
        // Also get preprocessed DataFrame with encoded categoricals
        let (preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;

        match &self.model {
            AutoTrainedModel::Universal(model) => {
                // For LinearThenTree, use dual-representation inference if FeatureExtractor is stored
                if matches!(self.mode, crate::model::BoostingMode::LinearThenTree) {
                    if let Some(ref extractor) = model.feature_extractor() {
                        // CRITICAL: Extract from preprocessed_df (with encoded categoricals),
                        // NOT from original df (with String categoricals)
                        let (raw_features, _num_features) =
                            extractor.extract(&preprocessed_df, &self.target_column)?;

                        return Ok(model.predict_with_raw_features(&dataset, &raw_features));
                    }
                }

                Ok(model.predict(&dataset))
            }
            AutoTrainedModel::Ensemble(ensemble) => Ok(ensemble.predict(&dataset)),
            AutoTrainedModel::LttEnsemble(ltt) => {
                if let Some(ref extractor) = ltt.feature_extractor() {
                    let (raw_features, _) = extractor.extract(&preprocessed_df, &self.target_column)?;
                    Ok(ltt.predict_with_raw_features(&dataset, &raw_features))
                } else {
                    Ok(ltt.predict(&dataset))
                }
            }
        }
    }

    /// Predict using only the linear component (LinearThenTree mode only)
    ///
    /// For fair comparison between linear-only and full LinearThenTree.
    /// Uses the same preprocessing pipeline as the full model.
    ///
    /// # Returns
    /// - `Ok(Vec<f32>)`: Linear-only predictions (base + linear model)
    /// - `Err`: If model is not LinearThenTree mode
    pub fn predict_linear_only(&self, df: &DataFrame) -> Result<Vec<f32>> {
        if !matches!(self.mode, crate::model::BoostingMode::LinearThenTree) {
            return Err(TreeBoostError::Config(
                "predict_linear_only() only available for LinearThenTree mode".to_string(),
            ));
        }

        // Use same preprocessing as full prediction
        let (preprocessed_df, dataset) = self.prepare_dataset_for_prediction(df)?;

        match &self.model {
            AutoTrainedModel::Universal(model) => {
                // Extract features using FeatureExtractor
                if let Some(ref extractor) = model.feature_extractor() {
                    let (raw_features, _num_features) =
                        extractor.extract(&preprocessed_df, &self.target_column)?;

                    // Get linear-only predictions from the model
                    return Ok(model.predict_linear_only(&dataset, &raw_features)?);
                }

                Err(TreeBoostError::Config(
                    "LinearThenTree model missing FeatureExtractor - cannot predict".to_string(),
                ))
            }
            AutoTrainedModel::Ensemble(_) => {
                Err(TreeBoostError::Config(
                    "predict_linear_only() not available for ensemble models".to_string(),
                ))
            }
            AutoTrainedModel::LttEnsemble(ltt) => {
                // For LTT Ensemble, compute base + shrinkage * linear predictions
                if let Some(ref extractor) = ltt.feature_extractor() {
                    let (raw_features, _num_features) =
                        extractor.extract(&preprocessed_df, &self.target_column)?;

                    // Use LTT's predict_linear_only (base + linear with shrinkage)
                    return Ok(ltt.predict_linear_only(&dataset, &raw_features));
                }

                Err(TreeBoostError::Config(
                    "LTT Ensemble model missing FeatureExtractor - cannot predict".to_string(),
                ))
            }
        }
    }

    /// Predict on a BinnedDataset (for advanced users)
    pub fn predict_binned(&self, dataset: &BinnedDataset) -> Vec<f32> {
        match &self.model {
            AutoTrainedModel::Universal(model) => model.predict(dataset),
            AutoTrainedModel::Ensemble(ensemble) => ensemble.predict(dataset),
            AutoTrainedModel::LttEnsemble(ltt) => ltt.predict(dataset),
        }
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

    /// Get tree tuning result (if PureTree/RandomForest mode was used)
    pub fn tree_tuning(&self) -> Option<&TreeTuningResult> {
        self.tree_tuning.as_ref()
    }

    /// Get the underlying UniversalModel (if available)
    pub fn inner(&self) -> Option<&UniversalModel> {
        match &self.model {
            AutoTrainedModel::Universal(model) => Some(model),
            AutoTrainedModel::Ensemble(_) => None,
            AutoTrainedModel::LttEnsemble(_) => None,
        }
    }

    /// Get a comprehensive summary of the training process
    ///
    /// This shows the full "Smart Engineer" report explaining every decision
    pub fn summary(&self) -> String {
        let mut lines = vec![
            "┌─────────────────────────────────────────────────────────────────┐".to_string(),
            "│                  TreeBoost Pipeline Report                      │".to_string(),
            "└─────────────────────────────────────────────────────────────────┘".to_string(),
            "".to_string(),
        ];

        // Section 1: Data Profile
        if let Some(ref profile) = self.column_profile {
            lines.push("═══ DATA PROFILE ═══".to_string());
            lines.push(format!("  Rows: {}", profile.num_rows));
            lines.push(format!("  Columns: {} total", profile.columns.len()));
            lines.push(format!("    • Numeric: {}", profile.num_numeric));
            lines.push(format!("    • Categorical: {}", profile.num_categorical));
            lines.push(format!(
                "  Target: {} ({:?})",
                self.target_column, profile.task_type
            ));

            if !profile.drop_columns.is_empty() {
                lines.push("".to_string());
                lines.push(format!("  Dropped {} columns:", profile.drop_columns.len()));
                for dropped in &profile.drop_columns {
                    lines.push(format!("    • '{}' - {}", dropped.name, dropped.reason));
                }
            }
            lines.push("".to_string());
        }

        // Section 2: Preprocessing Decisions
        if let Some(ref plan) = self.preprocessing_plan {
            lines.push("═══ PREPROCESSING DECISIONS ═══".to_string());
            if !plan.reasoning.is_empty() {
                for reason in &plan.reasoning {
                    lines.push(format!("  • {}", reason));
                }
            } else {
                lines.push("  • No special preprocessing required".to_string());
            }
            lines.push("".to_string());
        }

        // Section 3: Feature Engineering
        if let Some(ref plan) = self.feature_plan {
            lines.push("═══ FEATURE ENGINEERING ═══".to_string());

            if !plan.polynomial_features.is_empty() {
                lines.push(format!(
                    "  Polynomial features ({}): ",
                    plan.polynomial_features.len()
                ));
                for feat in &plan.polynomial_features {
                    lines.push(format!("    • {}", feat));
                }
            }

            if !plan.ratio_pairs.is_empty() {
                lines.push(format!("  Ratio features ({}): ", plan.ratio_pairs.len()));
                for (f1, f2) in &plan.ratio_pairs {
                    lines.push(format!("    • {}/{}", f1, f2));
                }
            }

            if !plan.interaction_pairs.is_empty() {
                lines.push(format!(
                    "  Interaction features ({}): ",
                    plan.interaction_pairs.len()
                ));
                for (f1, f2) in plan.interaction_pairs.iter().take(5) {
                    lines.push(format!("    • {} × {}", f1, f2));
                }
                if plan.interaction_pairs.len() > 5 {
                    lines.push(format!(
                        "    ... and {} more",
                        plan.interaction_pairs.len() - 5
                    ));
                }
            }

            if !plan.reasoning.is_empty() {
                lines.push("".to_string());
                lines.push("  Reasoning:".to_string());
                for reason in &plan.reasoning {
                    lines.push(format!("    • {}", reason));
                }
            }
            lines.push("".to_string());
        }

        // Section 4: Mode Selection
        lines.push("═══ MODE SELECTION ═══".to_string());
        lines.push(format!("  Selected: {:?}", self.mode));
        lines.push(format!(
            "  Confidence: {:?}",
            self.mode_confidence
                .as_ref()
                .map(|c| format!("{:?}", c))
                .unwrap_or("N/A".to_string())
        ));

        if let Some(ref analysis) = self.analysis {
            lines.push("".to_string());
            lines.push("  Analysis Results:".to_string());
            lines.push(format!(
                "    • Linear R²: {:.4} ({})",
                analysis.linear_r2,
                if analysis.linear_r2 > 0.5 {
                    "Strong"
                } else if analysis.linear_r2 > 0.3 {
                    "Moderate"
                } else {
                    "Weak"
                }
            ));
            lines.push(format!(
                "    • Tree Gain: {:.4} ({})",
                analysis.tree_gain,
                if analysis.tree_gain > 0.3 {
                    "Strong"
                } else if analysis.tree_gain > 0.1 {
                    "Moderate"
                } else {
                    "Weak"
                }
            ));

            // Show the recommended mode from analysis
            let recommended_mode = analysis.recommend_mode();
            let reasoning = if analysis.linear_r2 > 0.5 && analysis.tree_gain > 0.1 {
                "Strong linear trend + residual structure → Hybrid approach"
            } else if analysis.linear_r2 > 0.5 {
                "Strong linear relationship → Linear model dominates"
            } else if analysis.tree_gain > 0.1 {
                "Non-linear patterns → Tree-based approach"
            } else {
                "Moderate signals → Pure tree model"
            };

            lines.push("".to_string());
            lines.push(format!("  Recommended: {:?}", recommended_mode));
            lines.push(format!("  Reasoning: {}", reasoning));
        }
        lines.push("".to_string());

        // Section 5: Tuning Results (if applicable)
        if let Some(ref tuning) = self.ltt_tuning {
            lines.push("═══ LTT TUNING RESULTS ═══".to_string());
            lines.push("  Linear Phase:".to_string());
            lines.push(format!("    • R²: {:.4}", tuning.linear_r2));
            lines.push(format!("    • Lambda: {:.4}", tuning.linear_params.lambda));
            lines.push(format!(
                "    • L1 Ratio: {:.4} ({})",
                tuning.linear_params.l1_ratio,
                if tuning.linear_params.l1_ratio == 0.0 {
                    "Ridge"
                } else if tuning.linear_params.l1_ratio == 1.0 {
                    "LASSO"
                } else {
                    "ElasticNet"
                }
            ));
            lines.push("".to_string());
            lines.push("  Tree Phase:".to_string());
            lines.push(format!("    • Max Depth: {}", tuning.tree_params.max_depth));
            lines.push(format!(
                "    • Learning Rate: {:.4}",
                tuning.tree_params.learning_rate
            ));
            lines.push(format!(
                "    • Num Rounds: {}",
                tuning.tree_params.num_rounds
            ));
            lines.push("".to_string());
            lines.push(format!("  Final RMSE: {:.4}", tuning.final_rmse));
            lines.push("".to_string());
        }

        // Section 6: Training Summary
        lines.push("═══ TRAINING SUMMARY ═══".to_string());
        lines.push(format!(
            "  Total Time: {:.3}s",
            self.build_time.as_secs_f64()
        ));
        lines.push("".to_string());
        lines.push("  Phase Breakdown:".to_string());
        lines.push(format!("    • Profiling: {:?}", self.phase_times.profiling));
        lines.push(format!(
            "    • Preprocessing: {:?}",
            self.phase_times.preprocessing
        ));
        lines.push(format!(
            "    • Feature Engineering: {:?}",
            self.phase_times.feature_engineering
        ));
        lines.push(format!("    • Analysis: {:?}", self.phase_times.analysis));
        lines.push(format!("    • Tuning: {:?}", self.phase_times.tuning));
        lines.push(format!("    • Training: {:?}", self.phase_times.training));
        lines.push("".to_string());

        lines.push(
            "┌─────────────────────────────────────────────────────────────────┐".to_string(),
        );
        lines.push(
            "│      TreeBoost: The Smart Engineer That Explains Itself         │".to_string(),
        );
        lines.push(
            "└─────────────────────────────────────────────────────────────────┘".to_string(),
        );

        lines.join("\n")
    }

    /// Prepare a DataFrame for prediction
    fn prepare_dataset_for_prediction(&self, df: &DataFrame) -> Result<(DataFrame, BinnedDataset)> {
        // CRITICAL: Use the fitted pipeline state from training!
        // Without this, predictions will be nonsense because the model expects
        // features encoded the same way as during training.
        let pipeline_state = self.pipeline_state.as_ref().ok_or_else(|| {
            TreeBoostError::Data(
                "AutoModel missing fitted pipeline state - cannot make predictions".to_string(),
            )
        })?;

        // Use DataPipeline to transform the test data using the fitted state
        let pipeline = DataPipeline::with_defaults();

        // process_for_inference() applies the learned encodings/scalers/binners
        // Returns (preprocessed_df, dataset) where preprocessed_df has encoded categoricals
        let (preprocessed_df, dataset) =
            pipeline.process_for_inference(df.clone(), pipeline_state)?;

        Ok((preprocessed_df, dataset))
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
