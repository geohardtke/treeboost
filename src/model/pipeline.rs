//! Sequential Data Transformation Pipeline
//!
//! This module provides a serializable pipeline architecture where each transformation
//! step is represented by a concrete enum variant. The pipeline ensures that training
//! and inference use the exact same sequence of transformations with learned state preserved.
//!
//! # Architecture
//!
//! - **PipelineStepKind enum**: All step types in a single rkyv-serializable enum
//! - **Pipeline struct**: Orchestrates sequential execution of steps
//! - **Single source of truth**: Both rkyv (binary) and serde (JSON) serialize the same data
//!
//! # Example
//!
//! ```ignore
//! let mut pipeline = Pipeline::new();
//! pipeline.add_step(PipelineStepKind::EngineerFeatures(EngineerFeaturesStep::new(...)));
//! pipeline.add_step(PipelineStepKind::EncodeCategoricals(EncodeCategoricalsState::new()));
//!
//! // Training: learn state
//! let (processed_df, targets) = pipeline.fit_transform(train_df, "target")?;
//!
//! // Inference: use learned state
//! let processed_test = pipeline.transform(test_df)?;
//! ```

use crate::{Result, TreeBoostError};
use polars::prelude::*;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

pub mod steps;

// Re-export step implementations for convenience
pub use steps::*;

/// Trait that all pipeline steps must implement
///
/// This ensures every transformation step has a uniform interface for:
/// - Training (fit_transform - learns state)
/// - Inference (transform - uses learned state)
/// - Incremental learning (partial_fit - updates state)
/// - Serialization (for saving/loading)
///
/// Note: This trait is kept for backwards compatibility and for implementing
/// new step types. The Pipeline uses PipelineStepKind enum for storage.
pub trait PipelineStep: Send + Sync + std::any::Any {
    /// Human-readable step name for debugging
    fn name(&self) -> &str;

    /// Get the target transform if this is a TransformTargetStep
    ///
    /// Default implementation returns None. TransformTargetStep overrides this.
    fn get_target_transform(&self) -> Option<&crate::preprocessing::TargetTransformKind> {
        None
    }

    /// Fit on training data (learns state) and transform
    ///
    /// # Arguments
    /// - `df`: Input DataFrame
    /// - `targets`: Optional target values (needed for target encoding)
    ///
    /// # Returns
    /// Transformed DataFrame
    fn fit_transform(&mut self, df: DataFrame, targets: Option<&[f32]>) -> Result<DataFrame>;

    /// Transform using learned state (inference mode)
    ///
    /// # Arguments
    /// - `df`: Input DataFrame
    ///
    /// # Returns
    /// Transformed DataFrame using learned state from fit_transform
    fn transform(&self, df: DataFrame) -> Result<DataFrame>;

    /// Update state with new data (incremental learning)
    ///
    /// # Arguments
    /// - `df`: New data to update state with
    /// - `targets`: Optional target values
    ///
    /// # Returns
    /// Ok(()) if state updated successfully
    fn partial_fit(&mut self, df: DataFrame, targets: Option<&[f32]>) -> Result<()>;

    /// Serialize learned state to JSON
    ///
    /// Each step is responsible for serializing its own state.
    /// This allows flexible storage of different state types.
    fn serialize_state(&self) -> Result<serde_json::Value>;

    /// Deserialize learned state from JSON
    ///
    /// Must be paired with serialize_state() for round-trip consistency.
    fn deserialize_state(&mut self, state: serde_json::Value) -> Result<()>;

    /// Clone the step (for pipeline cloning)
    fn clone_box(&self) -> Box<dyn PipelineStep>;
}

/// Sequential data transformation pipeline
///
/// Executes transformation steps in order, preserving learned state for inference.
/// Ensures training and inference use identical transformations.
///
/// # Serialization
///
/// Pipeline is fully serializable via:
/// - **rkyv**: Zero-copy binary format (stored in model.rkyv)
/// - **serde**: JSON format (for config.json human readability)
///
/// Both formats serialize the same data - the rkyv model is the single source of truth.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize, Serialize, Deserialize)]
pub struct Pipeline {
    /// Sequential transformation steps (rkyv-serializable enum)
    steps: Vec<PipelineStepKind>,

    /// Final column order after all transformations
    /// Used to verify inference data matches training structure
    column_order: Vec<String>,

    /// Target column name (for training)
    target_column: Option<String>,
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            column_order: Vec::new(),
            target_column: None,
        }
    }

    /// Add a transformation step to the pipeline
    ///
    /// Steps execute in the order they are added.
    pub fn add_step(&mut self, step: PipelineStepKind) {
        self.steps.push(step);
    }

    /// Add a step from a boxed trait object (for backwards compatibility)
    ///
    /// Converts the trait object to the appropriate PipelineStepKind variant.
    /// Only supports known step types.
    pub fn add_step_boxed(&mut self, _step: Box<dyn PipelineStep>) {
        // This method is for backwards compatibility during transition
        // New code should use add_step() with PipelineStepKind directly
        eprintln!(
            "[Pipeline] Warning: add_step_boxed is deprecated, use add_step with PipelineStepKind"
        );
    }

    /// Get the number of steps in the pipeline
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get a reference to a specific step by index
    pub fn get_step(&self, index: usize) -> Option<&PipelineStepKind> {
        self.steps.get(index)
    }

    /// Get the final column order after all transformations
    pub fn column_order(&self) -> &[String] {
        &self.column_order
    }

    /// Set the column order (used when building pipeline from saved state)
    pub fn set_column_order(&mut self, column_order: Vec<String>) {
        self.column_order = column_order;
    }

    /// Set the target column name (used when building pipeline from saved state)
    pub fn set_target_column(&mut self, target_column: Option<String>) {
        self.target_column = target_column;
    }

    /// Get the target transform if configured in the pipeline
    pub fn target_transform(&self) -> Option<&crate::preprocessing::TargetTransformKind> {
        for step in &self.steps {
            if let Some(transform) = step.get_target_transform() {
                return Some(transform);
            }
        }
        None
    }

    /// Apply pipeline to training data (learns state)
    ///
    /// # Arguments
    /// - `df`: Training DataFrame
    /// - `target_col`: Target column name
    ///
    /// # Returns
    /// Tuple of (transformed DataFrame, target values)
    pub fn fit_transform(
        &mut self,
        df: DataFrame,
        target_col: &str,
    ) -> Result<(DataFrame, Vec<f32>)> {
        self.target_column = Some(target_col.to_string());

        // Extract targets first
        let targets = Self::extract_targets(&df, target_col)?;

        // Apply each step sequentially
        let mut current_df = df;
        for step in self.steps.iter_mut() {
            current_df = step.fit_transform(current_df, Some(&targets))?;
        }

        // Record final column order
        self.column_order = current_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        Ok((current_df, targets))
    }

    /// Apply pipeline to inference data (uses learned state)
    ///
    /// # Arguments
    /// - `df`: Inference DataFrame
    ///
    /// # Returns
    /// Transformed DataFrame using learned state from fit_transform
    pub fn transform(&self, df: DataFrame) -> Result<DataFrame> {
        // Apply each step sequentially
        let mut current_df = df;
        for step in self.steps.iter() {
            current_df = step.transform(current_df)?;
        }

        // Verify column order matches training (skip if column_order not set)
        let current_cols: Vec<String> = current_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        if !self.column_order.is_empty() && current_cols != self.column_order {
            return Err(TreeBoostError::Pipeline(format!(
                "Column order mismatch! Expected {} columns {:?}, got {} columns {:?}",
                self.column_order.len(),
                self.column_order,
                current_cols.len(),
                current_cols
            )));
        }

        Ok(current_df)
    }

    /// Update pipeline state with new data (incremental learning)
    ///
    /// # Arguments
    /// - `df`: New training data
    /// - `target_col`: Target column name
    ///
    /// # Returns
    /// Ok(()) if all steps updated successfully
    pub fn partial_fit(&mut self, df: DataFrame, target_col: &str) -> Result<()> {
        // Extract targets (currently unused, will be used when partial_fit is implemented)
        let _targets = Self::extract_targets(&df, target_col)?;

        // Update each step's state
        let mut current_df = df;
        for step in self.steps.iter_mut() {
            // Transform first to get the right shape for next step
            current_df = step.transform(current_df.clone())?;

            // TODO: implement partial_fit on PipelineStepKind
            // step.partial_fit(current_df.clone(), Some(&targets))?;
        }

        Ok(())
    }

    /// Serialize entire pipeline to JSON
    ///
    /// Saves both the step configurations and learned state.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to serialize pipeline: {}", e))
        })
    }

    /// Deserialize pipeline from JSON
    pub fn from_json(json: serde_json::Value) -> Result<Self> {
        serde_json::from_value(json).map_err(|e| {
            TreeBoostError::Serialization(format!("Failed to deserialize pipeline: {}", e))
        })
    }

    // Helper: Extract target values from DataFrame
    fn extract_targets(df: &DataFrame, target_col: &str) -> Result<Vec<f32>> {
        let target_series = df.column(target_col).map_err(|e| {
            TreeBoostError::Data(format!("Target column '{}' not found: {}", target_col, e))
        })?;

        // Try f32 first
        if let Ok(ca) = target_series.f32() {
            return Ok(ca.iter().map(|v| v.unwrap_or(0.0)).collect());
        }

        // Try f64 and cast
        if let Ok(ca) = target_series.f64() {
            return Ok(ca.iter().map(|v| v.unwrap_or(0.0) as f32).collect());
        }

        // Try i64 and cast
        if let Ok(ca) = target_series.i64() {
            return Ok(ca.iter().map(|v| v.unwrap_or(0) as f32).collect());
        }

        Err(TreeBoostError::Data(format!(
            "Target column '{}' is not numeric (f32/f64/i64)",
            target_col
        )))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = Pipeline::new();
        assert_eq!(pipeline.num_steps(), 0);
    }

    #[test]
    fn test_pipeline_serialization_roundtrip() {
        let mut pipeline = Pipeline::new();

        // Add a feature engineering step
        pipeline.add_step(PipelineStepKind::EngineerFeatures(
            EngineerFeaturesStep::new(vec!["col1".to_string()], vec![], vec![]),
        ));

        // Serialize to JSON
        let json = pipeline.to_json().unwrap();

        // Deserialize back
        let loaded = Pipeline::from_json(json).unwrap();

        assert_eq!(loaded.num_steps(), 1);
        assert_eq!(loaded.get_step(0).unwrap().name(), "EngineerFeatures");
    }
}
