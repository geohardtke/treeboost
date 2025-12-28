//! Python bindings for tuner module
//!
//! Provides access to:
//! - AutoTuner: Main hyperparameter tuning interface
//! - TunerConfig, ParameterSpace, ParamBounds: Configuration
//! - TuningMode, OptimizationMetric, TaskType: Enum wrappers
//! - GridStrategy, EvalStrategy, ModelFormat: Strategy enums
//! - TrialResult, SearchHistory: Result types

mod autotuner;
mod callback;
mod config;
mod enums;
mod results;

use pyo3::prelude::*;

/// Register all tuner classes with the module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    enums::register(m)?;
    config::register(m)?;
    results::register(m)?;
    autotuner::register(m)?;
    Ok(())
}
