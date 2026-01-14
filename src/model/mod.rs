//! High-level model abstractions
//!
//! This module provides:
//! - `UniversalModel`: Unified boosting framework supporting multiple modes
//! - `AutoBuilder`: High-level AutoML interface for simplified training
//! - `AutoModel`: Self-contained trained model from AutoBuilder
//! - `Pipeline`: Sequential data transformation pipeline with learned state

mod auto;
mod builder;
mod config;
mod pipeline;
mod progress;
mod tuning;
mod universal;

pub use auto::{AutoModel, AutoModelUpdateReport};
pub use builder::AutoBuilder;
pub use config::{
    AutoConfig, AutoEnsembleConfig, AutoEnsembleMethod, BuildPhaseTimes, BuildResult, EnsembleMode,
    FeatureEngineeringMode, PreprocessingMode, TargetBoundConfig, TreeTunerConfig, TreeTunerPreset,
    TreeTuningResult, TuningLevel,
};
pub use progress::{
    ConsoleProgress, ProgressCallback, ProgressUpdate, QuietProgress, TrainingPhase,
};
pub use pipeline::{
    BinNumericFeaturesState, BinNumericFeaturesStep, CategoryEncoding, CustomFeature,
    CustomFeaturesStep, DropColumnsStep, EncodeCategoricalsState, EncodeCategoricalsStep,
    EngineerFeaturesStep, EngineerTimeSeriesFeaturesStep, ExtractLinearFeaturesStep, FeatureOp,
    FormulaBuilder, LutMapping, Pipeline, PipelineStep, PipelineStepKind, TransformTargetStep,
    TrigFeature, TrigFunc,
};
pub use universal::{
    BoostingMode, IncrementalUpdateReport, ModeSelection, StackingStrategy, UniversalConfig,
    UniversalModel, UniversalPreset,
};
