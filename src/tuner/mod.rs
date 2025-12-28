//! AutoTuner for hyperparameter optimization
//!
//! This module provides an automated hyperparameter tuning system using
//! an Iterative Grid Search (Auto-Zoom) approach. The tuner progressively
//! narrows the search space around the best-performing configurations.
//!
//! # Features
//!
//! - **Zero-copy dataset reuse**: All trials share the same `BinnedDataset`
//! - **Flexible evaluation**: Holdout or K-fold cross-validation
//! - **Smart parallelization**: CPU trials run in parallel, GPU trials run sequentially
//! - **Progress callbacks**: Monitor tuning progress in real-time
//! - **Multiple grid strategies**: Cartesian, Latin Hypercube, or Random sampling
//!
//! # Example
//!
//! ```ignore
//! use treeboost::tuner::{AutoTuner, TunerConfig, ParameterSpace, EvalStrategy};
//! use treeboost::GBDTConfig;
//!
//! // Create base configuration
//! let base_config = GBDTConfig::default();
//!
//! // Create tuner with custom settings
//! let config = TunerConfig::new()
//!     .with_iterations(3)
//!     .with_eval_strategy(EvalStrategy::holdout(0.2));
//!
//! let mut tuner = AutoTuner::new(base_config)
//!     .with_config(config)
//!     .with_callback(|trial, current, total| {
//!         println!("Trial {}/{}: loss = {:.5}", current, total, trial.val_metric);
//!     });
//!
//! let (best_config, history) = tuner.tune(&dataset)?;
//! ```
//!
//! # Algorithm
//!
//! 1. Start with a center point (default or user-specified hyperparameters)
//! 2. Generate a grid of candidates around the center
//! 3. Evaluate each candidate using holdout or K-fold CV
//! 4. Select the best-performing candidate as the new center
//! 5. Reduce the search radius (zoom in)
//! 6. Repeat for N iterations
//!
//! This approach efficiently explores the hyperparameter space by starting
//! with a coarse search and progressively refining around promising regions.

// Submodules
mod autotuner;
mod config;
mod history;
mod metrics;
mod realistic;
mod trial;

// Re-exports from config
pub use config::{
    EvalStrategy, GridStrategy, ParamBounds, ParamDef, ParameterSpace, TunerConfig, TuningMode,
};

// Re-exports from metrics
pub use metrics::Metric;

// Re-exports from trial
pub use trial::TrialResult;

// Re-exports from history
pub use history::{ProgressCallback, SearchHistory};

// Re-exports from realistic
pub use realistic::RealisticModeConfig;

// Re-exports from autotuner
pub use autotuner::AutoTuner;
