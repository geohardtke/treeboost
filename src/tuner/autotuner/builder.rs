//! Builder methods for AutoTuner configuration
//!
//! This module provides a fluent API for configuring AutoTuner instances
//! with various options like search space, iterations, evaluation strategy, etc.

use crate::tuner::config::{EvalStrategy, ParameterSpace, TunerConfig};
use crate::tuner::traits::TunableModel;
use crate::tuner::trial::TrialResult;

use super::AutoTuner;

impl<M: TunableModel> AutoTuner<M> {
    /// Create a new AutoTuner with the given base configuration
    ///
    /// The base configuration provides default values for all parameters
    /// not being tuned.
    pub fn new(base_config: M::Config) -> Self {
        Self {
            config: TunerConfig::default(),
            base_config,
            history: crate::tuner::history::SearchHistory::new(),
            callback: None,
            next_trial_id: std::sync::atomic::AtomicUsize::new(0),
            custom_validation: None,
            raw_data: None,
            realistic_config: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the tuner configuration
    pub fn with_config(mut self, config: TunerConfig) -> Self {
        // Update history to use the configured optimization metric
        self.history =
            crate::tuner::history::SearchHistory::with_metric(config.optimization_metric);
        self.config = config;
        self
    }

    /// Set the parameter space
    pub fn with_space(mut self, space: ParameterSpace) -> Self {
        self.config.space = space;
        self
    }

    /// Set the number of iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.config.n_iterations = n;
        self
    }

    /// Set the evaluation strategy
    pub fn with_eval_strategy(mut self, strategy: EvalStrategy) -> Self {
        self.config.eval_strategy = strategy;
        self
    }

    /// Enable or disable parallel trial evaluation
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel_trials = enabled;
        self
    }

    /// Set the random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Set a progress callback
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&TrialResult, usize, usize) + Send + Sync + 'static,
    {
        self.callback = Some(Box::new(callback));
        self
    }

    /// Get the tuner configuration
    pub fn config(&self) -> &TunerConfig {
        &self.config
    }

    /// Get the search history
    pub fn history(&self) -> &crate::tuner::history::SearchHistory {
        &self.history
    }
}
