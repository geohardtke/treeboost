//! AutoTuner configuration types
//!
//! This module defines the configuration structures for the hyperparameter tuner:
//! - `ParamBounds`: Bounds and scaling for a single parameter
//! - `ParamDef`: A parameter definition with name, bounds, and center
//! - `ParameterSpace`: Collection of parameters to tune
//! - `EvalStrategy`: How to evaluate candidates (holdout vs K-fold)
//! - `GridStrategy`: How to generate candidate points
//! - `TunerConfig`: Main tuner configuration

use std::collections::HashMap;

/// Bounds and scaling for a tunable parameter
#[derive(Debug, Clone, PartialEq)]
pub enum ParamBounds {
    /// Continuous parameter with min, max, and optional log scaling
    ///
    /// When `log_scale` is true, values are sampled uniformly in log space.
    /// This is useful for parameters like learning_rate where the difference
    /// between 0.01 and 0.1 is more significant than between 0.1 and 0.2.
    Continuous {
        min: f32,
        max: f32,
        log_scale: bool,
    },
    /// Discrete integer parameter with min, max, and step size
    ///
    /// Values are sampled at intervals of `step` between min and max.
    /// For example, `Discrete { min: 2, max: 12, step: 2 }` produces [2, 4, 6, 8, 10, 12].
    Discrete {
        min: usize,
        max: usize,
        step: usize,
    },
}

impl ParamBounds {
    /// Create continuous bounds without log scaling
    pub fn continuous(min: f32, max: f32) -> Self {
        Self::Continuous {
            min,
            max,
            log_scale: false,
        }
    }

    /// Create continuous bounds with log scaling
    pub fn log_continuous(min: f32, max: f32) -> Self {
        Self::Continuous {
            min,
            max,
            log_scale: true,
        }
    }

    /// Create discrete bounds with step size 1
    pub fn discrete(min: usize, max: usize) -> Self {
        Self::Discrete { min, max, step: 1 }
    }

    /// Create discrete bounds with custom step size
    pub fn discrete_step(min: usize, max: usize, step: usize) -> Self {
        Self::Discrete { min, max, step }
    }

    /// Clamp a value to be within bounds
    pub fn clamp(&self, value: f32) -> f32 {
        match self {
            Self::Continuous { min, max, .. } => value.clamp(*min, *max),
            Self::Discrete { min, max, step } => {
                let clamped = (value as usize).clamp(*min, *max);
                // Round to nearest step
                let steps = (clamped - min) / step;
                (min + steps * step) as f32
            }
        }
    }

    /// Check if a value is within bounds
    pub fn contains(&self, value: f32) -> bool {
        match self {
            Self::Continuous { min, max, .. } => value >= *min && value <= *max,
            Self::Discrete { min, max, .. } => {
                let v = value as usize;
                v >= *min && v <= *max
            }
        }
    }

    /// Get the minimum value
    pub fn min_value(&self) -> f32 {
        match self {
            Self::Continuous { min, .. } => *min,
            Self::Discrete { min, .. } => *min as f32,
        }
    }

    /// Get the maximum value
    pub fn max_value(&self) -> f32 {
        match self {
            Self::Continuous { max, .. } => *max,
            Self::Discrete { max, .. } => *max as f32,
        }
    }

    /// Check if this uses log scaling
    pub fn is_log_scale(&self) -> bool {
        matches!(self, Self::Continuous { log_scale: true, .. })
    }
}

/// Definition of a single tunable parameter
#[derive(Debug, Clone)]
pub struct ParamDef {
    /// Parameter name (must match GBDTConfig field name)
    pub name: String,
    /// Bounds and scaling
    pub bounds: ParamBounds,
    /// Current center point for grid generation
    pub center: f32,
}

impl ParamDef {
    /// Create a new parameter definition
    pub fn new(name: impl Into<String>, bounds: ParamBounds, center: f32) -> Self {
        let name = name.into();
        let center = bounds.clamp(center);
        Self {
            name,
            bounds,
            center,
        }
    }

    /// Update the center point (clamped to bounds)
    pub fn set_center(&mut self, center: f32) {
        self.center = self.bounds.clamp(center);
    }
}

/// Search space: collection of parameters to tune
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    params: Vec<ParamDef>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::default_regression()
    }
}

impl ParameterSpace {
    /// Create an empty parameter space
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Create default search space for regression
    ///
    /// Includes: max_depth, learning_rate, subsample, lambda, entropy_weight
    pub fn default_regression() -> Self {
        Self {
            params: vec![
                ParamDef::new(
                    "max_depth",
                    ParamBounds::discrete(2, 12),
                    6.0,
                ),
                ParamDef::new(
                    "learning_rate",
                    ParamBounds::log_continuous(0.01, 0.5),
                    0.1,
                ),
                ParamDef::new(
                    "subsample",
                    ParamBounds::continuous(0.5, 1.0),
                    0.8,
                ),
                ParamDef::new(
                    "lambda",
                    ParamBounds::continuous(0.0, 10.0),
                    1.0,
                ),
                ParamDef::new(
                    "entropy_weight",
                    ParamBounds::continuous(0.0, 0.5),
                    0.0,
                ),
            ],
        }
    }

    /// Create default search space for classification
    ///
    /// Same as regression but with different default centers optimized for classification
    pub fn default_classification() -> Self {
        Self {
            params: vec![
                ParamDef::new(
                    "max_depth",
                    ParamBounds::discrete(2, 10),
                    5.0,
                ),
                ParamDef::new(
                    "learning_rate",
                    ParamBounds::log_continuous(0.01, 0.3),
                    0.1,
                ),
                ParamDef::new(
                    "subsample",
                    ParamBounds::continuous(0.6, 1.0),
                    0.8,
                ),
                ParamDef::new(
                    "lambda",
                    ParamBounds::continuous(0.0, 5.0),
                    1.0,
                ),
                ParamDef::new(
                    "entropy_weight",
                    ParamBounds::continuous(0.0, 0.3),
                    0.0,
                ),
            ],
        }
    }

    /// Create a minimal search space (only learning_rate and max_depth)
    pub fn minimal() -> Self {
        Self {
            params: vec![
                ParamDef::new(
                    "max_depth",
                    ParamBounds::discrete(3, 10),
                    6.0,
                ),
                ParamDef::new(
                    "learning_rate",
                    ParamBounds::log_continuous(0.01, 0.3),
                    0.1,
                ),
            ],
        }
    }

    /// Add or update a parameter in the search space
    ///
    /// If a parameter with the same name exists, it will be replaced.
    pub fn with_param(mut self, name: &str, bounds: ParamBounds, center: f32) -> Self {
        // Remove existing param with same name
        self.params.retain(|p| p.name != name);
        self.params.push(ParamDef::new(name, bounds, center));
        self
    }

    /// Remove a parameter from tuning
    ///
    /// The parameter will use its default value from the base config.
    pub fn without_param(mut self, name: &str) -> Self {
        self.params.retain(|p| p.name != name);
        self
    }

    /// Get a parameter by name
    pub fn get(&self, name: &str) -> Option<&ParamDef> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Get a mutable parameter by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ParamDef> {
        self.params.iter_mut().find(|p| p.name == name)
    }

    /// Get all parameters
    pub fn params(&self) -> &[ParamDef] {
        &self.params
    }

    /// Get mutable access to all parameters
    pub fn params_mut(&mut self) -> &mut [ParamDef] {
        &mut self.params
    }

    /// Number of parameters in the search space
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Check if the search space is empty
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Get current centers as a HashMap
    pub fn centers(&self) -> HashMap<String, f32> {
        self.params
            .iter()
            .map(|p| (p.name.clone(), p.center))
            .collect()
    }

    /// Update centers from a HashMap
    pub fn set_centers(&mut self, centers: &HashMap<String, f32>) {
        for param in &mut self.params {
            if let Some(&center) = centers.get(&param.name) {
                param.set_center(center);
            }
        }
    }

    /// Validate that all parameter names are recognized GBDTConfig fields
    pub fn validate(&self) -> Result<(), String> {
        const VALID_PARAMS: &[&str] = &[
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample",
            "lambda",
            "entropy_weight",
            "min_samples_leaf",
            "min_hessian_leaf",
            "min_gain",
            "num_rounds",
            "goss_top_rate",
            "goss_other_rate",
        ];

        for param in &self.params {
            if !VALID_PARAMS.contains(&param.name.as_str()) {
                return Err(format!(
                    "Unknown parameter '{}'. Valid parameters: {:?}",
                    param.name, VALID_PARAMS
                ));
            }
        }
        Ok(())
    }
}

/// Strategy for evaluating candidate configurations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvalStrategy {
    /// Simple train/validation split
    ///
    /// Fast but may have high variance for small datasets.
    Holdout {
        /// Fraction of data to use for validation (e.g., 0.2 = 20%)
        validation_ratio: f32,
    },
    /// K-fold cross-validation
    ///
    /// More robust but K times slower than holdout.
    KFold {
        /// Number of folds (typically 3-10)
        k: usize,
    },
}

impl Default for EvalStrategy {
    fn default() -> Self {
        Self::Holdout {
            validation_ratio: 0.2,
        }
    }
}

impl EvalStrategy {
    /// Create holdout strategy with given validation ratio
    pub fn holdout(validation_ratio: f32) -> Self {
        Self::Holdout { validation_ratio }
    }

    /// Create K-fold cross-validation strategy
    pub fn kfold(k: usize) -> Self {
        Self::KFold { k }
    }

    /// Automatically select strategy based on dataset size
    ///
    /// - < 1,000 samples: 5-fold CV
    /// - 1,000 - 5,000 samples: 3-fold CV
    /// - > 5,000 samples: 20% holdout
    pub fn auto(num_samples: usize) -> Self {
        if num_samples < 1_000 {
            Self::KFold { k: 5 }
        } else if num_samples < 5_000 {
            Self::KFold { k: 3 }
        } else {
            Self::Holdout {
                validation_ratio: 0.2,
            }
        }
    }

    /// Validate the strategy
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Holdout { validation_ratio } => {
                if *validation_ratio <= 0.0 || *validation_ratio >= 1.0 {
                    return Err(format!(
                        "validation_ratio must be in (0, 1), got {}",
                        validation_ratio
                    ));
                }
            }
            Self::KFold { k } => {
                if *k < 2 {
                    return Err(format!("k must be >= 2, got {}", k));
                }
            }
        }
        Ok(())
    }
}

/// Strategy for generating candidate hyperparameter configurations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridStrategy {
    /// Full Cartesian product grid
    ///
    /// Generates `points_per_dim^n` candidates where n = number of parameters.
    /// For 5 parameters with 3 points each: 3^5 = 243 candidates per iteration.
    Cartesian {
        /// Number of points per dimension (typically 3)
        points_per_dim: usize,
    },
    /// Latin Hypercube Sampling
    ///
    /// Space-filling design that ensures good coverage with fewer samples.
    /// More efficient than Cartesian for high-dimensional spaces.
    LatinHypercube {
        /// Total number of samples to generate
        n_samples: usize,
    },
    /// Pure random sampling
    ///
    /// Simple but may miss important regions. Good for very large spaces.
    Random {
        /// Total number of samples to generate
        n_samples: usize,
    },
}

impl Default for GridStrategy {
    fn default() -> Self {
        Self::Cartesian { points_per_dim: 3 }
    }
}

impl GridStrategy {
    /// Create Cartesian grid with given points per dimension
    pub fn cartesian(points_per_dim: usize) -> Self {
        Self::Cartesian { points_per_dim }
    }

    /// Create Latin Hypercube sampling with given sample count
    pub fn lhs(n_samples: usize) -> Self {
        Self::LatinHypercube { n_samples }
    }

    /// Create random sampling with given sample count
    pub fn random(n_samples: usize) -> Self {
        Self::Random { n_samples }
    }

    /// Get the number of candidates this strategy will generate
    pub fn num_candidates(&self, num_params: usize) -> usize {
        match self {
            Self::Cartesian { points_per_dim } => points_per_dim.pow(num_params as u32),
            Self::LatinHypercube { n_samples } => *n_samples,
            Self::Random { n_samples } => *n_samples,
        }
    }

    /// Validate the strategy
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Cartesian { points_per_dim } => {
                if *points_per_dim < 2 {
                    return Err(format!(
                        "points_per_dim must be >= 2, got {}",
                        points_per_dim
                    ));
                }
            }
            Self::LatinHypercube { n_samples } | Self::Random { n_samples } => {
                if *n_samples < 1 {
                    return Err(format!("n_samples must be >= 1, got {}", n_samples));
                }
            }
        }
        Ok(())
    }
}

/// Main tuner configuration
#[derive(Debug, Clone)]
pub struct TunerConfig {
    /// Search space definition
    pub space: ParameterSpace,
    /// Number of zoom iterations
    pub n_iterations: usize,
    /// Initial spread factor (fraction of range to explore)
    ///
    /// 0.5 means explore +/- 50% around the center on the first iteration.
    pub initial_spread: f32,
    /// Zoom factor per iteration
    ///
    /// After each iteration, spread *= zoom_factor.
    /// 0.5 means halve the search radius each iteration.
    pub zoom_factor: f32,
    /// Grid generation strategy
    pub grid_strategy: GridStrategy,
    /// Evaluation strategy
    pub eval_strategy: EvalStrategy,
    /// Enable parallel trial evaluation (for CPU backends)
    ///
    /// GPU backends always run sequentially to avoid contention.
    pub parallel_trials: bool,
    /// Maximum number of parallel trials (0 = auto)
    pub n_parallel: usize,
    /// Early stopping rounds per trial
    pub early_stopping_rounds: usize,
    /// Number of boosting rounds per trial
    pub num_rounds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Enable verbose logging
    pub verbose: bool,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            space: ParameterSpace::default_regression(),
            n_iterations: 3,
            initial_spread: 0.5,
            zoom_factor: 0.5,
            grid_strategy: GridStrategy::Cartesian { points_per_dim: 3 },
            eval_strategy: EvalStrategy::Holdout {
                validation_ratio: 0.2,
            },
            parallel_trials: false, // Conservative default (GPU contention)
            n_parallel: 0,
            early_stopping_rounds: 10,
            num_rounds: 100,
            seed: 42,
            verbose: true,
        }
    }
}

impl TunerConfig {
    /// Create a new tuner config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a quick tuning config (2 iterations, small grid)
    pub fn quick() -> Self {
        Self {
            n_iterations: 2,
            num_rounds: 50,
            early_stopping_rounds: 5,
            ..Default::default()
        }
    }

    /// Create a thorough tuning config (5 iterations, larger grid)
    pub fn thorough() -> Self {
        Self {
            n_iterations: 5,
            num_rounds: 200,
            early_stopping_rounds: 20,
            ..Default::default()
        }
    }

    // Builder methods

    /// Set the parameter space
    pub fn with_space(mut self, space: ParameterSpace) -> Self {
        self.space = space;
        self
    }

    /// Set the number of zoom iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Set the initial spread factor
    pub fn with_initial_spread(mut self, spread: f32) -> Self {
        self.initial_spread = spread;
        self
    }

    /// Set the zoom factor
    pub fn with_zoom_factor(mut self, factor: f32) -> Self {
        self.zoom_factor = factor;
        self
    }

    /// Set the grid generation strategy
    pub fn with_grid_strategy(mut self, strategy: GridStrategy) -> Self {
        self.grid_strategy = strategy;
        self
    }

    /// Set the evaluation strategy
    pub fn with_eval_strategy(mut self, strategy: EvalStrategy) -> Self {
        self.eval_strategy = strategy;
        self
    }

    /// Enable or disable parallel trial evaluation
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel_trials = enabled;
        self
    }

    /// Set the maximum number of parallel trials
    pub fn with_n_parallel(mut self, n: usize) -> Self {
        self.n_parallel = n;
        self
    }

    /// Set early stopping rounds per trial
    pub fn with_early_stopping(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = rounds;
        self
    }

    /// Set number of boosting rounds per trial
    pub fn with_num_rounds(mut self, rounds: usize) -> Self {
        self.num_rounds = rounds;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enable or disable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.n_iterations == 0 {
            return Err("n_iterations must be > 0".into());
        }
        if self.initial_spread <= 0.0 || self.initial_spread > 1.0 {
            return Err(format!(
                "initial_spread must be in (0, 1], got {}",
                self.initial_spread
            ));
        }
        if self.zoom_factor <= 0.0 || self.zoom_factor >= 1.0 {
            return Err(format!(
                "zoom_factor must be in (0, 1), got {}",
                self.zoom_factor
            ));
        }
        if self.num_rounds == 0 {
            return Err("num_rounds must be > 0".into());
        }

        self.space.validate()?;
        self.grid_strategy.validate()?;
        self.eval_strategy.validate()?;

        Ok(())
    }

    /// Get the spread factor for a given iteration
    pub fn spread_for_iteration(&self, iteration: usize) -> f32 {
        self.initial_spread * self.zoom_factor.powi(iteration as i32)
    }

    /// Estimate total number of trials
    pub fn estimated_trials(&self) -> usize {
        let candidates_per_iter = self.grid_strategy.num_candidates(self.space.len());
        candidates_per_iter * self.n_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_bounds_continuous() {
        let bounds = ParamBounds::continuous(0.0, 1.0);
        assert_eq!(bounds.clamp(-0.5), 0.0);
        assert_eq!(bounds.clamp(0.5), 0.5);
        assert_eq!(bounds.clamp(1.5), 1.0);
        assert!(bounds.contains(0.5));
        assert!(!bounds.contains(-0.1));
        assert!(!bounds.is_log_scale());
    }

    #[test]
    fn test_param_bounds_log_continuous() {
        let bounds = ParamBounds::log_continuous(0.01, 1.0);
        assert!(bounds.is_log_scale());
        assert_eq!(bounds.min_value(), 0.01);
        assert_eq!(bounds.max_value(), 1.0);
    }

    #[test]
    fn test_param_bounds_discrete() {
        let bounds = ParamBounds::discrete(2, 10);
        assert_eq!(bounds.clamp(1.0), 2.0);
        assert_eq!(bounds.clamp(5.0), 5.0);
        assert_eq!(bounds.clamp(15.0), 10.0);
        assert!(!bounds.is_log_scale());
    }

    #[test]
    fn test_param_bounds_discrete_step() {
        let bounds = ParamBounds::discrete_step(2, 10, 2);
        // Value 5 should round down to 4 (nearest step from 2)
        assert_eq!(bounds.clamp(5.0), 4.0);
        assert_eq!(bounds.clamp(6.0), 6.0);
    }

    #[test]
    fn test_param_def() {
        let mut param = ParamDef::new("test", ParamBounds::continuous(0.0, 1.0), 0.5);
        assert_eq!(param.name, "test");
        assert_eq!(param.center, 0.5);

        param.set_center(2.0); // Out of bounds
        assert_eq!(param.center, 1.0); // Clamped
    }

    #[test]
    fn test_parameter_space_default() {
        let space = ParameterSpace::default_regression();
        assert_eq!(space.len(), 5);
        assert!(space.get("max_depth").is_some());
        assert!(space.get("learning_rate").is_some());
        assert!(space.get("subsample").is_some());
        assert!(space.get("lambda").is_some());
        assert!(space.get("entropy_weight").is_some());
    }

    #[test]
    fn test_parameter_space_with_param() {
        let space = ParameterSpace::minimal()
            .with_param("colsample", ParamBounds::continuous(0.5, 1.0), 0.8);

        assert_eq!(space.len(), 3);
        assert!(space.get("colsample").is_some());
    }

    #[test]
    fn test_parameter_space_without_param() {
        let space = ParameterSpace::default_regression().without_param("entropy_weight");

        assert_eq!(space.len(), 4);
        assert!(space.get("entropy_weight").is_none());
    }

    #[test]
    fn test_parameter_space_centers() {
        let mut space = ParameterSpace::minimal();
        let centers = space.centers();
        assert_eq!(centers.get("max_depth"), Some(&6.0));
        assert_eq!(centers.get("learning_rate"), Some(&0.1));

        let mut new_centers = HashMap::new();
        new_centers.insert("max_depth".into(), 8.0);
        space.set_centers(&new_centers);
        assert_eq!(space.get("max_depth").unwrap().center, 8.0);
    }

    #[test]
    fn test_parameter_space_validate() {
        let valid = ParameterSpace::default_regression();
        assert!(valid.validate().is_ok());

        let invalid = ParameterSpace::new()
            .with_param("invalid_param", ParamBounds::continuous(0.0, 1.0), 0.5);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_eval_strategy() {
        let holdout = EvalStrategy::holdout(0.2);
        assert!(holdout.validate().is_ok());

        let kfold = EvalStrategy::kfold(5);
        assert!(kfold.validate().is_ok());

        let invalid_holdout = EvalStrategy::holdout(1.5);
        assert!(invalid_holdout.validate().is_err());

        let invalid_kfold = EvalStrategy::kfold(1);
        assert!(invalid_kfold.validate().is_err());
    }

    #[test]
    fn test_eval_strategy_auto() {
        assert!(matches!(EvalStrategy::auto(500), EvalStrategy::KFold { k: 5 }));
        assert!(matches!(EvalStrategy::auto(2000), EvalStrategy::KFold { k: 3 }));
        assert!(matches!(EvalStrategy::auto(10000), EvalStrategy::Holdout { .. }));
    }

    #[test]
    fn test_grid_strategy() {
        let cart = GridStrategy::cartesian(3);
        assert_eq!(cart.num_candidates(5), 243); // 3^5

        let lhs = GridStrategy::lhs(50);
        assert_eq!(lhs.num_candidates(5), 50);

        let rand = GridStrategy::random(100);
        assert_eq!(rand.num_candidates(5), 100);
    }

    #[test]
    fn test_grid_strategy_validate() {
        assert!(GridStrategy::cartesian(3).validate().is_ok());
        assert!(GridStrategy::cartesian(1).validate().is_err());
        assert!(GridStrategy::lhs(0).validate().is_err());
    }

    #[test]
    fn test_tuner_config_default() {
        let config = TunerConfig::default();
        assert_eq!(config.n_iterations, 3);
        assert_eq!(config.initial_spread, 0.5);
        assert_eq!(config.zoom_factor, 0.5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tuner_config_quick() {
        let config = TunerConfig::quick();
        assert_eq!(config.n_iterations, 2);
        assert_eq!(config.num_rounds, 50);
    }

    #[test]
    fn test_tuner_config_thorough() {
        let config = TunerConfig::thorough();
        assert_eq!(config.n_iterations, 5);
        assert_eq!(config.num_rounds, 200);
    }

    #[test]
    fn test_tuner_config_builders() {
        let config = TunerConfig::new()
            .with_iterations(5)
            .with_num_rounds(200)
            .with_seed(123)
            .with_verbose(false);

        assert_eq!(config.n_iterations, 5);
        assert_eq!(config.num_rounds, 200);
        assert_eq!(config.seed, 123);
        assert!(!config.verbose);
    }

    #[test]
    fn test_tuner_config_spread_for_iteration() {
        let config = TunerConfig::default();
        assert_eq!(config.spread_for_iteration(0), 0.5);
        assert_eq!(config.spread_for_iteration(1), 0.25);
        assert_eq!(config.spread_for_iteration(2), 0.125);
    }

    #[test]
    fn test_tuner_config_estimated_trials() {
        let config = TunerConfig::default();
        // 3^5 = 243 candidates per iteration * 3 iterations = 729
        assert_eq!(config.estimated_trials(), 729);
    }

    #[test]
    fn test_tuner_config_validate() {
        assert!(TunerConfig::default().validate().is_ok());

        let invalid = TunerConfig::default().with_iterations(0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_initial_spread(0.0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_zoom_factor(1.0);
        assert!(invalid.validate().is_err());

        let invalid = TunerConfig::default().with_num_rounds(0);
        assert!(invalid.validate().is_err());
    }
}
