//! GBDT training configuration

use crate::loss::{LossFunction, MseLoss, PseudoHuberLoss};
use rkyv::{Archive, Deserialize, Serialize};

/// Loss function type for serialization
#[derive(Debug, Clone, Copy, PartialEq, Archive, Serialize, Deserialize)]
pub enum LossType {
    /// Mean Squared Error
    Mse,
    /// Pseudo-Huber Loss with given delta
    PseudoHuber { delta: f32 },
}

impl Default for LossType {
    fn default() -> Self {
        Self::Mse
    }
}

impl LossType {
    /// Create a boxed loss function
    pub fn create(&self) -> Box<dyn LossFunction> {
        match self {
            LossType::Mse => Box::new(MseLoss::new()),
            LossType::PseudoHuber { delta } => Box::new(PseudoHuberLoss::new(*delta)),
        }
    }
}

/// GBDT training configuration
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct GBDTConfig {
    // Ensemble parameters
    /// Number of boosting rounds (trees)
    pub num_rounds: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f32,

    // Tree parameters
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Maximum number of leaves per tree
    pub max_leaves: usize,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: usize,
    /// Minimum hessian sum required in a leaf
    pub min_hessian_leaf: f32,
    /// Minimum gain to make a split
    pub min_gain: f32,

    // Regularization
    /// L2 regularization (lambda)
    pub lambda: f32,
    /// Shannon Entropy regularization weight (beta)
    pub entropy_weight: f32,

    // Loss function
    /// Loss function type
    pub loss_type: LossType,

    // Subsampling
    /// Row subsampling ratio (0.0-1.0)
    pub subsample: f32,
    /// Column subsampling ratio (0.0-1.0)
    pub colsample: f32,

    // Binning
    /// Number of histogram bins
    pub num_bins: usize,

    // Conformal prediction
    /// Calibration set ratio for conformal prediction (0.0 to disable)
    pub calibration_ratio: f32,
    /// Conformal prediction quantile (e.g., 0.9 for 90% coverage)
    pub conformal_quantile: f32,
}

impl Default for GBDTConfig {
    fn default() -> Self {
        Self {
            // Ensemble
            num_rounds: 100,
            learning_rate: 0.1,

            // Tree
            max_depth: 6,
            max_leaves: 31,
            min_samples_leaf: 1,
            min_hessian_leaf: 1.0,
            min_gain: 0.0,

            // Regularization
            lambda: 1.0,
            entropy_weight: 0.0,

            // Loss
            loss_type: LossType::Mse,

            // Subsampling
            subsample: 1.0,
            colsample: 1.0,

            // Binning
            num_bins: 255,

            // Conformal
            calibration_ratio: 0.0,
            conformal_quantile: 0.9,
        }
    }
}

impl GBDTConfig {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of boosting rounds
    pub fn with_num_rounds(mut self, num_rounds: usize) -> Self {
        self.num_rounds = num_rounds;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum tree depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set maximum leaves per tree
    pub fn with_max_leaves(mut self, max_leaves: usize) -> Self {
        self.max_leaves = max_leaves;
        self
    }

    /// Set L2 regularization
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set Shannon Entropy regularization weight
    pub fn with_entropy_weight(mut self, weight: f32) -> Self {
        self.entropy_weight = weight;
        self
    }

    /// Set loss function to MSE
    pub fn with_mse_loss(mut self) -> Self {
        self.loss_type = LossType::Mse;
        self
    }

    /// Set loss function to Pseudo-Huber
    pub fn with_pseudo_huber_loss(mut self, delta: f32) -> Self {
        self.loss_type = LossType::PseudoHuber { delta };
        self
    }

    /// Set row subsampling ratio
    pub fn with_subsample(mut self, ratio: f32) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0);
        self.subsample = ratio;
        self
    }

    /// Set column subsampling ratio
    pub fn with_colsample(mut self, ratio: f32) -> Self {
        assert!(ratio > 0.0 && ratio <= 1.0);
        self.colsample = ratio;
        self
    }

    /// Enable conformal prediction
    pub fn with_conformal(mut self, calibration_ratio: f32, quantile: f32) -> Self {
        assert!(calibration_ratio >= 0.0 && calibration_ratio < 1.0);
        assert!(quantile > 0.0 && quantile < 1.0);
        self.calibration_ratio = calibration_ratio;
        self.conformal_quantile = quantile;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set minimum hessian per leaf
    pub fn with_min_hessian_leaf(mut self, min_hessian: f32) -> Self {
        self.min_hessian_leaf = min_hessian;
        self
    }

    /// Set minimum gain for splitting
    pub fn with_min_gain(mut self, min_gain: f32) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_rounds == 0 {
            return Err("num_rounds must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.max_depth == 0 {
            return Err("max_depth must be > 0".to_string());
        }
        if self.max_leaves == 0 {
            return Err("max_leaves must be > 0".to_string());
        }
        if self.lambda < 0.0 {
            return Err("lambda must be >= 0".to_string());
        }
        if self.subsample <= 0.0 || self.subsample > 1.0 {
            return Err("subsample must be in (0, 1]".to_string());
        }
        if self.colsample <= 0.0 || self.colsample > 1.0 {
            return Err("colsample must be in (0, 1]".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GBDTConfig::default();

        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.max_depth, 6);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = GBDTConfig::new()
            .with_num_rounds(50)
            .with_learning_rate(0.05)
            .with_max_depth(4)
            .with_pseudo_huber_loss(1.0)
            .with_entropy_weight(0.1)
            .with_conformal(0.1, 0.9);

        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.max_depth, 4);
        assert_eq!(config.loss_type, LossType::PseudoHuber { delta: 1.0 });
        assert_eq!(config.entropy_weight, 0.1);
        assert_eq!(config.calibration_ratio, 0.1);
    }

    #[test]
    fn test_config_validation() {
        let invalid = GBDTConfig::default().with_num_rounds(0);
        assert!(invalid.validate().is_err());

        let invalid = GBDTConfig {
            subsample: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
