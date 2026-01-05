//! Automatic feature generation
//!
//! Provides tools for generating polynomial, ratio, and interaction features
//! from raw numeric data before binning.
//!
//! # Overview
//!
//! This module generates synthetic features that can improve model performance:
//!
//! 1. **Polynomial features**: x², √x, log(1+x)
//! 2. **Ratio features**: x_i / x_j for correlated pairs
//! 3. **Feature selection**: Filter by variance, correlation, target importance
//!
//! All feature generation happens BEFORE binning (required by the data pipeline).
//!
//! # Example
//!
//! ```ignore
//! use treeboost::features::{FeatureGenerator, PolynomialGenerator};
//!
//! let poly = PolynomialGenerator::new().with_square().with_sqrt();
//! let (new_features, new_names) = poly.generate(&features, &feature_names);
//! ```

mod polynomial;
mod ratio;
mod selector;

pub use polynomial::PolynomialGenerator;
pub use ratio::RatioGenerator;
pub use selector::{FeatureSelector, SelectionConfig};

/// Trait for feature generation strategies
pub trait FeatureGenerator: Send + Sync {
    /// Generate new features from input data
    ///
    /// # Arguments
    /// * `data` - Row-major feature matrix (num_rows × num_features)
    /// * `num_features` - Number of input features
    /// * `feature_names` - Names of input features
    ///
    /// # Returns
    /// Tuple of (new_features, new_names) where:
    /// - new_features: Row-major matrix of generated features
    /// - new_names: Names of generated features
    fn generate(
        &self,
        data: &[f32],
        num_features: usize,
        feature_names: &[String],
    ) -> (Vec<f32>, Vec<String>);

    /// Name of the generator
    fn name(&self) -> &'static str;
}

/// Configuration for feature generation pipeline
#[derive(Debug, Clone)]
pub struct FeatureGenerationConfig {
    /// Enable polynomial features
    pub polynomials: bool,
    /// Enable ratio features
    pub ratios: bool,
    /// Maximum polynomial degree (default: 2)
    pub max_degree: usize,
    /// Maximum ratio features per input feature
    pub max_ratios_per_feature: usize,
    /// Selection config for filtering generated features
    pub selection: SelectionConfig,
}

impl Default for FeatureGenerationConfig {
    fn default() -> Self {
        Self {
            polynomials: true,
            ratios: true,
            max_degree: 2,
            max_ratios_per_feature: 3,
            selection: SelectionConfig::default(),
        }
    }
}

impl FeatureGenerationConfig {
    /// Create a new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable polynomial features
    pub fn with_polynomials(mut self, enabled: bool) -> Self {
        self.polynomials = enabled;
        self
    }

    /// Enable or disable ratio features
    pub fn with_ratios(mut self, enabled: bool) -> Self {
        self.ratios = enabled;
        self
    }

    /// Set maximum polynomial degree
    pub fn with_max_degree(mut self, degree: usize) -> Self {
        self.max_degree = degree;
        self
    }

    /// Set maximum ratios per feature
    pub fn with_max_ratios_per_feature(mut self, max: usize) -> Self {
        self.max_ratios_per_feature = max;
        self
    }

    /// Set selection config
    pub fn with_selection(mut self, config: SelectionConfig) -> Self {
        self.selection = config;
        self
    }
}
