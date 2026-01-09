//! Incremental preprocessing support
//!
//! Provides traits and utilities for incrementally updating preprocessors
//! with new data batches, enabling online learning and model updates.
//!
//! # Supported Preprocessors
//!
//! | Preprocessor | Incremental Support | Notes |
//! |--------------|---------------------|-------|
//! | StandardScaler | Full | Welford's algorithm for online mean/variance |
//! | MinMaxScaler | Full | Min/max expand monotonically |
//! | FrequencyEncoder | Full | Counts accumulate |
//! | RobustScaler | Approximation | T-Digest merge (quantile approximation) |
//! | OneHotEncoder | Not Supported | Schema frozen after first fit |
//! | LabelEncoder | Not Supported | Schema frozen after first fit |
//!
//! # Example
//!
//! ```ignore
//! use treeboost::preprocessing::{StandardScaler, Scaler};
//! use treeboost::preprocessing::incremental::IncrementalScaler;
//!
//! let mut scaler = StandardScaler::new();
//!
//! // First batch
//! let batch1 = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows × 2 features
//! scaler.partial_fit(&batch1, 2)?;
//!
//! // Second batch
//! let batch2 = vec![5.0, 6.0, 7.0, 8.0]; // 2 more rows
//! scaler.partial_fit(&batch2, 2)?;
//!
//! // Now scaler has statistics from all 4 rows
//! assert_eq!(scaler.n_samples(), 4);
//! ```

use crate::{Result, TreeBoostError};

/// Trait for scalers that support incremental fitting
///
/// Scalers implementing this trait can update their statistics with new data
/// batches without requiring access to previously seen data.
pub trait IncrementalScaler {
    /// Update internal state with a new batch of data
    ///
    /// # Arguments
    /// * `data` - Row-major flat array (row0_feat0, row0_feat1, ..., row1_feat0, ...)
    /// * `num_features` - Number of features per row
    ///
    /// # Notes
    /// - First call initializes the scaler
    /// - Subsequent calls update statistics incrementally
    /// - `num_features` must be consistent across all calls
    fn partial_fit(&mut self, data: &[f32], num_features: usize) -> Result<()>;

    /// Get the total number of samples seen across all partial_fit calls
    fn n_samples(&self) -> u64;

    /// Merge state from another scaler of the same type
    ///
    /// Useful for distributed training where each worker has a partial view.
    fn merge(&mut self, other: &Self) -> Result<()>;
}

/// Trait for encoders that support incremental fitting
///
/// Encoders implementing this trait can update their category statistics
/// with new data batches.
pub trait IncrementalEncoder {
    /// Update internal state with a new batch of categories
    ///
    /// # Arguments
    /// * `categories` - Slice of category strings
    ///
    /// # Notes
    /// - New categories are added to the encoding dictionary
    /// - Existing category counts are updated
    fn partial_fit(&mut self, categories: &[&str]) -> Result<()>;

    /// Get the total number of samples seen across all partial_fit calls
    fn n_samples(&self) -> u64;
}

/// Error returned when incremental operations are not supported
pub fn not_supported_error(preprocessor_name: &str) -> TreeBoostError {
    TreeBoostError::Config(format!(
        "{} does not support incremental fitting. Schema is frozen after first fit. \
         For incremental learning, use FrequencyEncoder or TargetEncoder instead of {}.",
        preprocessor_name, preprocessor_name
    ))
}

/// Welford's online algorithm state for a single feature
///
/// Maintains running mean and variance with numerical stability.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WelfordState {
    /// Number of samples seen
    pub n: u64,
    /// Running mean
    pub mean: f64,
    /// Sum of squared differences from the mean (M2)
    /// Variance = m2 / n, Sample variance = m2 / (n-1)
    pub m2: f64,
}

impl WelfordState {
    /// Create a new empty state
    pub fn new() -> Self {
        Self::default()
    }

    /// Update state with a new value
    ///
    /// Uses Welford's online algorithm for numerical stability:
    /// - Avoids catastrophic cancellation with large means
    /// - Single-pass, O(1) memory
    #[inline]
    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the population variance (divide by n)
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Get the population standard deviation
    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Merge another WelfordState into this one
    ///
    /// Uses Chan's parallel algorithm for combining partial statistics.
    pub fn merge(&mut self, other: &WelfordState) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let combined_n = self.n + other.n;
        let delta = other.mean - self.mean;

        // Chan's parallel algorithm
        let combined_mean = self.mean + delta * (other.n as f64 / combined_n as f64);
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.n as f64 * other.n as f64 / combined_n as f64);

        self.n = combined_n;
        self.mean = combined_mean;
        self.m2 = combined_m2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let mut state = WelfordState::new();

        // Add values [1, 2, 3, 4, 5]
        for x in 1..=5 {
            state.update(x as f64);
        }

        assert_eq!(state.n, 5);
        assert!((state.mean - 3.0).abs() < 1e-10);

        // Variance of [1,2,3,4,5] = 2.0
        assert!((state.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_welford_merge() {
        // State A: [1, 2, 3]
        let mut state_a = WelfordState::new();
        for x in 1..=3 {
            state_a.update(x as f64);
        }

        // State B: [4, 5]
        let mut state_b = WelfordState::new();
        for x in 4..=5 {
            state_b.update(x as f64);
        }

        // Merge B into A
        state_a.merge(&state_b);

        // Should be equivalent to [1, 2, 3, 4, 5]
        assert_eq!(state_a.n, 5);
        assert!((state_a.mean - 3.0).abs() < 1e-10);
        assert!((state_a.variance() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_welford_numerical_stability() {
        // Test with large offset (classic problem case)
        let mut state = WelfordState::new();
        let offset = 1e8_f64;

        // Values: [1e8, 1e8+1, 1e8+2]
        for i in 0..3 {
            state.update(offset + i as f64);
        }

        assert_eq!(state.n, 3);
        assert!((state.mean - (offset + 1.0)).abs() < 1e-6);

        // Variance should be ~0.666... (2/3)
        let expected_var = 2.0 / 3.0;
        assert!(
            (state.variance() - expected_var).abs() < 1e-10,
            "Welford should handle large offsets: got {} expected {}",
            state.variance(),
            expected_var
        );
    }
}
