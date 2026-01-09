//! Shared utility functions
//!
//! This module provides convenient re-exports of common utilities for easier access.
//! Users can write `use treeboost::utils::apply_*` instead of navigating submodules.
//!
//! # Feature Engineering Utilities
//! - `apply_timeseries_features` - Generate lag, rolling, EWMA features for panel data
//! - `apply_crosssectional_features` - Generate rank/zscore for panel data (critical for Rank IC)
//! - `apply_polynomial_features` - Generate x², x³, √x, log(x+1) transformations
//! - `apply_ratio_features` - Generate ratio features (x_i / x_j)
//! - `apply_interaction_features` - Generate interaction features (x_i × x_j, etc.)
//! - `extract_selected_features` - Extract subset of features (used in LinearThenTree mode)
//!
//! # Preprocessing Utilities
//! - `apply_standard_scaler` - Z-score normalization (zero mean, unit variance)
//! - `apply_minmax_scaler` - Range normalization (scale to [min, max])
//! - `apply_robust_scaler` - Median/IQR scaling (robust to outliers)
//! - `apply_frequency_encoder` - Category → frequency mapping (best for trees)
//! - `apply_label_encoder` - String → integer encoding
//! - `apply_onehot_encoder` - Category → binary columns (best for linear models)
//! - `apply_simple_imputer` - Fill missing values (Mean/Median/Mode/Constant)

pub mod features;
pub mod preprocessing;

// Re-export feature application utilities for convenience
pub use features::{
    apply_crosssectional_features, apply_crosssectional_features_selective,
    apply_interaction_features, apply_polynomial_features, apply_ratio_features,
    apply_timeseries_features, extract_selected_features,
};

// Re-export preprocessing application utilities for convenience
pub use preprocessing::{
    apply_frequency_encoder, apply_label_encoder, apply_minmax_scaler, apply_onehot_encoder,
    apply_robust_scaler, apply_simple_imputer, apply_standard_scaler,
};

/// Check if two f32 values are approximately equal.
///
/// Uses a combined tolerance approach (similar to numpy's `isclose`):
/// `|a - b| <= atol + rtol * max(|a|, |b|)`
///
/// This ensures:
/// - Small absolute differences always pass (catches floating-point noise)
/// - Large values are compared relative to their magnitude
///
/// # Arguments
/// * `expected` - The expected value (e.g., parent sum)
/// * `actual` - The actual value (e.g., left + right)
/// * `rel_tol` - Relative tolerance (e.g., 1e-3 for 0.1% error)
///
/// Uses a fixed absolute tolerance of 1e-3 for backward compatibility.
#[inline]
pub fn approx_equal_relative(expected: f32, actual: f32, rel_tol: f32) -> bool {
    let diff = (expected - actual).abs();
    let abs_tol = 1e-3; // Fixed absolute tolerance
    let max_magnitude = expected.abs().max(actual.abs());
    diff <= abs_tol + rel_tol * max_magnitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_equal_relative_exact() {
        assert!(approx_equal_relative(1.0, 1.0, 1e-3));
        assert!(approx_equal_relative(0.0, 0.0, 1e-3));
    }

    #[test]
    fn test_approx_equal_relative_small_diff() {
        // 0.05% difference should pass with 1e-3 tolerance
        assert!(approx_equal_relative(1000.0, 1000.5, 1e-3));
        assert!(approx_equal_relative(1000.0, 999.5, 1e-3));
    }

    #[test]
    fn test_approx_equal_relative_large_diff() {
        // 1% difference should fail with 1e-3 tolerance for large values
        assert!(!approx_equal_relative(1000.0, 1010.0, 1e-3));
    }

    #[test]
    fn test_approx_equal_relative_small_values() {
        // Very small absolute diff should pass due to abs_tol
        assert!(approx_equal_relative(0.001, 0.00105, 1e-3));
    }

    #[test]
    fn test_approx_equal_relative_gradient_case() {
        // Real case from tree building: left + right ≈ parent
        // left=697.85895, right=-697.8577, parent=0.0012440681
        let left_plus_right = 697.85895_f32 + (-697.8577_f32);
        let parent = 0.0012440681_f32;
        // Diff is ~0.000006, which should pass with abs_tol=1e-3
        assert!(approx_equal_relative(parent, left_plus_right, 1e-3));
    }
}
