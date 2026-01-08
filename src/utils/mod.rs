//! Shared utility functions

pub mod features;

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
