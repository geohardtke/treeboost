//! Split Conformal Prediction
//!
//! Provides distribution-free prediction intervals with guaranteed coverage.

use crate::inference::Prediction;
use rkyv::{Archive, Deserialize, Serialize};

/// Split Conformal Predictor
///
/// Computes prediction intervals from calibration residuals.
/// Provides finite-sample coverage guarantee: P(Y ∈ [ŷ - q, ŷ + q]) ≥ 1 - α
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct ConformalPredictor {
    /// Quantile value for symmetric intervals
    quantile: f32,
    /// Coverage level (e.g., 0.9 for 90%)
    coverage: f32,
}

impl ConformalPredictor {
    /// Create a new conformal predictor from calibration residuals
    ///
    /// # Arguments
    /// * `residuals` - Absolute residuals |y - ŷ| from calibration set
    /// * `coverage` - Desired coverage level (e.g., 0.9 for 90%)
    pub fn from_residuals(residuals: &[f32], coverage: f32) -> Self {
        assert!(
            coverage > 0.0 && coverage < 1.0,
            "coverage must be in (0, 1)"
        );

        let quantile = Self::compute_quantile(residuals, coverage);

        Self { quantile, coverage }
    }

    /// Create from precomputed quantile
    pub fn from_quantile(quantile: f32, coverage: f32) -> Self {
        Self { quantile, coverage }
    }

    /// Compute the (1-α)(1 + 1/n) quantile of residuals
    ///
    /// This specific quantile provides the finite-sample coverage guarantee.
    fn compute_quantile(residuals: &[f32], coverage: f32) -> f32 {
        if residuals.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<f32> = residuals.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute adjusted quantile level for finite-sample guarantee
        let n = sorted.len() as f32;
        let adjusted_coverage = coverage * (1.0 + 1.0 / n);
        let adjusted_coverage = adjusted_coverage.min(1.0);

        let idx = ((sorted.len() as f32) * adjusted_coverage).ceil() as usize;
        let idx = idx.min(sorted.len() - 1);

        sorted[idx]
    }

    /// Create prediction intervals for point predictions
    pub fn predict(&self, point: f32) -> Prediction {
        Prediction::with_interval(point, point - self.quantile, point + self.quantile)
    }

    /// Create prediction intervals for multiple predictions
    pub fn predict_batch(&self, points: &[f32]) -> Vec<Prediction> {
        points.iter().map(|&p| self.predict(p)).collect()
    }

    /// Get the quantile value
    pub fn quantile(&self) -> f32 {
        self.quantile
    }

    /// Get the coverage level
    pub fn coverage(&self) -> f32 {
        self.coverage
    }

    /// Compute empirical coverage on test data
    ///
    /// Returns the fraction of test points where true value falls within interval.
    pub fn empirical_coverage(&self, true_values: &[f32], predictions: &[f32]) -> f32 {
        if true_values.is_empty() {
            return 0.0;
        }

        let covered: usize = true_values
            .iter()
            .zip(predictions.iter())
            .filter(|(&y, &pred)| {
                let lower = pred - self.quantile;
                let upper = pred + self.quantile;
                y >= lower && y <= upper
            })
            .count();

        covered as f32 / true_values.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_predictor() {
        // Calibration residuals
        let residuals: Vec<f32> = (1..=100).map(|i| i as f32).collect();

        let predictor = ConformalPredictor::from_residuals(&residuals, 0.9);

        // Quantile should be approximately the 90th percentile
        assert!(predictor.quantile() >= 90.0);
        assert!(predictor.quantile() <= 100.0);
    }

    #[test]
    fn test_prediction_intervals() {
        let predictor = ConformalPredictor::from_quantile(5.0, 0.9);

        let pred = predictor.predict(100.0);

        assert_eq!(pred.point, 100.0);
        assert_eq!(pred.lower, Some(95.0));
        assert_eq!(pred.upper, Some(105.0));
    }

    #[test]
    fn test_empirical_coverage() {
        let predictor = ConformalPredictor::from_quantile(10.0, 0.9);

        // All within interval
        let true_vals = vec![100.0, 101.0, 99.0];
        let preds = vec![100.0, 100.0, 100.0];
        let cov = predictor.empirical_coverage(&true_vals, &preds);
        assert_eq!(cov, 1.0);

        // One outside
        let true_vals = vec![100.0, 115.0, 99.0];
        let cov = predictor.empirical_coverage(&true_vals, &preds);
        assert!((cov - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_prediction() {
        let predictor = ConformalPredictor::from_quantile(5.0, 0.9);

        let preds = predictor.predict_batch(&[10.0, 20.0, 30.0]);

        assert_eq!(preds.len(), 3);
        assert_eq!(preds[0].point, 10.0);
        assert_eq!(preds[1].point, 20.0);
        assert_eq!(preds[2].point, 30.0);

        for pred in &preds {
            assert!(pred.has_interval());
            assert_eq!(pred.interval_width(), Some(10.0));
        }
    }
}
