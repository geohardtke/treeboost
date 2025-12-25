//! Prediction types

use rkyv::{Archive, Deserialize, Serialize};

/// Prediction result with optional confidence interval
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
pub struct Prediction {
    /// Point prediction
    pub point: f32,
    /// Lower bound of prediction interval (if available)
    pub lower: Option<f32>,
    /// Upper bound of prediction interval (if available)
    pub upper: Option<f32>,
}

impl Prediction {
    /// Create a point prediction without intervals
    pub fn point(value: f32) -> Self {
        Self {
            point: value,
            lower: None,
            upper: None,
        }
    }

    /// Create a prediction with conformal intervals
    pub fn with_interval(point: f32, lower: f32, upper: f32) -> Self {
        Self {
            point,
            lower: Some(lower),
            upper: Some(upper),
        }
    }

    /// Check if this prediction has confidence intervals
    #[inline]
    pub fn has_interval(&self) -> bool {
        self.lower.is_some() && self.upper.is_some()
    }

    /// Get interval width (if available)
    pub fn interval_width(&self) -> Option<f32> {
        match (self.lower, self.upper) {
            (Some(l), Some(u)) => Some(u - l),
            _ => None,
        }
    }
}

impl From<f32> for Prediction {
    fn from(value: f32) -> Self {
        Self::point(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_prediction() {
        let pred = Prediction::point(42.0);

        assert_eq!(pred.point, 42.0);
        assert!(!pred.has_interval());
        assert!(pred.interval_width().is_none());
    }

    #[test]
    fn test_prediction_with_interval() {
        let pred = Prediction::with_interval(50.0, 45.0, 55.0);

        assert_eq!(pred.point, 50.0);
        assert!(pred.has_interval());
        assert_eq!(pred.lower, Some(45.0));
        assert_eq!(pred.upper, Some(55.0));
        assert_eq!(pred.interval_width(), Some(10.0));
    }
}
