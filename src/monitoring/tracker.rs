//! CV-Holdout gap tracking for training-time distribution shift detection
//!
//! Monitors the gap between cross-validation and holdout metrics during
//! hyperparameter tuning to detect potential distribution shift or overfitting.

/// Trend direction for gap analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Gap is stable over iterations
    Stable,
    /// Gap is increasing (getting worse)
    Increasing,
    /// Gap is decreasing (getting better)
    Decreasing,
}

/// Record of a single CV vs holdout comparison
#[derive(Debug, Clone)]
pub struct GapRecord {
    /// Iteration number
    pub iteration: usize,
    /// Cross-validation metric value
    pub cv_metric: f32,
    /// Holdout metric value
    pub holdout_metric: f32,
    /// Absolute gap (cv - holdout or holdout - cv depending on metric direction)
    pub gap: f32,
    /// Relative gap (gap / cv_metric)
    pub relative_gap: f32,
}

/// Tracks the gap between CV and holdout metrics
///
/// Large or increasing gaps indicate potential issues:
/// - Distribution shift between folds and holdout
/// - Overfitting to validation data structure
/// - Data leakage in preprocessing
///
/// # Example
///
/// ```ignore
/// let mut tracker = CVHoldoutTracker::new(0.01); // 1% acceptable gap
///
/// for iteration in 0..100 {
///     let cv = model.cross_validate();
///     let holdout = model.evaluate(&holdout_data);
///     tracker.record(cv, holdout, iteration);
///
///     if tracker.is_shift_suspected() {
///         println!("Warning: {}", tracker.warning_message().unwrap());
///     }
/// }
/// ```
pub struct CVHoldoutTracker {
    /// Historical gap records
    history: Vec<GapRecord>,
    /// Acceptable gap threshold
    acceptable_gap: f32,
    /// Whether to use relative gap (gap / cv) or absolute gap
    use_relative: bool,
    /// Whether lower metric values are better
    lower_is_better: bool,
    /// Number of recent records to use for trend detection
    trend_window: usize,
}

impl CVHoldoutTracker {
    /// Create a new CV-Holdout tracker
    ///
    /// # Arguments
    /// * `acceptable_gap` - Maximum acceptable gap before warning
    pub fn new(acceptable_gap: f32) -> Self {
        Self {
            history: Vec::new(),
            acceptable_gap,
            use_relative: true,
            lower_is_better: true,
            trend_window: 5,
        }
    }

    /// Set whether to use relative gap
    pub fn with_relative(mut self, relative: bool) -> Self {
        self.use_relative = relative;
        self
    }

    /// Set whether lower metric values are better
    pub fn with_lower_is_better(mut self, lower_is_better: bool) -> Self {
        self.lower_is_better = lower_is_better;
        self
    }

    /// Set the trend detection window size
    pub fn with_trend_window(mut self, window: usize) -> Self {
        self.trend_window = window.max(2);
        self
    }

    /// Record a CV vs holdout comparison
    ///
    /// # Arguments
    /// * `cv_metric` - Cross-validation metric value
    /// * `holdout_metric` - Holdout metric value
    /// * `iteration` - Iteration number (for tracking)
    pub fn record(&mut self, cv_metric: f32, holdout_metric: f32, iteration: usize) {
        // Compute gap based on metric direction
        // If lower is better: gap = cv - holdout (negative = CV is too optimistic)
        // If higher is better: gap = holdout - cv (negative = CV is too optimistic)
        let gap = if self.lower_is_better {
            holdout_metric - cv_metric
        } else {
            cv_metric - holdout_metric
        };

        // Relative gap
        let relative_gap = if cv_metric.abs() > 1e-10 {
            gap.abs() / cv_metric.abs()
        } else {
            gap.abs()
        };

        self.history.push(GapRecord {
            iteration,
            cv_metric,
            holdout_metric,
            gap,
            relative_gap,
        });
    }

    /// Check if distribution shift is suspected
    ///
    /// Returns true if:
    /// - Current gap exceeds threshold
    /// - Gap trend is increasing
    pub fn is_shift_suspected(&self) -> bool {
        if self.history.is_empty() {
            return false;
        }

        let current = self.history.last().unwrap();
        let threshold = if self.use_relative {
            current.relative_gap > self.acceptable_gap
        } else {
            current.gap.abs() > self.acceptable_gap
        };

        threshold || self.gap_trend() == Trend::Increasing
    }

    /// Get the trend of the gap over recent iterations
    pub fn gap_trend(&self) -> Trend {
        if self.history.len() < self.trend_window {
            return Trend::Stable;
        }

        let recent = &self.history[self.history.len() - self.trend_window..];
        let gaps: Vec<f32> = if self.use_relative {
            recent.iter().map(|r| r.relative_gap).collect()
        } else {
            recent.iter().map(|r| r.gap.abs()).collect()
        };

        // Simple linear trend detection
        let n = gaps.len() as f32;
        let sum_x: f32 = (0..gaps.len()).map(|i| i as f32).sum();
        let sum_y: f32 = gaps.iter().sum();
        let sum_xy: f32 = gaps.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_xx: f32 = (0..gaps.len()).map(|i| (i * i) as f32).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-10);

        // Thresholds for trend detection
        if slope > 0.001 {
            Trend::Increasing
        } else if slope < -0.001 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    /// Generate a warning message if shift is suspected
    pub fn warning_message(&self) -> Option<String> {
        if !self.is_shift_suspected() {
            return None;
        }

        let current = self.history.last()?;
        let gap_str = if self.use_relative {
            format!("{:.2}%", current.relative_gap * 100.0)
        } else {
            format!("{:.4}", current.gap.abs())
        };

        let trend = self.gap_trend();
        let trend_str = match trend {
            Trend::Increasing => " and increasing",
            Trend::Decreasing => " but decreasing",
            Trend::Stable => "",
        };

        Some(format!(
            "CV-Holdout gap of {}{} (threshold: {:.2}%) at iteration {}",
            gap_str,
            trend_str,
            self.acceptable_gap * 100.0,
            current.iteration
        ))
    }

    /// Get the current gap value
    pub fn current_gap(&self) -> Option<f32> {
        self.history.last().map(|r| {
            if self.use_relative {
                r.relative_gap
            } else {
                r.gap.abs()
            }
        })
    }

    /// Get the mean gap across all history
    pub fn mean_gap(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }

        let sum: f32 = if self.use_relative {
            self.history.iter().map(|r| r.relative_gap).sum()
        } else {
            self.history.iter().map(|r| r.gap.abs()).sum()
        };

        sum / self.history.len() as f32
    }

    /// Get the maximum gap observed
    pub fn max_gap(&self) -> f32 {
        if self.use_relative {
            self.history
                .iter()
                .map(|r| r.relative_gap)
                .fold(0.0, f32::max)
        } else {
            self.history.iter().map(|r| r.gap.abs()).fold(0.0, f32::max)
        }
    }

    /// Get the full history
    pub fn history(&self) -> &[GapRecord] {
        &self.history
    }

    /// Get number of records
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Clear the history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Get summary statistics
    pub fn summary(&self) -> TrackerSummary {
        TrackerSummary {
            n_records: self.history.len(),
            current_gap: self.current_gap().unwrap_or(0.0),
            mean_gap: self.mean_gap(),
            max_gap: self.max_gap(),
            trend: self.gap_trend(),
            shift_suspected: self.is_shift_suspected(),
        }
    }
}

/// Summary statistics from the tracker
#[derive(Debug, Clone)]
pub struct TrackerSummary {
    /// Number of records
    pub n_records: usize,
    /// Current gap value
    pub current_gap: f32,
    /// Mean gap across history
    pub mean_gap: f32,
    /// Maximum gap observed
    pub max_gap: f32,
    /// Trend direction
    pub trend: Trend,
    /// Whether shift is suspected
    pub shift_suspected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_basic() {
        let mut tracker = CVHoldoutTracker::new(0.01);
        tracker.record(0.5, 0.51, 0);
        tracker.record(0.5, 0.52, 1);

        assert_eq!(tracker.len(), 2);
        assert!(!tracker.is_empty());
    }

    #[test]
    fn test_tracker_small_gap() {
        let mut tracker = CVHoldoutTracker::new(0.1); // 10% threshold
        tracker.record(0.5, 0.51, 0); // 2% gap

        assert!(!tracker.is_shift_suspected());
        assert!(tracker.warning_message().is_none());
    }

    #[test]
    fn test_tracker_large_gap() {
        let mut tracker = CVHoldoutTracker::new(0.01); // 1% threshold
        tracker.record(0.5, 0.6, 0); // 20% gap

        assert!(tracker.is_shift_suspected());
        assert!(tracker.warning_message().is_some());
    }

    #[test]
    fn test_tracker_increasing_trend() {
        let mut tracker = CVHoldoutTracker::new(0.5).with_trend_window(3);

        // Increasing gaps
        tracker.record(0.5, 0.51, 0);
        tracker.record(0.5, 0.55, 1);
        tracker.record(0.5, 0.60, 2);
        tracker.record(0.5, 0.70, 3);
        tracker.record(0.5, 0.85, 4);

        assert_eq!(tracker.gap_trend(), Trend::Increasing);
    }

    #[test]
    fn test_tracker_decreasing_trend() {
        let mut tracker = CVHoldoutTracker::new(0.5).with_trend_window(3);

        // Decreasing gaps
        tracker.record(0.5, 0.7, 0);
        tracker.record(0.5, 0.6, 1);
        tracker.record(0.5, 0.55, 2);
        tracker.record(0.5, 0.52, 3);
        tracker.record(0.5, 0.51, 4);

        assert_eq!(tracker.gap_trend(), Trend::Decreasing);
    }

    #[test]
    fn test_tracker_stable_trend() {
        let mut tracker = CVHoldoutTracker::new(0.5).with_trend_window(3);

        // Stable gaps
        tracker.record(0.5, 0.51, 0);
        tracker.record(0.5, 0.52, 1);
        tracker.record(0.5, 0.51, 2);
        tracker.record(0.5, 0.52, 3);
        tracker.record(0.5, 0.51, 4);

        assert_eq!(tracker.gap_trend(), Trend::Stable);
    }

    #[test]
    fn test_tracker_summary() {
        let mut tracker = CVHoldoutTracker::new(0.1);
        tracker.record(0.5, 0.55, 0);
        tracker.record(0.5, 0.52, 1);

        let summary = tracker.summary();
        assert_eq!(summary.n_records, 2);
        assert!(summary.current_gap > 0.0);
    }

    #[test]
    fn test_tracker_absolute_gap() {
        let mut tracker = CVHoldoutTracker::new(0.05).with_relative(false);
        tracker.record(0.5, 0.52, 0); // absolute gap of 0.02

        assert!(!tracker.is_shift_suspected()); // 0.02 < 0.05
    }

    #[test]
    fn test_tracker_higher_is_better() {
        let mut tracker = CVHoldoutTracker::new(0.1).with_lower_is_better(false);
        // For accuracy-like metrics where higher is better
        // CV = 0.9, holdout = 0.85 means gap = 0.9 - 0.85 = 0.05
        tracker.record(0.9, 0.85, 0);

        let gap = tracker.history.last().unwrap().gap;
        assert!(gap > 0.0); // CV is more optimistic
    }

    #[test]
    fn test_tracker_clear() {
        let mut tracker = CVHoldoutTracker::new(0.1);
        tracker.record(0.5, 0.51, 0);
        tracker.record(0.5, 0.52, 1);

        tracker.clear();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }
}
