//! Progress tracking for AutoML training
//!
//! Provides callback-based progress tracking for long-running AutoML operations.

use std::time::Duration;

/// Training phase identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Profiling DataFrame to understand column types
    Profiling,
    /// Planning preprocessing strategy
    Preprocessing,
    /// Planning feature engineering
    FeatureEngineering,
    /// Preparing dataset (binning, encoding)
    DatasetPreparation,
    /// Analyzing dataset for mode selection
    Analysis,
    /// Tuning hyperparameters
    Tuning,
    /// Training final model
    Training,
    /// Build complete
    Complete,
}

impl TrainingPhase {
    /// Get human-readable phase name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Profiling => "Profiling",
            Self::Preprocessing => "Preprocessing",
            Self::FeatureEngineering => "Feature Engineering",
            Self::DatasetPreparation => "Dataset Preparation",
            Self::Analysis => "Analysis",
            Self::Tuning => "Tuning",
            Self::Training => "Training",
            Self::Complete => "Complete",
        }
    }

    /// Get estimated progress percentage (0-100) for this phase
    /// Assumes roughly equal time per phase (will be more accurate with actual timing)
    pub fn progress_pct(&self) -> u8 {
        match self {
            Self::Profiling => 10,
            Self::Preprocessing => 20,
            Self::FeatureEngineering => 30,
            Self::DatasetPreparation => 40,
            Self::Analysis => 50,
            Self::Tuning => 70,
            Self::Training => 90,
            Self::Complete => 100,
        }
    }
}

/// Progress update information
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Current phase
    pub phase: TrainingPhase,

    /// Estimated progress (0-100)
    pub progress_pct: u8,

    /// Time elapsed since start
    pub elapsed: Duration,

    /// Optional message with phase-specific details
    pub message: Option<String>,
}

/// Callback trait for progress updates
///
/// Implement this trait to receive progress updates during AutoML training.
/// The callback is invoked at the start of each training phase.
///
/// # Example
///
/// ```ignore
/// struct ConsoleProgress;
///
/// impl ProgressCallback for ConsoleProgress {
///     fn on_progress(&self, update: &ProgressUpdate) {
///         println!(
///             "[{:3}%] {} - {:?}",
///             update.progress_pct,
///             update.phase.name(),
///             update.elapsed
///         );
///     }
/// }
///
/// let callback = Box::new(ConsoleProgress);
/// let builder = AutoBuilder::new().with_progress_callback(callback);
/// ```
pub trait ProgressCallback: Send + Sync {
    /// Called when a new training phase starts
    fn on_progress(&self, update: &ProgressUpdate);
}

/// Simple console progress callback (prints to stdout)
pub struct ConsoleProgress {
    /// Whether to show detailed messages
    pub detailed: bool,
}

impl ConsoleProgress {
    /// Create a new console progress tracker
    pub fn new() -> Self {
        Self { detailed: false }
    }

    /// Create a detailed console progress tracker (shows messages)
    pub fn detailed() -> Self {
        Self { detailed: true }
    }
}

impl Default for ConsoleProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressCallback for ConsoleProgress {
    fn on_progress(&self, update: &ProgressUpdate) {
        let bar_width = 30;
        let filled = (update.progress_pct as usize * bar_width) / 100;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

        if self.detailed {
            if let Some(ref msg) = update.message {
                println!(
                    "[{:3}%] {} │{}│ {:?} - {}",
                    update.progress_pct,
                    update.phase.name(),
                    bar,
                    update.elapsed,
                    msg
                );
            } else {
                println!(
                    "[{:3}%] {} │{}│ {:?}",
                    update.progress_pct,
                    update.phase.name(),
                    bar,
                    update.elapsed
                );
            }
        } else {
            println!(
                "[{:3}%] {} │{}│",
                update.progress_pct, update.phase.name(), bar
            );
        }
    }
}

/// Quiet callback that does nothing (for testing or when progress not needed)
pub struct QuietProgress;

impl ProgressCallback for QuietProgress {
    fn on_progress(&self, _update: &ProgressUpdate) {
        // Do nothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_progress() {
        assert_eq!(TrainingPhase::Profiling.progress_pct(), 10);
        assert_eq!(TrainingPhase::Complete.progress_pct(), 100);
    }

    #[test]
    fn test_phase_name() {
        assert_eq!(TrainingPhase::Tuning.name(), "Tuning");
        assert_eq!(TrainingPhase::Training.name(), "Training");
    }

    #[test]
    fn test_console_progress() {
        let progress = ConsoleProgress::new();
        let update = ProgressUpdate {
            phase: TrainingPhase::Profiling,
            progress_pct: 10,
            elapsed: Duration::from_secs(5),
            message: Some("Analyzing 50 columns".to_string()),
        };

        // Should not panic
        progress.on_progress(&update);
    }

    #[test]
    fn test_quiet_progress() {
        let progress = QuietProgress;
        let update = ProgressUpdate {
            phase: TrainingPhase::Training,
            progress_pct: 90,
            elapsed: Duration::from_secs(120),
            message: None,
        };

        // Should do nothing
        progress.on_progress(&update);
    }
}
