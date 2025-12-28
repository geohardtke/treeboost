//! Trial result types for hyperparameter tuning

use std::collections::HashMap;

/// Result of a single trial (candidate evaluation)
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Unique trial identifier
    pub trial_id: usize,
    /// Iteration (zoom level) when this trial was run
    pub iteration: usize,
    /// Hyperparameter values used
    pub params: HashMap<String, f32>,
    /// Validation metric (lower is better for MSE/LogLoss)
    pub val_metric: f32,
    /// Training metric
    pub train_metric: f32,
    /// Number of trees actually trained (may be < num_rounds if early stopped)
    pub num_trees: usize,
    /// Training time in milliseconds
    pub train_time_ms: u64,
    /// F1 score for classification (None for regression)
    ///
    /// F1 is the harmonic mean of precision and recall.
    /// A low F1 score indicates an unbalanced model (e.g., predicting
    /// all negatives gives F1 = 0).
    pub f1_score: Option<f32>,
    /// ROC-AUC score for binary classification (None for regression/multi-class)
    ///
    /// Area Under the ROC Curve measures ranking quality.
    /// Used by Kaggle for many binary classification competitions.
    pub roc_auc: Option<f64>,
}

impl TrialResult {
    /// CSV column headers (excluding dynamic param columns)
    pub fn csv_headers() -> &'static [&'static str] {
        &[
            "trial_id",
            "iteration",
            "val_metric",
            "train_metric",
            "f1_score",
            "roc_auc",
            "num_trees",
            "train_time_ms",
        ]
    }

    /// Convert trial result to CSV row values (excluding dynamic param columns)
    pub fn to_csv_row(&self) -> Vec<String> {
        vec![
            self.trial_id.to_string(),
            self.iteration.to_string(),
            format!("{:.6}", self.val_metric),
            format!("{:.6}", self.train_metric),
            self.f1_score.map(|f| format!("{:.4}", f)).unwrap_or_default(),
            self.roc_auc.map(|a| format!("{:.6}", a)).unwrap_or_default(),
            self.num_trees.to_string(),
            self.train_time_ms.to_string(),
        ]
    }

    /// Get param value formatted for CSV
    pub fn param_to_csv(&self, name: &str) -> String {
        self.params.get(name).map(|v| format!("{:.6}", v)).unwrap_or_default()
    }
}
