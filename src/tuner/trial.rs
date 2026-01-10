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
    /// Validation loss (MSE for regression, LogLoss for classification)
    /// Lower is better. This is NOT the optimization metric - see f1_score, roc_auc, rank_ic.
    pub val_loss: f32,
    /// Training loss (MSE for regression, LogLoss for classification)
    pub train_loss: f32,
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
    pub roc_auc: Option<f64>,
    /// Rank IC (Spearman correlation) for regression (None for classification)
    ///
    /// Measures correlation between prediction ranks and target ranks.
    /// Common in quantitative finance for measuring prediction quality.
    pub rank_ic: Option<f64>,
}

impl TrialResult {
    /// CSV column headers (excluding dynamic param columns)
    pub fn csv_headers() -> &'static [&'static str] {
        &[
            "trial_id",
            "iteration",
            "val_loss",   // Renamed from val_metric - actual loss value (MSE/LogLoss)
            "train_loss", // Renamed from train_metric
            "f1_score",
            "roc_auc",
            "rank_ic",
            "num_trees",
            "train_time_ms",
        ]
    }

    /// Convert trial result to CSV row values (excluding dynamic param columns)
    pub fn to_csv_row(&self) -> Vec<String> {
        vec![
            self.trial_id.to_string(),
            self.iteration.to_string(),
            format!("{:.6}", self.val_loss),
            format!("{:.6}", self.train_loss),
            self.f1_score
                .map(|f| format!("{:.4}", f))
                .unwrap_or_default(),
            self.roc_auc
                .map(|a| format!("{:.6}", a))
                .unwrap_or_default(),
            self.rank_ic
                .map(|r| format!("{:.6}", r))
                .unwrap_or_default(),
            self.num_trees.to_string(),
            self.train_time_ms.to_string(),
        ]
    }

    /// Get param value formatted for CSV
    pub fn param_to_csv(&self, name: &str) -> String {
        self.params
            .get(name)
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default()
    }
}
