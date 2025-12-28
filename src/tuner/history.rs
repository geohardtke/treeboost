//! Search history tracking for hyperparameter tuning

use super::trial::TrialResult;

/// Search history tracking all trials
#[derive(Debug, Clone, Default)]
pub struct SearchHistory {
    trials: Vec<TrialResult>,
    /// Index into trials Vec for O(1) lookup (not trial_id)
    best_trial_idx: Option<usize>,
}

impl SearchHistory {
    /// Create a new empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a trial result
    ///
    /// For classification (when f1_score is present), uses F1 as the primary
    /// comparison metric (higher is better). For regression, uses val_metric
    /// (lower is better).
    pub fn add(&mut self, result: TrialResult) {
        let new_idx = self.trials.len();

        let is_better = self
            .best_trial_idx
            .and_then(|idx| self.trials.get(idx))
            .map(|best| {
                // For classification: use F1 score (higher is better)
                // For regression: use val_metric (lower is better)
                match (result.f1_score, best.f1_score) {
                    // Both have valid F1 scores - compare them (handle NaN)
                    (Some(new_f1), Some(best_f1)) if !new_f1.is_nan() && !best_f1.is_nan() => {
                        new_f1 > best_f1
                    }
                    // New has valid F1, best is NaN - prefer non-NaN
                    (Some(new_f1), Some(_)) if !new_f1.is_nan() => true,
                    // Fall back to val_metric for regression or if F1 is unavailable/NaN
                    _ => result.val_metric < best.val_metric,
                }
            })
            .unwrap_or(true);

        self.trials.push(result);

        if is_better {
            self.best_trial_idx = Some(new_idx);
        }
    }

    /// Get the best trial so far (O(1) lookup)
    pub fn best(&self) -> Option<&TrialResult> {
        self.best_trial_idx.and_then(|idx| self.trials.get(idx))
    }

    /// Get all trials
    pub fn trials(&self) -> &[TrialResult] {
        &self.trials
    }

    /// Get trials for a specific iteration
    pub fn trials_for_iteration(&self, iteration: usize) -> Vec<&TrialResult> {
        self.trials
            .iter()
            .filter(|t| t.iteration == iteration)
            .collect()
    }

    /// Number of trials
    pub fn len(&self) -> usize {
        self.trials.len()
    }

    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }

    /// Export history to JSON string
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n  \"trials\": [\n");
        for (i, trial) in self.trials.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"trial_id\": {},\n", trial.trial_id));
            json.push_str(&format!("      \"iteration\": {},\n", trial.iteration));
            json.push_str(&format!("      \"val_metric\": {},\n", trial.val_metric));
            json.push_str(&format!("      \"train_metric\": {},\n", trial.train_metric));
            json.push_str(&format!("      \"num_trees\": {},\n", trial.num_trees));
            json.push_str(&format!("      \"train_time_ms\": {},\n", trial.train_time_ms));
            json.push_str("      \"params\": {\n");
            for (j, (k, v)) in trial.params.iter().enumerate() {
                let comma = if j < trial.params.len() - 1 { "," } else { "" };
                json.push_str(&format!("        \"{}\": {}{}\n", k, v, comma));
            }
            json.push_str("      }\n");
            let comma = if i < self.trials.len() - 1 { "," } else { "" };
            json.push_str(&format!("    }}{}\n", comma));
        }
        json.push_str("  ],\n");
        // Output the actual trial_id of the best trial (not the internal index)
        if let Some(best) = self.best() {
            json.push_str(&format!("  \"best_trial_id\": {}\n", best.trial_id));
        } else {
            json.push_str("  \"best_trial_id\": null\n");
        }
        json.push_str("}\n");
        json
    }
}

/// Progress callback type
///
/// Called after each trial with:
/// - `trial`: The completed trial result
/// - `current`: Current trial number (1-indexed)
/// - `total`: Total number of trials
pub type ProgressCallback = Box<dyn Fn(&TrialResult, usize, usize) + Send + Sync>;
