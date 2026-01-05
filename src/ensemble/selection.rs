//! Hill climbing ensemble selection
//!
//! Greedily selects models that improve cross-validation score.
//! Uses forward selection starting from empty ensemble.

use super::multi_seed::TrainedMember;
use crate::tuner::Metric;

/// Configuration for hill climbing selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Maximum models to select (0 = no limit)
    pub max_models: usize,
    /// Minimum improvement required to add a model
    pub min_improvement: f32,
    /// Maximum iterations without improvement before stopping
    pub patience: usize,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            max_models: 0, // No limit
            min_improvement: 1e-6,
            patience: 5,
        }
    }
}

impl SelectionConfig {
    /// Create a new selection config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum models to select
    pub fn with_max_models(mut self, max: usize) -> Self {
        self.max_models = max;
        self
    }

    /// Set minimum improvement threshold
    pub fn with_min_improvement(mut self, min: f32) -> Self {
        self.min_improvement = min;
        self
    }

    /// Set patience (max iterations without improvement)
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }
}

/// Hill climbing selector for greedy ensemble selection
pub struct HillClimbingSelector {
    config: SelectionConfig,
    metric: Metric,
}

impl HillClimbingSelector {
    /// Create a new hill climbing selector
    pub fn new(config: SelectionConfig, metric: Metric) -> Self {
        Self { config, metric }
    }

    /// Select optimal subset of models using forward selection
    ///
    /// # Algorithm
    /// 1. Start with empty ensemble
    /// 2. For each candidate not in ensemble:
    ///    - Compute CV metric of (ensemble + candidate) using simple average
    /// 3. Add candidate with best improvement
    /// 4. Repeat until no improvement or max_models reached
    ///
    /// # Returns
    /// Indices of selected models in the order they were added
    pub fn select(&self, candidates: &[TrainedMember], targets: &[f32]) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut selected: Vec<usize> = Vec::new();
        let mut iterations_without_improvement = 0;

        // Initial metric is worst possible
        let mut current_metric = if self.metric.lower_is_better() {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };

        loop {
            let mut best_candidate: Option<(usize, f32)> = None;
            let mut best_improvement = 0.0f32;

            // Try adding each unselected candidate
            for (idx, _candidate) in candidates.iter().enumerate() {
                if selected.contains(&idx) {
                    continue;
                }

                // Compute blended predictions with this candidate
                let blended = self.blend_oof_predictions(candidates, &selected, idx);
                let new_metric = self.metric.compute(&blended, targets);

                // Check if this is an improvement
                let improvement = if self.metric.lower_is_better() {
                    current_metric - new_metric
                } else {
                    new_metric - current_metric
                };

                if improvement > best_improvement {
                    best_improvement = improvement;
                    best_candidate = Some((idx, new_metric));
                }
            }

            // Check if we found an improvement
            match best_candidate {
                Some((idx, new_metric)) if best_improvement >= self.config.min_improvement => {
                    selected.push(idx);
                    current_metric = new_metric;
                    iterations_without_improvement = 0;

                    // Check max models limit
                    if self.config.max_models > 0 && selected.len() >= self.config.max_models {
                        break;
                    }

                    // Check if all models selected
                    if selected.len() == candidates.len() {
                        break;
                    }
                }
                _ => {
                    // No improvement found
                    iterations_without_improvement += 1;
                    if iterations_without_improvement >= self.config.patience {
                        break;
                    }
                }
            }
        }

        selected
    }

    /// Blend OOF predictions using simple average
    fn blend_oof_predictions(
        &self,
        candidates: &[TrainedMember],
        selected: &[usize],
        new_idx: usize,
    ) -> Vec<f32> {
        // Get all indices to blend (selected + new)
        let indices: Vec<usize> = selected.iter().copied().chain(std::iter::once(new_idx)).collect();

        if indices.is_empty() {
            return Vec::new();
        }

        let n_samples = candidates[indices[0]].oof_preds.len();
        let n_members = indices.len() as f32;

        (0..n_samples)
            .map(|i| {
                let sum: f32 = indices.iter().map(|&idx| candidates[idx].oof_preds[i]).sum();
                sum / n_members
            })
            .collect()
    }

    /// Get selection statistics
    pub fn selection_stats(
        &self,
        candidates: &[TrainedMember],
        selected: &[usize],
        targets: &[f32],
    ) -> SelectionStats {
        // Individual model metrics
        let individual_metrics: Vec<f32> = candidates
            .iter()
            .map(|c| self.metric.compute(&c.oof_preds, targets))
            .collect();

        // Best individual
        let best_individual = if self.metric.lower_is_better() {
            individual_metrics
                .iter()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f32::INFINITY)
        } else {
            individual_metrics
                .iter()
                .cloned()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f32::NEG_INFINITY)
        };

        // Ensemble metric
        let ensemble_preds = if selected.is_empty() {
            vec![0.0; targets.len()]
        } else {
            let n_samples = candidates[selected[0]].oof_preds.len();
            let n_members = selected.len() as f32;
            (0..n_samples)
                .map(|i| {
                    let sum: f32 = selected.iter().map(|&idx| candidates[idx].oof_preds[i]).sum();
                    sum / n_members
                })
                .collect()
        };
        let ensemble_metric = self.metric.compute(&ensemble_preds, targets);

        SelectionStats {
            n_candidates: candidates.len(),
            n_selected: selected.len(),
            best_individual_metric: best_individual,
            ensemble_metric,
            improvement: if self.metric.lower_is_better() {
                best_individual - ensemble_metric
            } else {
                ensemble_metric - best_individual
            },
        }
    }
}

/// Statistics about the selection process
#[derive(Debug, Clone)]
pub struct SelectionStats {
    /// Total number of candidate models
    pub n_candidates: usize,
    /// Number of models selected
    pub n_selected: usize,
    /// Best individual model metric
    pub best_individual_metric: f32,
    /// Ensemble metric after selection
    pub ensemble_metric: f32,
    /// Improvement over best individual
    pub improvement: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_config_default() {
        let config = SelectionConfig::default();
        assert_eq!(config.max_models, 0);
        assert!((config.min_improvement - 1e-6).abs() < 1e-9);
        assert_eq!(config.patience, 5);
    }

    #[test]
    fn test_selection_config_builder() {
        let config = SelectionConfig::new()
            .with_max_models(10)
            .with_min_improvement(0.001)
            .with_patience(3);

        assert_eq!(config.max_models, 10);
        assert!((config.min_improvement - 0.001).abs() < 1e-9);
        assert_eq!(config.patience, 3);
    }
}
