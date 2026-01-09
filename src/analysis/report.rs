//! Pretty report generation for dataset analysis
//!
//! Produces human-readable diagnostic reports that explain
//! WHY TreeBoost recommends a particular mode.

use std::fmt;

use super::analyzer::DatasetAnalysis;
use crate::model::BoostingMode;

/// A formatted analysis report
pub struct AnalysisReport<'a> {
    analysis: &'a DatasetAnalysis,
}

impl<'a> AnalysisReport<'a> {
    pub fn from_analysis(analysis: &'a DatasetAnalysis) -> Self {
        Self { analysis }
    }
}

impl<'a> fmt::Display for AnalysisReport<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.analysis;
        let width = 65;

        // Header
        writeln!(f, "┌{}┐", "─".repeat(width))?;
        writeln!(f, "│{:^width$}│", "TreeBoost Dataset Analysis")?;
        writeln!(f, "├{}┤", "─".repeat(width))?;

        // Dataset Info
        let padding = width
            .saturating_sub(54)
            .saturating_sub(digit_count(a.num_rows))
            .saturating_sub(digit_count(a.num_features))
            .saturating_sub(digit_count(a.num_numeric))
            .saturating_sub(digit_count(a.num_categorical));
        writeln!(
            f,
            "│ Samples: {:>7}    Features: {:>3} ({} numeric, {} categorical){} │",
            format_number(a.num_rows),
            a.num_features,
            a.num_numeric,
            a.num_categorical,
            " ".repeat(padding)
        )?;
        writeln!(f, "├{}┤", "─".repeat(width))?;

        // Linear Signal Section
        writeln!(f, "│ {:^63} │", "Linear Signal")?;
        writeln!(
            f,
            "│   R² (linear fit):   {:.2}  {}  {} │",
            a.linear_r2,
            progress_bar(a.linear_r2, 20),
            strength_label(a.linear_r2)
        )?;

        if !a.top_correlations.is_empty() {
            let top_corr = a.top_correlations.first().map(|(_, c)| *c).unwrap_or(0.0);
            writeln!(
                f,
                "│   Top correlation:   {:.2}  {}  {} │",
                top_corr,
                progress_bar(top_corr, 20),
                strength_label(top_corr)
            )?;
        }

        writeln!(
            f,
            "│   Monotonicity:      {:.2}  {}  {} │",
            a.avg_monotonicity,
            progress_bar(a.avg_monotonicity, 20),
            if a.avg_monotonicity > 0.7 {
                "High    "
            } else if a.avg_monotonicity > 0.5 {
                "Moderate"
            } else {
                "Low     "
            }
        )?;

        writeln!(f, "│{:width$}│", "")?;

        // Non-linear Structure Section
        writeln!(f, "│ {:^63} │", "Non-linear Structure")?;
        writeln!(
            f,
            "│   Tree gain:         {:.2}  {}  {} │",
            a.tree_gain,
            progress_bar(a.tree_gain, 20),
            if a.tree_gain > 0.3 {
                "Strong  "
            } else if a.tree_gain > 0.1 {
                "Moderate"
            } else {
                "Weak    "
            }
        )?;
        writeln!(
            f,
            "│   Combined R²:       {:.2}  {}  {} │",
            a.combined_r2,
            progress_bar(a.combined_r2, 20),
            strength_label(a.combined_r2)
        )?;

        writeln!(f, "│{:width$}│", "")?;

        // Data Characteristics Section
        writeln!(f, "│ {:^63} │", "Data Characteristics")?;
        writeln!(
            f,
            "│   Noise floor:       {:.2}  {}  {} │",
            a.noise_floor,
            progress_bar(a.noise_floor, 20),
            if a.noise_floor > 0.4 {
                "High    "
            } else if a.noise_floor > 0.2 {
                "Moderate"
            } else {
                "Low     "
            }
        )?;
        writeln!(
            f,
            "│   Categorical ratio: {:.2}  {}  {:>8} │",
            a.categorical_ratio,
            progress_bar(a.categorical_ratio, 20),
            if a.categorical_ratio > 0.5 {
                "High"
            } else if a.categorical_ratio > 0.2 {
                "Mixed"
            } else {
                "Numeric"
            }
        )?;

        if a.target_is_discrete {
            let target_padding = 28usize.saturating_sub(digit_count(a.target_unique_values));
            writeln!(
                f,
                "│   Target type:       Discrete ({} unique values){} │",
                a.target_unique_values,
                " ".repeat(target_padding)
            )?;
        }

        writeln!(f, "├{}┤", "─".repeat(width))?;

        // Mode Scores
        writeln!(f, "│ {:^63} │", "Mode Scores")?;
        writeln!(
            f,
            "│   PureTree:          {:.2}  {}    │",
            a.mode_scores.pure_tree,
            score_bar(a.mode_scores.pure_tree, a.mode_scores.best_score(), 30)
        )?;
        writeln!(
            f,
            "│   LinearThenTree:    {:.2}  {}    │",
            a.mode_scores.linear_then_tree,
            score_bar(
                a.mode_scores.linear_then_tree,
                a.mode_scores.best_score(),
                30
            )
        )?;
        writeln!(
            f,
            "│   RandomForest:      {:.2}  {}    │",
            a.mode_scores.random_forest,
            score_bar(a.mode_scores.random_forest, a.mode_scores.best_score(), 30)
        )?;

        writeln!(f, "├{}┤", "─".repeat(width))?;

        // Recommendation
        let mode_str = match a.recommendation.mode {
            BoostingMode::PureTree => "PureTree",
            BoostingMode::LinearThenTree => "LinearThenTree",
            BoostingMode::RandomForest => "RandomForest",
        };
        let confidence_str = format!("(Confidence: {})", a.recommendation.confidence.as_str());

        writeln!(
            f,
            "│ RECOMMENDATION: {:<20} {:>24} │",
            mode_str, confidence_str
        )?;
        writeln!(f, "│{:width$}│", "")?;

        // Wrap reasoning text
        let reasoning = &a.recommendation.reasoning;
        for line in wrap_text(reasoning, width - 4) {
            writeln!(f, "│ {:<63} │", line)?;
        }

        // Alternatives
        if !a.recommendation.alternatives.is_empty() {
            writeln!(f, "├{}┤", "─".repeat(width))?;
            writeln!(f, "│ {:^63} │", "Alternatives")?;
            for (mode, reason) in &a.recommendation.alternatives {
                let mode_str = match mode {
                    BoostingMode::PureTree => "PureTree",
                    BoostingMode::LinearThenTree => "LinearThenTree",
                    BoostingMode::RandomForest => "RandomForest",
                };
                writeln!(f, "│   • {}: {} │", mode_str, truncate(reason, 45))?;
            }
        }

        writeln!(f, "└{}┘", "─".repeat(width))?;

        Ok(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn progress_bar(value: f32, width: usize) -> String {
    let filled = ((value * width as f32).round() as usize).min(width);
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn score_bar(value: f32, max_value: f32, width: usize) -> String {
    let normalized = if max_value > 0.0 {
        value / max_value
    } else {
        0.0
    };
    let filled = ((normalized * width as f32).round() as usize).min(width);
    let empty = width - filled;

    let bar_char = if (value - max_value).abs() < 0.01 {
        "█"
    } else {
        "▓"
    };
    format!("{}{}", bar_char.repeat(filled), "░".repeat(empty))
}

fn strength_label(value: f32) -> &'static str {
    if value > 0.7 {
        "Strong  "
    } else if value > 0.4 {
        "Moderate"
    } else if value > 0.1 {
        "Weak    "
    } else {
        "None    "
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn digit_count(n: usize) -> usize {
    format_number(n).len()
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    // Pad lines to width
    lines.iter().map(|l| format!("{:<width$}", l)).collect()
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:<width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Compact single-line summary for logging
pub fn compact_summary(analysis: &DatasetAnalysis) -> String {
    let mode_str = match analysis.recommendation.mode {
        BoostingMode::PureTree => "PureTree",
        BoostingMode::LinearThenTree => "LinearThenTree",
        BoostingMode::RandomForest => "RandomForest",
    };

    format!(
        "TreeBoost Analysis: {} rows, {} features | Linear R²={:.2}, Tree gain={:.2} | \
         Recommended: {} ({})",
        analysis.num_rows,
        analysis.num_features,
        analysis.linear_r2,
        analysis.tree_gain,
        mode_str,
        analysis.recommendation.confidence.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar() {
        assert_eq!(progress_bar(0.0, 10), "░░░░░░░░░░");
        assert_eq!(progress_bar(0.5, 10), "█████░░░░░");
        assert_eq!(progress_bar(1.0, 10), "██████████");
    }

    #[test]
    fn test_wrap_text() {
        let text = "This is a test of the text wrapping function";
        let lines = wrap_text(text, 20);
        assert!(lines.len() > 1);
        assert!(lines.iter().all(|l| l.len() <= 20));
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(500), "500");
        assert_eq!(format_number(5000), "5.0k");
        assert_eq!(format_number(5_000_000), "5.0M");
    }
}
