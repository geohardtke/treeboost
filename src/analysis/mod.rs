//! Dataset Analysis and Intelligent Mode Selection
//!
//! This module provides TreeBoost's "MRI scan" capability - analyzing dataset
//! characteristics to automatically recommend the optimal boosting mode.
//!
//! # Philosophy
//!
//! Unlike other AutoML tools that waste compute trying every model, TreeBoost
//! **analyzes first, then prescribes**. A 5-second analysis beats a 4-hour search.
//!
//! # How It Works
//!
//! 1. **Subsample**: Work on 10k-50k rows (enough to detect patterns)
//! 2. **Linear Probe**: Quick Ridge regression to measure linear signal
//! 3. **Tree Probe**: Shallow tree on residuals to measure non-linear structure
//! 4. **Feature Analysis**: Categorical ratio, correlations, interactions
//! 5. **Noise Estimation**: Local variance to detect irreducible error
//! 6. **Recommend**: Pick mode with confidence score and full explanation
//!
//! # Example
//!
//! ```ignore
//! use treeboost::analysis::DatasetAnalysis;
//!
//! let analysis = DatasetAnalysis::analyze(&dataset);
//! println!("{}", analysis.report());  // See the full diagnostic
//!
//! let mode = analysis.recommend_mode();
//! let confidence = analysis.confidence();
//! ```
//!
//! # The Statistics We Compute
//!
//! | Metric | Range | What It Measures |
//! |--------|-------|------------------|
//! | `linear_r2` | 0-1 | How much variance a linear model explains |
//! | `tree_gain` | 0-1 | How much trees improve over linear |
//! | `interaction_strength` | 0-1 | Non-additive feature interactions |
//! | `categorical_ratio` | 0-1 | Proportion of categorical features |
//! | `noise_floor` | 0-1 | Estimated irreducible error |
//! | `monotonicity_score` | 0-1 | How monotonic feature-target relationships are |
//!
//! # Decision Logic
//!
//! The recommendation isn't based on single thresholds but on **combinations**:
//!
//! - **LinearThenTree**: High linear signal (R² > 0.3) AND trees add value (gain > 0.1)
//! - **PureTree**: Weak linear signal OR categorical-heavy OR high interactions
//! - **RandomForest**: High noise floor AND need variance reduction

mod stats;
mod probes;
mod analyzer;
mod report;

pub use analyzer::{DatasetAnalysis, AnalysisConfig, Recommendation, Confidence, ModeScores};
pub use report::{AnalysisReport, compact_summary};
pub use stats::{compute_r2, compute_correlation, compute_variance, compute_mean, compute_std};
