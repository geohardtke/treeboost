//! Distribution shift detection and monitoring
//!
//! Provides tools for detecting data drift between training and inference,
//! and monitoring CV vs holdout gaps during training.
//!
//! # Overview
//!
//! This module helps identify two types of distribution shift:
//!
//! 1. **Training-time**: Monitoring the gap between cross-validation and holdout
//!    metrics. Large or increasing gaps signal potential data structure issues.
//!
//! 2. **Inference-time**: Comparing feature distributions between training data
//!    and new inference data using metrics like PSI, KL divergence, and KS test.
//!
//! 3. **Incremental learning**: Monitoring distribution changes between training
//!    batches during model updates. Warns when drift may degrade model performance.
//!
//! # Example
//!
//! ```ignore
//! use treeboost::monitoring::{ShiftDetector, PSI};
//!
//! // Create detector from training data
//! let detector = ShiftDetector::from_dataset(&train_data)
//!     .with_metric(PSI::new(10))
//!     .with_thresholds(0.1, 0.25);
//!
//! // Check inference data for drift
//! let result = detector.check(&inference_data);
//! if result.alert == AlertLevel::Critical {
//!     println!("Critical drift: {:?}", result.drifted_features);
//! }
//! ```
//!
//! # Incremental Learning Drift Detection
//!
//! ```ignore
//! use treeboost::monitoring::{IncrementalDriftDetector, check_drift};
//!
//! // Before updating a model, check for drift
//! let drift_result = check_drift(&training_data, &update_data);
//! if drift_result.has_significant_drift() {
//!     println!("Warning: {}", drift_result);
//!     println!("Recommendation: {}", drift_result.recommendation);
//! }
//! ```

mod detector;
pub mod incremental;
mod metrics;
mod tracker;

pub use detector::{AlertLevel, ShiftDetector, ShiftResult};
pub use incremental::{
    check_drift, DriftHistory, DriftRecommendation, IncrementalDriftDetector, IncrementalDriftResult,
};
pub use metrics::{DistributionMetric, JensenShannon, KolmogorovSmirnov, PSI};
pub use tracker::{CVHoldoutTracker, GapRecord, Trend};
